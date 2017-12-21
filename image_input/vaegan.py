
from __future__ import print_function
import tensorflow as tf
# from Distribution import Product , Gaussian , Categorical ,Bernoulli,MeanBernoulli
from tensorflow.core.protobuf import saver_pb2

from ops import batch_normal, lrelu, de_conv, conv2d, fully_connect, instance_norm, residual, deresidual
from utils import save_images
from utils import CelebA
import numpy as np
import cv2
import os
import utils as util
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import Spectrogram

TINY = 1e-8

class vaegan(object):

    con_learning_rate = 0.0001
    con_ecoh = 100
    # point = []

    #build model
    def __init__(self, batch_size, max_epoch, model_path, data,
                 network_type , sample_size, sample_path, log_dir, gen_learning_rate, dis_learning_rate, info_reg_coeff):

        self.batch_size = batch_size
        self.max_epoch = max_epoch

        self.infogan_model_path = model_path[0]

        self.ds_train = data
        self.type = network_type
        self.sample_size = sample_size
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate_gen = gen_learning_rate
        self.learning_rate_dis = dis_learning_rate
        self.info_reg_coeff = info_reg_coeff
        self.log_vars = []
        
        ### Parameters ###
        self.fft_size = 2048  # window size for the FFT
        self.step_size = self.fft_size / 16  # distance to slide along the window (in time)
        self.spec_thresh = 4  # threshold for spectrograms (lower filters out more noise)
        self.lowcut = 500  # Hz # Low cut for our butter bandpass filter
        self.highcut = 15000  # Hz # High cut for our butter bandpass filter

        self.channel = 1

        self.output_size = CelebA().image_size

        self.content_music = tf.placeholder(tf.float32, [self.batch_size, 6880, 1024, 1])

        self.images = tf.placeholder(tf.float32, [self.batch_size, 6880, 1024, 1])
        # self.style_music = tf.placeholder(tf.float32, [self.batch_size, 2, 352800, 1])

        self.z_p = tf.placeholder(tf.float32, [self.batch_size , self.sample_size])
        self.ep = tf.random_normal(shape=[self.batch_size, 2])
        self.y = tf.placeholder(tf.float32, [self.batch_size , 2])
        self.con_dataset = self.con_read('/home2/yyan7582/Sample/SampleDataSet/01Happy_Birthday.wav')


    def build_model_infoGan(self):


        #content encoder

        self.global_train_step = tf.Variable(0, trainable=False)
        self.global_style_train_step = tf.Variable(0, trainable=False)


        self.con_z_mean, self.con_z_sigm, self.cov1,self.cov2, self.sty_encoder  = self.Style_Encode(self.images)

        # KL loss
        self.con_kl_loss = self.con_KL_loss(self.con_z_mean, self.con_z_sigm)

        self.con_x_tilde = self.Style_generate(self.sty_encoder)

        self.style_gen_lost = tf.reduce_mean(tf.square(self.images- self.con_x_tilde))
        self.style_en_lost = self.con_kl_loss



        #encode
        self.z_mean, self.z_sigm, self.con_cov1, self.con_cov2, self.con_cov3 = self.Encode(self.content_music, self.cov1, self.cov2,self.sty_encoder)


        # self.z_x = tf.add(self.z_mean, tf.sqrt(tf.exp(self.z_sigm))*self.ep)

        # self.x_tilde = self.generate(self.res_encoder, reuse=False)

        #the feature
        # self.l_x_tilde, self.D_pro_tilde = self.discriminate(self.x_tilde)

        #for Gan generator
        self.gen_learning_rate = tf.train.exponential_decay(self.learning_rate_gen,self.global_train_step,50,0.95,staircase=True)
        self.Style_gen_learning_rate = tf.train.exponential_decay(self.learning_rate_gen, self.global_style_train_step, 50, 0.95,
                                                            staircase=True)

        self.x_p, self.con_dcov2, self.con_dcov3, self.con_dcov4= self.generate(self.con_cov3)

        # for G Loss
        self.gen_lost = tf.reduce_mean(tf.square(self.content_music - self.x_p))
        self.gen_detail_lost1 = tf.reduce_mean(tf.square(self.con_cov1 - self.con_dcov4))
        self.gen_detail_lost2 = tf.reduce_mean(tf.square(self.con_cov2 - self.con_dcov3))
        self.gen_detail_lost3 = tf.reduce_mean(tf.square(self.con_cov3 - self.con_dcov2))
        self.sty_gen_detail_lost = tf.reduce_mean(tf.square(self.images - self.x_p))
        self.sty_gen_detail_lost2 = self.KL_loss(self.images, self.x_p)
        #self.dtw = self.DTW(self.images,self.x_p)

        # the loss of dis network
        self.l_x,  self.Style_logits = self.discriminate(self.images)

        _, self.G_pro_logits = self.discriminate(self.x_p, True)
        # the defination of loss

        # transfer
        self.result,_,_,_ = self.generate(self.con_cov3, reuse=True)

        #KL loss
        self.kl_loss = self.KL_loss(self.z_mean, self.z_sigm)

        #optimize D

        self.D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.G_pro_logits), logits=self.G_pro_logits))
        # self.D_real_loss = tf.reduce_mean(
        #      tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_pro_tilde), logits=self.D_pro_tilde))
        self.style_tilde_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.Style_logits), logits=self.Style_logits))

        #Optimize G

        self.G_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.G_pro_logits), logits=self.G_pro_logits))
        # self.loss = tf.reduce_mean(tf.square(self.images - self.x_p))
        # self.G_tilde_loss = tf.reduce_mean(
        #      tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.Style_logits), logits=self.Style_logits))


        #for Dis
        self.D_loss = self.D_fake_loss + self.style_tilde_loss

        #For encode
        # self.encode_loss = self.kl_loss/(12*4) - self.LL_loss
        self.encode_loss = self.kl_loss

        #for Gen
        #self.G_fake_loss +
        self.g_loss = self.gen_lost #+ self.sty_gen_detail_lost +self.gen_detail_lost1+self.gen_detail_lost2+self.gen_detail_lost3
        # self.g_loss = - tf.log(self.G_pro_logits)

        self.log_vars.append(("encode_loss", self.encode_loss))
        self.log_vars.append(("generator_loss", self.g_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))
        self.log_vars.append(("style_encode_loss",self.style_en_lost))
        self.log_vars.append(("style_generator_loss", self.style_gen_lost))

        t_vars = tf.trainable_variables()

        # print(len(t_vars))

        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        self.e_vars = [var for var in t_vars if 'e_' in var.name]
        self.sty_e_vars = [var for var in t_vars if 'sty_e' in var.name]
        self.sty_g_vars = [var for var in t_vars if 'sty_gen' in var.name]

        print("d_vars", len(self.d_vars))
        print("g_vars", len(self.g_vars))
        print("e_vars", len(self.e_vars))
        print("sty_e_vars",len(self.sty_e_vars))
        print("sty_g_vars",len(self.sty_g_vars))


        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        # self.saver = tf.train.NewCheckpointReader(self.infogan_model_path)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)


    @property
    def con_train(self):

        opti_con_e = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.style_en_lost,
                                                                                                var_list=self.sty_e_vars)
        opti_con_G = tf.train.AdamOptimizer(learning_rate=self.Style_gen_learning_rate).minimize(self.style_gen_lost,
                                                                                              var_list=self.sty_g_vars)

        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            print("Start .....")
            sess.run(init)
            print("Finish Init....")
            self.saver.save(sess=sess, save_path=self.infogan_model_path)

            batch_num = 0
            point = []
            e = 0
            step = 0
            sample_z = np.random.normal(size=[self.batch_size, self.sample_size])
            # print(np.shape(input_data))
            # get the []
            # z_var = self.z_var.eval()
            print("Start Training...")
            while e <= self.con_ecoh:
                print("echo: ", e)

                max_iter = len(self.ds_train) / self.batch_size - 1
                #np.random.shuffle(self.ds_train)

                while batch_num < len(self.ds_train) / self.batch_size:
                    step = step + 1
                    print("iterate time: ", step)
                    input_data = util.get_Next_Batch(self.ds_train, self.batch_size, max_iter, batch_num)
                    #input_data = np.reshape(input_data, [-1, 2, 44100*20, 1])
                    input_data = np.reshape(input_data, [-1, 6880, 1024, 1])

                    sess.run(opti_con_e, feed_dict={self.images: input_data, self.z_p: sample_z})
                    sess.run(opti_con_G, feed_dict={self.images: input_data, self.global_style_train_step:step, self.z_p: sample_z})

                    E_loss = sess.run(self.style_en_lost, feed_dict={self.images: input_data})

                    G_loss = sess.run(self.style_gen_lost, feed_dict={self.images: input_data})

                    print("Encoder_loss:", E_loss)
                    print("Generator_loss:", G_loss)
                    point.append(G_loss)
                    batch_num += 1

                    # if np.mod(step, 200) == 0:
                    #     tem = np.reshape(input_data[0], [-1, 2])
                    #     write("output_Style_original_{:04d}.wav".format(step), 44100, tem)

                    if np.mod(step, 500) == 0:
                        #plt.plot(range(step), point)
                        #plt.savefig("Style_Gen_cost.jpg")
                        sample_audio, _ = sess.run([self.con_x_tilde, self.sty_encoder],
                                                   feed_dict={self.images: input_data})
                        print(np.shape(sample_audio))
                        sample_audio = np.reshape(sample_audio, [ 6880,1024])
                        recovered_audio_orig = Spectrogram.invert_pretty_spectrogram(sample_audio, fft_size=self.fft_size,
                                                                                     step_size=self.step_size, log=True,
                                                                                     n_iter=10)
                        recovered_audio_orig = recovered_audio_orig * 10000000
                        write("output_Style_generated_{:04d}.wav".format(step), 44100, recovered_audio_orig)
                        self.saver.save(sess, self.infogan_model_path)
                e += 1
                batch_num = 0
            save_path = self.saver.save(sess, self.infogan_model_path)
            np.savetxt("Style_train_Gen_cost.txt",point)
            print("Model saved in file: %s" % save_path)






    #do train
    def train(self):

        opti_D = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_dis).minimize(self.D_loss , var_list=self.d_vars)
        # opti_D = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_dis).minimize(self.D_loss_summary)
        opti_G = tf.train.RMSPropOptimizer(learning_rate=self.gen_learning_rate).minimize(self.g_loss, var_list=self.g_vars)
        opti_e = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_dis).minimize(self.encode_loss, var_list=self.e_vars)


        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            # print("Start Init.....")


            sess.run(init)
            # print("Finish Init....")


            self.saver.save(sess= sess, save_path= self.infogan_model_path)

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            batch_num = 0
            e = 0
            step = 0
            point = []
            print("Start Training...")
#             tem = np.reshape(self.con_dataset,[-1,2])
#             write("output_originalContent_{:04d}.wav".format(step), 44100, tem)
#             tem = np.reshape(self.ds_train[0],[-1,2])
#             write("output_originalStyle_{:04d}.wav".format(step), 44100, tem)
            #realbatch_array = self.ds_train[5]
            #realbatch_array = np.reshape(realbatch_array, [-1, 2, 352800, 1])
            while e <= self.max_epoch:

                max_iter = len(self.ds_train)/self.batch_size - 1
                print("Echo time: ", e)

                np.random.shuffle(self.ds_train)      #-----------

                while batch_num < len(self.ds_train)/self.batch_size:
                    step = step + 1
                    print("iterate time: ", step)
                    
                    realbatch_array = util.get_Next_Batch(self.ds_train,self.batch_size,max_iter,batch_num)
                    realbatch_array = np.reshape(realbatch_array,  [-1, 6880, 1024, 1])
                    # print("Shape of Data", np.shape(realbatch_array))
                    sample_z = np.random.normal(size=[self.batch_size, self.sample_size])


                    # print("Start Optimizing.....")
                    # print("Optimize Encoder....")
                    sess.run(opti_e, feed_dict={self.content_music: self.con_dataset, self.images: realbatch_array})
                    # print("optimized finished....")
                    #optimization D
                    # print("Optimize Dis....")
                    if(step%1==0):
                        sess.run(opti_D, feed_dict={self.images: realbatch_array, self.content_music: self.con_dataset})
                        D_loss = sess.run(self.D_loss, feed_dict={self.images: realbatch_array, self.content_music: self.con_dataset})
                       # print(D_loss,"-----D_lOSS")
                    # print("optimized finished....")
                    #optimizaiton G
                    # print("Optimize Generateor.....")
                    sess.run(opti_G, feed_dict={self.content_music: self.con_dataset,self.images: realbatch_array, self.global_train_step : step}) #, self.z_p: sample_z
                    # print("optimized finished....")

                    # print("Start Writting.....")
                    summary_str = sess.run(summary_op, feed_dict = {self.images:realbatch_array, self.content_music: self.con_dataset, self.z_p: sample_z})#
                    summary_writer.add_summary(summary_str , step)

                    batch_num += 1
                    #print("Finishing Writting....")
                    
                    fake_loss = sess.run(self.g_loss,
                                         feed_dict={self.content_music: self.con_dataset,self.images: realbatch_array, self.z_p: sample_z})

                    print(fake_loss, "-----G_lOSS")
                    point.append(fake_loss)
                    if step%100 == 0:

                        D_loss = sess.run(self.D_loss, feed_dict={self.images: realbatch_array,self.content_music: self.con_dataset})
                        fake_loss = sess.run(self.g_loss, feed_dict={self.content_music: self.con_dataset,self.images: realbatch_array, self.z_p: sample_z})
                        encode_loss = sess.run(self.encode_loss, feed_dict={self.content_music: self.con_dataset,self.images: realbatch_array, self.z_p: sample_z})
                        lr = sess.run(self.gen_learning_rate, feed_dict={self.global_train_step:step})
                        print("EPOCH %d step %d: D: loss = %.7f G: loss=%.7f Encode: loss=%.7f Gen_LearningRate:%.7f" % (e, step, D_loss, fake_loss, encode_loss,lr))


                    if np.mod(step , 500) == 0:
                        #plt.plot(range(step),point)
                        #plt.savefig("Content_Gen_cost.jpg")

                        sample_audio = sess.run(self.x_p, feed_dict={self.content_music: self.con_dataset,self.images: realbatch_array})
                        # sample_audio = np.int32(sample_audio)
                        sample_audio = np.reshape(sample_audio, [ 6880,1024])
                        recovered_audio_orig = Spectrogram.invert_pretty_spectrogram(sample_audio, fft_size=self.fft_size,
                                                                                     step_size=self.step_size, log=True,
                                                                                     n_iter=10)
                        recovered_audio_orig = recovered_audio_orig * 10000000
                        write("output_generated_{:04d}.wav".format(step), 44100, recovered_audio_orig)

                        # save_images(sample_images[0:100] , [10 , 10], '{}/train_{:02d}_{:04d}.png'.format(self.sample_path, e, step))
                        self.saver.save(sess , self.infogan_model_path)


                e += 1
                # print ("Epoch: ",e)
                batch_num = 0

            save_path = self.saver.save(sess , self.infogan_model_path)
            np.savetxt("Train_Gen_cost.txt",point)
            print("Model saved in file: %s" % save_path)

    #do test
    def test(self):

        flag = 0

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config= config) as sess:

            sess.run(init)

            self.saver.restore(sess , self.infogan_model_path)

            flag = 3

            for i in range(0, flag):

                # train_list = CelebA.getNextBatch(self.ds_train, np.inf, i, self.batch_size)
                # input_data = util.get_Next_Batch(self.con_dataset, self.batch_size, i, self.batch_size)
                input_data = self.con_dataset
                # realbatch_array = CelebA.getShapeForData(train_list)

                input_data = self.con_dataset
                input_style = self.ds_train
                np.random.shuffle(input_style)
                input_style_data = input_style[0]
                input_style_data = np.reshape(input_style_data, [-1, 6880, 1024, 1])
                # realbatch_array = CelebA.getShapeForData(train_list)

                # tem = sess.run(self.con_encoder,feed_dict={self.content_music: input_data})

                output_music = sess.run(self.result, feed_dict={self.content_music: input_data,self.images:input_style_data})
                
                output_music = np.reshape(output_music, [ 6880,1024])
                
                recovered_audio_orig = Spectrogram.invert_pretty_spectrogram(output_music, fft_size=self.fft_size,
                                                                             step_size=self.step_size, log=True, n_iter=10)
                recovered_audio_orig = recovered_audio_orig * 10000000

                print("Strat generate Music")
                input_style_data = np.reshape(input_style_data, [ 6880,1024])
                
                recovered_style = Spectrogram.invert_pretty_spectrogram(input_style_data, fft_size=self.fft_size,
                                                                             step_size=self.step_size, log=True, n_iter=10)
                recovered_style = recovered_style * 10000000
                
                write("output_resultTest_Style_{:04d}.wav".format(i), 44100, recovered_style)

                write("output_resultTest_generated_{:04d}.wav".format(i), 44100, recovered_audio_orig)


            print("Test finish!")

    def discriminate(self, x_var, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            conv1 = lrelu(conv2d(x_var, output_dim=64, name='dis_conv1'))
            conv2= lrelu(batch_normal(conv2d(conv1, output_dim=64 , name='dis_conv2'), scope='dis_bn1', reuse=reuse))
            conv3= lrelu(batch_normal(conv2d(conv2, output_dim=64 ,name='dis_conv3'), scope='dis_bn2', reuse=reuse))
            # conv4= lrelu(batch_normal(conv2d(conv3, output_dim=10, name='dis_conv4'), scope='dis_bn3', reuse=reuse))
            middle_conv = conv3
            conv4 = tf.reshape(conv3, [self.batch_size, -1])
            # print(np.shape(conv4))
            #fl = tf.nn.relu(batch_normal(fully_connect(conv4, output_size=512, scope='dis_fully1'),scope='dis_bn4', reuse=reuse))
            output = fully_connect(conv4, output_size=1, scope='dis_fully2')

            return middle_conv, output

    def generate(self, z_var, reuse=False):

        with tf.variable_scope('generator') as scope:

            if reuse == True:
                scope.reuse_variables()

            # d1 = tf.nn.relu(batch_normal(fully_connect(z_var , output_size=64*2*44100, scope='gen_fully1'), scope='gen_bn1', reuse=reuse))
            d2 = tf.reshape(z_var, [self.batch_size, 860, 128, 64])
            d3 = tf.nn.relu(batch_normal(de_conv(d2 , output_shape=[self.batch_size, 1720, 256, 64] , name='gen_deconv2',d_h=2), scope='gen_bn2', reuse=reuse))
            d4 = tf.nn.relu(batch_normal(de_conv(d3, output_shape=[self.batch_size, 3440, 512,64] , name='gen_deconv3'), scope='gen_bn3', reuse=reuse))
            d5 = tf.nn.relu(batch_normal(de_conv(d4, output_shape=[self.batch_size, 6880, 1024, 1], name='gen_deconv4'), scope='gen_bn4', reuse=reuse))
            d6 = de_conv(d5, output_shape=[self.batch_size, 6880, 1024, 1], name='gen_deconv5', d_h=1, d_w=1)


            return tf.nn.relu(d6), d2, d3, d4

    def Encode(self, x, cov1,cov2,cov3):

        with tf.variable_scope('encode') as scope:

            conv1 = tf.nn.relu(batch_normal(conv2d(x, output_dim=64, name='e_c1'), scope='e_bn1'))

            # conv1 = conv1 + cov1
            #print(np.shape(conv1))
            conv2 = tf.nn.relu(batch_normal(conv2d(conv1, output_dim=64, name='e_c2'), scope='e_bn2'))
            # conv2 = conv2 + cov2
            #print(np.shape(conv2))

            conv3 = tf.nn.relu(batch_normal(conv2d(conv2 , output_dim=64, name='e_c3',d_h=2), scope='e_bn3'))
            # conv3 = conv3 + cov3
            #print(np.shape(conv3))

            # conv4 = tf.nn.relu(batch_normal(conv2d(conv3 , output_dim=10, name='e_c4'), scope='e_bn4'))
            conv5 = tf.reshape(conv3, [self.batch_size, 64*128*860])


            z_mean = batch_normal(fully_connect(conv5 , output_size=1, scope='e_f5'), scope='e_bn5')
            z_sigma = batch_normal(fully_connect(conv5, output_size=1, scope='e_f6'), scope='e_bn6')

            return z_mean, z_sigma, conv1, conv2, conv3

    def Style_Encode(self, x):

        with tf.variable_scope('sty_encode') as scope:

            conv1 = tf.nn.relu(batch_normal(conv2d(x, output_dim=64 , name='e_c1'), scope='e_bn1'))
            print(np.shape(conv1))
            conv2 = tf.nn.relu(batch_normal(conv2d(conv1, output_dim=64, name='e_c2'), scope='e_bn2'))
            print(np.shape(conv2))
            conv3 = tf.nn.relu(batch_normal(conv2d(conv2 , output_dim=64, name='e_c3',d_h=2), scope='e_bn3'))
            print(np.shape(conv3))
            # conv4 = tf.nn.relu(batch_normal(conv2d(conv3 , output_dim=10, name='e_c4'), scope='e_bn4'))
            conv5 = tf.reshape(conv3, [self.batch_size, 64*128*860])


            z_mean = batch_normal(fully_connect(conv5 , output_size=1, scope='e_f5'), scope='e_bn5')
            z_sigma = batch_normal(fully_connect(conv5, output_size=1, scope='e_f6'), scope='e_bn6')

            return z_mean, z_sigma,conv1,conv2,conv3


    def Style_generate(self, z_var, reuse=False):

        with tf.variable_scope('sty_generator') as scope:

            d2 = tf.reshape(z_var, [self.batch_size, 860, 128, 64])
            d2 = tf.nn.relu(batch_normal(de_conv(d2 , output_shape=[self.batch_size, 1720, 256, 64] , name='gen_deconv2',d_h=2), scope='gen_bn2'))
            d3 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[self.batch_size, 3440, 512,64] , name='gen_deconv3'), scope='gen_bn3'))
            d4 = tf.nn.relu(batch_normal(de_conv(d3, output_shape=[self.batch_size, 6880, 1024, 1] , name='gen_deconv4'),scope='gen_bn4', reuse=reuse))
            d5 = de_conv(d4, output_shape=[self.batch_size,6880, 1024, 1], name='gen_deconv5', d_h=1, d_w=1)

            return tf.nn.relu(d5)

    def KL_loss(self, mu, log_var):
        return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps

    def NLLNormal(self, pred, target):

        c = -0.5 * tf.log(2 * np.pi)
        multiplier = 1.0 / (2.0 * 1)
        tmp = tf.square(pred - target)
        tmp *= -multiplier
        tmp += c

        return tmp

    def con_KL_loss(self, mu, log_var):
        return -0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu, 2) - tf.exp(log_var))


    def con_read(self, path):
        rate,song = read(path)
        data = Spectrogram.butter_bandpass_filter(song, self.lowcut, self.highcut, rate, order=1)
        dataSet = []
        tem = []
        second = 44100


        print("Start slice...")
        for i in range(0, data.__len__()):
            if i != 0:
                tem.append([data.__getitem__(i)[0], data.__getitem__(i)[1]])
                # print(to_one(data.__getitem__(i)[0]), ",", to_one(data.__getitem__(i)[1]))

            if i % (second * 20) == 0 and i != 0:
                # print np.shape(tem)
                tem = np.mean(tem, axis=1)
                wav_spectrogram = Spectrogram.pretty_spectrogram(tem.astype('float32'), fft_size=self.fft_size,
                                                                 step_size=self.step_size, log=True, thresh=self.spec_thresh)
                dataSet.append(wav_spectrogram)
                tem = []
        print("Slice Finished....")

        # print(np.shape(dataSet))
#         dataSet = np.float32(dataSet)
# 
#         max = np.max(dataSet)
#         min = np.min(dataSet)
#         dataSet = (dataSet - min) / (max - min)

        x = dataSet[0]

        x = np.reshape(x, [-1,6880,1024,1])
        print(np.shape(x),"-----")

        #write("output_originalContent100.wav", 44100, np.reshape(x, [-1, 2]))
        # dataSet = np.reshape(dataSet, [-1, 2, 352800, 1])
        return x
        
    def DTW(self,style, content, window=None, d=lambda x, y: tf.abs(tf.subtract(x, y))):
        style = tf.reshape(style, [882000, 2])
        content = tf.reshape(content, [882000, 2])

        style_input = tf.Variable(tf.zeros([882000, 2], tf.float32))
        content_input = tf.Variable(tf.zeros([882000, 2], tf.float32))

        #print(style_input[0])

        style_input = tf.assign_add(style_input, style)
        #print(style_input[0])
        content_input = tf.assign_add(content_input, content)
        # style_input = tf.TensorArray(dtype=tf.float32, handle=style_input,tensor_array_name=None,flow=1)

        style = tf.reduce_mean(style_input, 1)
        content = tf.reduce_mean(content_input, 1)

        content_array = tf.TensorArray(dtype=tf.float32, size=882000, flow=1)

        style_array = tf.TensorArray(dtype=tf.float32, size=882000, flow=1)

        con = tf.unstack(content)
        sty = tf.unstack(style)

        M, N = len(con), len(sty)
        # M, N = len(content), len(style)
        cost = tf.Variable(tf.ones([M, N], tf.float32))
        cost = tf.reshape(cost, [M, N])
        tem = 882000 * 882000
        cost = tf.TensorArray(dtype=tf.float32, size=tem)

        # cost = tf.make_ndarray(cost)

        content_array = content_array.scatter([0, 882000], content)
        style_array = style_array.scatter([0, 882000], style)

        # temp1 = content_array.unstack(content).read(0)
        # temp1 = content_array.read(0)
        # temp2 = style_array.read(0)
        # temp3 = d(temp1,temp2)

        # temp4 = cost[0,0]


        # cost.write(0,12)
        tem0 = cost.read(0)
        # temp6 = cost[0][0]

        # cost = cost[0,0].assign(d(con[0], sty[0]))

        # cost = tf.scatter_add(cost,[0], [d(content[0], style[0])])
        temcon = content_array.read(0)
        temsty = style_array.read(0)
        dis = d(temcon, temsty)
        cost.write(0, dis)

        for i in range(1, M):
            # cost = tf.scatter_update(cost, [i][0], cost[i - 1, 0] + d(con[i], sty[0]))
            cost.write(i, cost.read(i - 1) + d(content_array.read(0), style_array.read(i)))

        for j in range(1, tem, M):
            # cost = tf.scatter_update(cost, [0][j], cost[0, j-1] + d(con[0], sty[j]))
            cost.write(j, cost.read(j - M) + d(content_array.read(0), style_array.read(j)))
            # print(j)

        for i in range(0, tem - M):
            b = i
            if i % M != 9:
                a = i % M
                temp1 = cost.read(i)
                temp2 = cost.read(i + 1)
                temp3 = cost.read(i + M)
                temp_mid = tf.minimum(temp1, temp2)
                min_value = tf.minimum(temp_mid, temp3)
                new_value = min_value + d(content_array.read(i), style_array.read(i % M))
                index = i + M + 1
                cost.write(index, new_value)

        minimum_cost = cost.read(tem - 1)

        return minimum_cost










import os
import errno
import numpy as np
import scipy
import scipy.misc
import tensorflow as tf
import re
from scipy.io.wavfile import read, write
import Spectrogram

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class CelebA(object):

    def __init__(self):

        self.dataname = "celeba"
        self.dims = 64*64
        self.shape = [64 , 64 , 3]
        self.image_size = 64


### Parameters ###
fft_size = 2048 # window size for the FFT
step_size = fft_size/16 # distance to slide along the window (in time)
spec_thresh = 4 # threshold for spectrograms (lower filters out more noise)
lowcut = 500 # Hz # Low cut for our butter bandpass filter
highcut = 15000 # Hz # High cut for our butter bandpass filter


def read_music(path):
    # song = read(path)
    # data = song.__getitem__(1)
    # data = data.tolist()
    dataSet = []
    tem = []
    temp = []
    second = 44100
    # maxrange = 0
    # minrange = 0

    music_file = []
    file_list = os.listdir(path)
    np.random.shuffle(file_list)
    for file in file_list:
        tem = re.match('.+?\.wav', file)
        if (tem and music_file.__len__() <2):
            music_file.append(path + file)
            print(path, ",", file)



    print ("Strart normalize...")

    print ("Start slice...")
    for music in music_file:
        rate, song = read(music)
        #data = Spectrogram.butter_bandpass_filter(song, lowcut, highcut, rate, order=1)
        data = song


        tem = []
        for i in range(0,data.__len__()):
            if i != 0:
                tem.append([data.__getitem__(i)[0], data.__getitem__(i)[1]])
                # print(to_one(data.__getitem__(i)[0]), ",", to_one(data.__getitem__(i)[1]))


            if i % (second * 20) == 0 and i != 0:
#                 tem = np.mean(tem,axis=1)
#                 wav_spectrogram = Spectrogram.pretty_spectrogram(tem.astype('float32'), fft_size=fft_size,
#                                                                  step_size=step_size, log=True, thresh=spec_thresh)
                dataSet.append(tem)
                tem = []
#                 print(np.shape(dataSet))

        # print (np.shape(dataSet))
        dataSet = np.float32(dataSet)

        max = np.max(dataSet)
        min = np.min(dataSet)
        dataSet = (dataSet - min) / (max - min)
        for i in range(0,len(dataSet)):
            temp.append(dataSet[i])
        dataSet = []
        print(np.shape(temp))


        # maxrange = 0
        # minrange = 0
    print ("Slice Finished....")
    # if (len(temp) % 2 != 0):
    #     temp.pop()
    # write("output_originalStyle100.wav", 44100, np.reshape(temp[2],[-1,2]))
    # x = dataSet[2]
    dataSet = np.reshape(temp,[-1,2,882000,1])
    print (np.shape(dataSet))
    return dataSet

def get_Next_Batch(input_dataSet, batch_size, maxiter_num ,  batch_num):
    if batch_num >= maxiter_num - 1:
        print ("shuffe one time")
        length = len(input_dataSet)
        perm = np.arange(length)
        np.random.shuffle(perm)
    return input_dataSet[(batch_num) * batch_size: (batch_num + 1) * batch_size]



def get_image(image_path , image_size , is_crop=True, resize_w = 64 , is_grayscale = False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w)

def transform(image, npx = 64 , is_crop=False, resize_w = 64):

    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w = resize_w)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])

    return np.array(cropped_image)/127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

# def get_image(image_path, is_grayscale=False):
#     return np.array(inverse_transform(imread(image_path, is_grayscale)))

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def inverse_transform(image):
    return (image + 1.) / 2.

def read_image_list_file(category, is_test):

    path = ''
    skip_num = 0
    if is_test == False:

        skip_num = 1202
        path = "/home/jichao/data/celebA/"

    else:

        skip_num = 2
        path = "/home/jichao/data/celeba_test/"

    list_image = []
    list_label = []
    lines = open(category+"list_attr_celeba.txt")

    li_num = 0

    for line in lines:

        if li_num < skip_num:

            li_num += 1

            continue

        flag = line.split('1 ', 41)[20]
        file_name = line.split(' ', 1)[0]

        #add the image
        list_image.append(path + file_name)

        # print flag
        if flag == ' ':
            #one-hot
            list_label.append([1,0])
        else:
            list_label.append([0,1])

        li_num += 1

    return list_image, list_label

def read_image_list(category):

    filenames = []
    print("list file")
    list = os.listdir(category)



    for file in list:
        tem = re.match('.+?\.jpeg', file)
        if(tem):
            filenames.append(category + "/" + file)
            print(category + "/" + file)
    print("list file ending!")

    length = len(filenames)
    perm = np.arange(length)
    np.random.shuffle(perm)
    filenames = np.array(filenames)
    #filenames = filenames[perm]

    return filenames

def sample_label():

    num = 64
    label_vector = np.zeros((num , 128), dtype=np.float)
    for i in range(0 , num):
        label_vector[i , (i/8)%2] = 1.0
    return label_vector

def log10(x):

  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator




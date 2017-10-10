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
        if (tem and music_file.__len__() < 2):
            music_file.append(path + file)
            print(path, ",", file)



    print ("Strart normalize...")

    print ("Start slice...")
    for music in music_file:
        rate, song = read(music)
        data = Spectrogram.butter_bandpass_filter(song, lowcut, highcut, rate, order=1)


        tem = []
        for i in range(0,data.__len__()):
            if i != 0:
                tem.append([data.__getitem__(i)[0], data.__getitem__(i)[1]])
                # print(to_one(data.__getitem__(i)[0]), ",", to_one(data.__getitem__(i)[1]))


            if i % (second * 20) == 0 and i != 0:
                tem = np.mean(tem,axis=1)
                wav_spectrogram = Spectrogram.pretty_spectrogram(tem.astype('float32'), fft_size=fft_size,
                                                                 step_size=step_size, log=True, thresh=spec_thresh)
                dataSet.append(wav_spectrogram)
                tem = []

    print ("Slice Finished....")
    # if (len(temp) % 2 != 0):
    #     temp.pop()
    # write("output_originalStyle100.wav", 44100, np.reshape(temp[2],[-1,2]))
    # x = dataSet[2]
    dataSet = np.reshape(dataSet,[-1,6880,1024,1])
    print (np.shape(dataSet))
    return dataSet

def get_Next_Batch(input_dataSet, batch_size, maxiter_num ,  batch_num):
    if batch_num >= maxiter_num - 1:
        print ("shuffe one time")
        length = len(input_dataSet)
        perm = np.arange(length)
        np.random.shuffle(perm)
    return input_dataSet[(batch_num) * batch_size: (batch_num + 1) * batch_size]


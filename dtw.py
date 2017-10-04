import tensorflow as tf
import fastdtw as dw
import numpy as np
import sys

content_music = tf.placeholder(tf.float32, [1, 2, 44100, 1])
style = tf.placeholder(tf.float32, [1, 2, 44100, 1])


def DTW(style, content, window=None, d=lambda x, y: tf.abs(x - y)):
    style = tf.reshape(style, [44100, 2])
    content = tf.reshape(content, [44100, 2])

    style = tf.reduce_mean(style,1,keep_dims=True)
    content = tf.reduce_mean(content,1,keep_dims=True)

    con = tf.unstack(content)
    sty = tf.unstack(style)

    # print(content)

    M, N = len(con), len(sty)
    cost = tf.Variable(tf.ones([M,N], tf.float32))


    # cost = cost[0,0].assign(d(con[0], sty[0]))
    cost = tf.scatter_update(cost,[0][0],d(con[0], sty[0]))

    for i in range(1, M):
        cost = tf.scatter_update(cost, [i][0], cost[i - 1, 0] + d(con[i], sty[0])) 

    for j in range(1, N):
        cost = tf.scatter_update(cost, [0][j], cost[0, j-1] + d(con[0], sty[j]))

    for i in range(1, M):
        for j in range(i, N):
            pass




    # dist, cost = dw.fastdtw(style, content)

    return 0


a = DTW(style, content_music)

import tensorflow as tf
import fastdtw as dw
import numpy as np
import sys

content_music = tf.placeholder(tf.float32, [1, 2, 44100, 1])
style = tf.placeholder(tf.float32, [1, 2, 44100, 1])


def DTW(style_m, content, window=None, d=lambda x, y: tf.abs(x - y)):
    style = tf.reshape(style_m, [44100, 2])
    content = tf.reshape(content, [44100, 2])

    style_input = tf.Variable(tf.zeros([44100, 2],tf.float32))
    content_input = tf.Variable(tf.zeros([44100, 2], tf.float32))

    style_input = tf.assign_add(style_input,style)
    content_input = tf.assign_add(content_input,content)

    style = tf.reduce_mean(style_input,1,keep_dims=True)
    content = tf.reduce_mean(content_input,1,keep_dims=True)

    con = tf.unstack(content)
    sty = tf.unstack(style)

    M, N = len(con), len(sty)
    # M, N = len(content), len(style)
    cost = tf.Variable(tf.ones([M,N], tf.float32))

    temp1 = con[0]
    temp2 = sty[0]
    temp3 = d(temp1,temp2)
    temp4 = cost[0,0]
    temp5 = cost[0]
    temp6 = cost[0][0]

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

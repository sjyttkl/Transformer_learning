# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     test
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/8/15
   Description :  仅供 测试使用。不是主要代码，可以忽略
==================================================
"""
__author__ = 'songdongdong'
import tensorflow as tf
import numpy as np
tf.enable_eager_execution() #可以直接打印出 格式
a=tf.constant( [[ 1,  1,  2, 3],[-1,  2,  1, 2],[-2, -1,  3, 1],
                 [-3, -2, -1, 5]],dtype=tf.float32)
b=tf.linalg.band_part(a,2,0)
c=tf.linalg.band_part(a,1,1)
d=tf.linalg.band_part(a,-1,0)

f = tf.tile([1,2,3],[2])
g = tf.tile([[1,2],
             [3,4],
             [5,6]],[2,3])
h = [[1,2,3],[4,5,6]]
i = [[1,0,3],[1,5,1]]

a2=np.array([[1,0,0],[0,1,1]])
a3=np.array([[3,2,3],[4,5,6]])

decoder_input = tf.constant([[1,2],[2,1]],dtype=tf.int32)
y = tf.one_hot(decoder_input, depth=4)
with tf.Session() as sess:
    print(a)
    print("=====b======")
    print(b)
    print("=====c======")
    print(c)
    print("=====d======")
    print(d)
    print(f)
    print(g)
    print(sess.run(tf.equal(h, i)))
    print(sess.run(tf.equal(a2, 1)))
    print(sess.run(tf.where(tf.equal(a2, 1), a2, 1-a3)))
    print("------------one-hot------------------")
    print(sess.run(y))

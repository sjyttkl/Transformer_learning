# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     1. embedding+positions
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/8/5
   Description :  https://mp.weixin.qq.com/s?__biz=MzI0ODcxODk5OA==&mid=2247505978&idx=2&sn=709e15361deb0a153d133ffd58c9e7d9&chksm=e99eebc3dee962d517f7457a3c6458919ec6bae11fef3e723bc1522700adc45689194e1341e8&mpshare=1&scene=1&srcid=&key=25f866a180001e47f6f412f7bcdd1c38236e060763a8bdb0a72a1ad4cccb172d1ba062f4b9f7554c2226018b41e0166a28357e0c7016e75584d403d0797bbfff972c0fcfaa8ea181655927bf6143caaa&ascene=1&uin=MTgwMTM4MzIw&devicetype=Windows+7&version=62060833&lang=zh_CN&pass_ticket=KH6EbXFTICf8MO3W1iDWJP80a0zJWGwTOnQndjBLyFs%3D
==================================================
"""
__author__ = 'songdongdong'

import tensorflow as tf

chinese_embedding = tf.constant([[0.11,0.21,0.31,0.41],
                         [0.21,0.31,0.41,0.51],
                         [0.31,0.41,0.51,0.61],
                         [0.41,0.51,0.61,0.71]],dtype=tf.float32)


english_embedding = tf.constant([[0.51,0.61,0.71,0.81],
                         [0.52,0.62,0.72,0.82],
                         [0.53,0.63,0.73,0.83],
                         [0.54,0.64,0.74,0.84]],dtype=tf.float32)


position_encoding = tf.constant([[0.01,0.01,0.01,0.01],
                         [0.02,0.02,0.02,0.02],
                         [0.03,0.03,0.03,0.03],
                         [0.04,0.04,0.04,0.04]],dtype=tf.float32)

encoder_input = tf.constant([[0,1,2,3],[2,3,0,1]],dtype=tf.int32)


with tf.variable_scope("encoder_input"):
    encoder_embedding_input = tf.nn.embedding_lookup(chinese_embedding,encoder_input) #(4,4) *(2,4) = (2,4,4)
    encoder_embedding_input = encoder_embedding_input + position_encoding #(2,4,4)

if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([encoder_embedding_input]))
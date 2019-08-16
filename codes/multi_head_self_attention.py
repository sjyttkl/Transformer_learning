# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     3. multi-head self attention
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/8/5
   Description :  https://mp.weixin.qq.com/s?__biz=MzI0ODcxODk5OA==&mid=2247505978&idx=2&sn=709e15361deb0a153d133ffd58c9e7d9&chksm=e99eebc3dee962d517f7457a3c6458919ec6bae11fef3e723bc1522700adc45689194e1341e8&mpshare=1&scene=1&srcid=&key=25f866a180001e47f6f412f7bcdd1c38236e060763a8bdb0a72a1ad4cccb172d1ba062f4b9f7554c2226018b41e0166a28357e0c7016e75584d403d0797bbfff972c0fcfaa8ea181655927bf6143caaa&ascene=1&uin=MTgwMTM4MzIw&devicetype=Windows+7&version=62060833&lang=zh_CN&pass_ticket=KH6EbXFTICf8MO3W1iDWJP80a0zJWGwTOnQndjBLyFs%3D
==================================================
"""
__author__ = 'songdongdong'
import tensorflow as tf
import Encoder_Block
import Encoder_input
w_Z = tf.constant([[0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4],
                   [0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)

with tf.variable_scope("encoder_input"):
    encoder_embedding_input = tf.nn.embedding_lookup(Encoder_input.chinese_embedding, Encoder_input.encoder_input)
    encoder_embedding_input = encoder_embedding_input + Encoder_input.position_encoding

with tf.variable_scope("encoder_multi_head_product_attention"):
    encoder_Q = tf.matmul(tf.reshape(encoder_embedding_input, (-1, tf.shape(encoder_embedding_input)[2])), Encoder_Block.w_Q)#(2,4,4) reshape(8,4)==> (8,4)*(4,6) = (8,6)
    encoder_K = tf.matmul(tf.reshape(encoder_embedding_input, (-1, tf.shape(encoder_embedding_input)[2])), Encoder_Block.w_K)#(2,4,4) reshape(8,4)==> (8,4)*(4,6) = (8,6)
    encoder_V = tf.matmul(tf.reshape(encoder_embedding_input, (-1, tf.shape(encoder_embedding_input)[2])), Encoder_Block.w_V)#(2,4,4) reshape(8,4)==> (8,4)*(4,6) = (8,6)

    encoder_Q = tf.reshape(encoder_Q, (tf.shape(encoder_embedding_input)[0], tf.shape(encoder_embedding_input)[1], -1))#(8,6) reshape(2,4,6)
    encoder_K = tf.reshape(encoder_K, (tf.shape(encoder_embedding_input)[0], tf.shape(encoder_embedding_input)[1], -1))#(8,6) reshape(2,4,6)
    encoder_V = tf.reshape(encoder_V, (tf.shape(encoder_embedding_input)[0], tf.shape(encoder_embedding_input)[1], -1))#(8,6) reshape(2,4,6)

    encoder_Q_split = tf.split(encoder_Q, 2, axis=2)#(2,4,3),(2,4,3)
    encoder_K_split = tf.split(encoder_K, 2, axis=2)#(2,4,3),(2,4,3)
    encoder_V_split = tf.split(encoder_V, 2, axis=2)#(2,4,3),(2,4,3)

    encoder_Q_concat = tf.concat(encoder_Q_split, axis=0)#(4,4,3)
    encoder_K_concat = tf.concat(encoder_K_split, axis=0)#(4,4,3)
    encoder_V_concat = tf.concat(encoder_V_split, axis=0)#(4,4,3)

    attention_map = tf.matmul(encoder_Q_concat, tf.transpose(encoder_K_concat, [0, 2, 1]))#(4,4,3) *(4,3,4) = (4,4,4)
    attention_map = attention_map / 8
    attention_map = tf.nn.softmax(attention_map)

    weightedSumV = tf.matmul(attention_map, encoder_V_concat) #(4,4,4) *(4,4,3) = (4,4,3)

    outputs_z = tf.concat(tf.split(weightedSumV, 2, axis=0), axis=2) # [(2,4,3) (2,4,3 )] == (2,4,6)

    outputs = tf.matmul(tf.reshape(outputs_z, (-1, tf.shape(outputs_z)[2])), w_Z)#(8,6)*(6,4) = (8,4)
    outputs = tf.reshape(outputs, (tf.shape(encoder_embedding_input)[0], tf.shape(encoder_embedding_input)[1], -1)) #(8,4) ===>(2,4,4)

import numpy as np
if __name__ == "__main__":
    with tf.Session() as sess:
        #     print(sess.run(encoder_Q))
        #     print(sess.run(encoder_Q_split))
        # print(sess.run(weightedSumV))
        # print(sess.run(outputs_z))
        print(sess.run(outputs))


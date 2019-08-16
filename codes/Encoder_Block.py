# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     2、Encoder Block
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/8/5
   Description :  https://mp.weixin.qq.com/s?__biz=MzI0ODcxODk5OA==&mid=2247505978&idx=2&sn=709e15361deb0a153d133ffd58c9e7d9&chksm=e99eebc3dee962d517f7457a3c6458919ec6bae11fef3e723bc1522700adc45689194e1341e8&mpshare=1&scene=1&srcid=&key=25f866a180001e47f6f412f7bcdd1c38236e060763a8bdb0a72a1ad4cccb172d1ba062f4b9f7554c2226018b41e0166a28357e0c7016e75584d403d0797bbfff972c0fcfaa8ea181655927bf6143caaa&ascene=1&uin=MTgwMTM4MzIw&devicetype=Windows+7&version=62060833&lang=zh_CN&pass_ticket=KH6EbXFTICf8MO3W1iDWJP80a0zJWGwTOnQndjBLyFs%3D
==================================================
"""
__author__ = 'songdongdong'
import tensorflow as tf
import Encoder_input as en_input

w_Q = tf.constant([[0.1,0.2,0.3,0.4,0.5,0.6],
                         [0.2,0.3,0.4,0.5,0.6,0.7],
                         [0.3,0.4,0.5,0.5,0.7,0.8],
                         [0.4,0.5,0.6,0.7,0.8,0.9]],dtype=tf.float32)

w_K = tf.constant([[0.08,0.18,0.28,0.38,0.48,0.58],
                         [0.18,0.28,0.38,0.48,0.58,0.68],
                         [0.28,0.38,0.48,0.58,0.68,0.78],
                         [0.38,0.48,0.58,0.68,0.78,0.88]],dtype=tf.float32)

w_V = tf.constant([[0.12,0.22,0.32,0.42,0.52,0.62],
                         [0.22,0.32,0.42,0.52,0.62,0.72],
                         [0.32,0.42,0.52,0.62,0.72,0.82],
                         [0.42,0.52,0.62,0.72,0.82,0.92]],dtype=tf.float32)

with tf.variable_scope("encoder_scaled_dot_product_attention"):

    encoder_Q = tf.matmul(tf.reshape(en_input.encoder_embedding_input, (-1, tf.shape(en_input.encoder_embedding_input)[2])),w_Q) #(2,4,4) reshape(8,4)==> (8,4)*(4,6) = (8,6)
    encoder_K = tf.matmul(tf.reshape(en_input.encoder_embedding_input, (-1, tf.shape(en_input.encoder_embedding_input)[2])), w_K)#(2,4,4) reshape(8,4)==> (8,4)*(4,6) = (8,6)
    encoder_V = tf.matmul(tf.reshape(en_input.encoder_embedding_input, (-1, tf.shape(en_input.encoder_embedding_input)[2])), w_V)#(2,4,4) reshape(8,4)==> (8,4)*(4,6) = (8,6)

    encoder_Q = tf.reshape(encoder_Q, (tf.shape(en_input.encoder_embedding_input)[0], tf.shape(en_input.encoder_embedding_input)[1], -1)) #(8,4) reshape(2,4,6)
    encoder_K = tf.reshape(encoder_K, (tf.shape(en_input.encoder_embedding_input)[0], tf.shape(en_input.encoder_embedding_input)[1], -1))#(8,4) reshape(2,4,6)
    encoder_V = tf.reshape(encoder_V, (tf.shape(en_input.encoder_embedding_input)[0], tf.shape(en_input.encoder_embedding_input)[1], -1))#(8,4) reshape(2,4,6)

    attention_map = tf.matmul(encoder_Q, tf.transpose(encoder_K, [0, 2, 1]))#(2,4,6)* (2,6,4) = (2,4,4)
    attention_map = attention_map / 8
    attention_map = tf.nn.softmax(attention_map)

if __name__ == "__main__":
    with tf.Session() as sess:
        print("2.Encoder_Block.........")
        sess.run(tf.global_variables_initializer())
        print(sess.run(attention_map))
        # print(sess.run(en_input.encoder_first_sa_output))
# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     4. Decoder_Block
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/8/15
   Description :  Decoder_block属于   decoder开始阶段
==================================================
"""
__author__ = 'songdongdong'

import tensorflow as tf
#Decoder阶段的输入是：
english_embedding = tf.constant([[0.51,0.61,0.71,0.81],
                         [0.61,0.71,0.81,0.91],
                         [0.71,0.81,0.91,1.01],
                         [0.81,0.91,1.01,1.11]],dtype=tf.float32)


position_encoding = tf.constant([[0.01,0.01,0.01,0.01],
                         [0.02,0.02,0.02,0.02],
                         [0.03,0.03,0.03,0.03],
                         [0.04,0.04,0.04,0.04]],dtype=tf.float32)

decoder_input = tf.constant([[1,2],[2,1]],dtype=tf.int32)
#咱们先来实现这部分的代码，masked attention map的计算过程：
# 先定义下权重矩阵，同encoder一样，定义成常数：
w_Q_decoder_sa = tf.constant([[0.15,0.25,0.35,0.45,0.55,0.65],
                   [0.25,0.35,0.45,0.55,0.65,0.75],
                   [0.35,0.45,0.55,0.65,0.75,0.85],
                   [0.45,0.55,0.65,0.75,0.85,0.95]],dtype=tf.float32)

w_K_decoder_sa = tf.constant([[0.13,0.23,0.33,0.43,0.53,0.63],
                   [0.23,0.33,0.43,0.53,0.63,0.73],
                   [0.33,0.43,0.53,0.63,0.73,0.83],
                   [0.43,0.53,0.63,0.73,0.83,0.93]],dtype=tf.float32)

w_V_decoder_sa = tf.constant([[0.17,0.27,0.37,0.47,0.57,0.67],
                   [0.27,0.37,0.47,0.57,0.67,0.77],
                   [0.37,0.47,0.57,0.67,0.77,0.87],
                   [0.47,0.57,0.67,0.77,0.87,0.97]],dtype=tf.float32)
w_Z_decoder_sa = tf.constant([[0.1,0.2,0.3,0.4],
                   [0.1,0.2,0.3,0.4],
                   [0.1,0.2,0.3,0.4],
                   [0.1,0.2,0.3,0.4],
                   [0.1,0.2,0.3,0.4],
                   [0.1,0.2,0.3,0.4]],dtype=tf.float32)
with tf.variable_scope("decoder_input"):
    decoder_embedding_input = tf.nn.embedding_lookup(english_embedding,decoder_input)#(2,2,4)
    decoder_embedding_input = decoder_embedding_input + position_encoding[0:tf.shape(decoder_embedding_input)[1]]#(2,2,4)  +(2,4) =(2,2,4)
#随后，计算添加mask之前的attention map：
with tf.variable_scope("decoder_sa_block"):
    decoder_Q = tf.matmul(tf.reshape(decoder_embedding_input, (-1, tf.shape(decoder_embedding_input)[2])),
                          w_Q_decoder_sa)#(2,2,4)==>(4,4) * (4,6) = (4,6)
    decoder_K = tf.matmul(tf.reshape(decoder_embedding_input, (-1, tf.shape(decoder_embedding_input)[2])),
                          w_K_decoder_sa)#(2,2,4)==>(4,4) * (4,6) = (4,6)
    decoder_V = tf.matmul(tf.reshape(decoder_embedding_input, (-1, tf.shape(decoder_embedding_input)[2])),
                          w_V_decoder_sa)#(2,2,4)==>(4,4) * (4,6) = (4,6)

    decoder_Q = tf.reshape(decoder_Q, (tf.shape(decoder_embedding_input)[0], tf.shape(decoder_embedding_input)[1], -1))#(4,6) ==> (2,2,6)
    decoder_K = tf.reshape(decoder_K, (tf.shape(decoder_embedding_input)[0], tf.shape(decoder_embedding_input)[1], -1))#(4,6) ==> (2,2,6)
    decoder_V = tf.reshape(decoder_V, (tf.shape(decoder_embedding_input)[0], tf.shape(decoder_embedding_input)[1], -1))#(4,6) ==> (2,2,6)

    decoder_Q_split = tf.split(decoder_Q, 2, axis=2) #[(2,2,3) (2,2,3)]
    decoder_K_split = tf.split(decoder_K, 2, axis=2) #[(2,2,3) (2,2,3)]
    decoder_V_split = tf.split(decoder_V, 2, axis=2) #[(2,2,3) (2,2,3)]

    decoder_Q_concat = tf.concat(decoder_Q_split, axis=0)#(4,2,3)
    decoder_K_concat = tf.concat(decoder_K_split, axis=0)#(4,2,3)
    decoder_V_concat = tf.concat(decoder_V_split, axis=0)#(4,2,3)

    decoder_sa_attention_map_raw = tf.matmul(decoder_Q_concat, tf.transpose(decoder_K_concat, [0, 2, 1]))#(4,2,3) * (4,3,2) == (4,2,2)
    decoder_sa_attention_map = decoder_sa_attention_map_raw / 8 #(4,2,2)
    # 随后，对attention map添加mask
    # tf.linalg.LinearOperatorLowerTriangular
    diag_vals = tf.ones_like(decoder_sa_attention_map[0,:,:])#(2,2) #这里我们首先构造一个全1的矩阵diag_vals，这个矩阵的大小同attention map
    # tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() #随后通过tf.contrib.linalg.LinearOperatorTriL方法把上三角部分变为0
    tril = tf.linalg.band_part(diag_vals, -1, 0) #(2,2) #上三角为 0
    masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(decoder_sa_attention_map)[0],1,1]) #（1,2,2） ==>(4,1,1)  = (4,2,2) ,这里第一个维度是4 维度，每个 2*2，都是一个 三角矩阵
    paddings = tf.ones_like(masks) * (-2 ** 32 + 1)#(4,2,2) 要加mask的地方，不能赋值为0，而是需要赋值一个很小的数，这里为-2^32 + 1
    decoder_sa_attention_map = tf.where(tf.equal(masks,0),paddings,decoder_sa_attention_map) #(4,2,2)
    decoder_sa_attention_map = tf.nn.softmax(decoder_sa_attention_map)#(4,2,2)
    #
    weightedSumV = tf.matmul(decoder_sa_attention_map, decoder_V_concat) #（4,2,2） * （4,2,3） = （4,2,3）

    decoder_outputs_z = tf.concat(tf.split(weightedSumV, 2, axis=0), axis=2)#[(2,2,3),(2,2,3)] = (2,2,6)

    decoder_sa_outputs = tf.matmul(tf.reshape(decoder_outputs_z, (-1, tf.shape(decoder_outputs_z)[2])), w_Z_decoder_sa) #(2,2,6) => (4,6) *(6,4) = (4,4)

    decoder_sa_outputs = tf.reshape(decoder_sa_outputs,
                                    (tf.shape(decoder_embedding_input)[0], tf.shape(decoder_embedding_input)[1], -1)) #(4,4) ==> (2,2,4)
if __name__ == "__main__":
    with tf.Session() as sess:
        print(sess.run(decoder_sa_outputs))
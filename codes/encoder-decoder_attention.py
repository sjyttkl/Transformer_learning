# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     5. encoder-decoder_attention
    email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/8/15
   Description :   encoder-decoder_attention  层，属于 transormer中， decoder阶段
==================================================
"""
__author__ = 'songdongdong'
import multi_head_self_attention  #这是  encoder 阶段
import tensorflow as tf
import Decoder_Block

w_Q_decoder_sa2 = tf.constant([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                               [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                               [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                               [0.5, 0.6, 0.7, 0.8, 0.9, 1]], dtype=tf.float32)

w_K_decoder_sa2 = tf.constant([[0.18, 0.28, 0.38, 0.48, 0.58, 0.68],
                               [0.28, 0.38, 0.48, 0.58, 0.68, 0.78],
                               [0.38, 0.48, 0.58, 0.68, 0.78, 0.88],
                               [0.48, 0.58, 0.68, 0.78, 0.88, 0.98]], dtype=tf.float32)

w_V_decoder_sa2 = tf.constant([[0.22, 0.32, 0.42, 0.52, 0.62, 0.72],
                               [0.32, 0.42, 0.52, 0.62, 0.72, 0.82],
                               [0.42, 0.52, 0.62, 0.72, 0.82, 0.92],
                               [0.52, 0.62, 0.72, 0.82, 0.92, 1.02]], dtype=tf.float32)

w_Z_decoder_sa2 = tf.constant([[0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4],
                               [0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)

# encoder层的输出：（2,4,4)
encoder_outputs = multi_head_self_attention.outputs

with tf.variable_scope("decoder_encoder_attention_block"):
    decoder_sa_outputs = Decoder_Block.decoder_sa_outputs + Decoder_Block.decoder_embedding_input #(2,2,4)  #保留原有数据信息

    encoder_decoder_Q = tf.matmul(tf.reshape(decoder_sa_outputs, (-1, tf.shape(decoder_sa_outputs)[2])),
                                  w_Q_decoder_sa2)#(4,4) *(4,6) = (4.6)
    #接下来，就是encoder-decoder了，这里跟multi-head attention相同，但是需要注意的一点是，我们这里想要做的是，
    # 计算decoder的每个阶段的输入和encoder阶段所有输出的attention，所以Q的计算通过decoder对应的embedding计算，
    # 而K和V通过encoder阶段输出的embedding来计算：
    encoder_decoder_K = tf.matmul(tf.reshape(encoder_outputs, (-1, tf.shape(encoder_outputs)[2])), w_K_decoder_sa2)#（2,4,4）==》(8,4) *(4,6)= (8,6)
    encoder_decoder_V = tf.matmul(tf.reshape(encoder_outputs, (-1, tf.shape(encoder_outputs)[2])), w_V_decoder_sa2)#（2,4,4）==》(8,4) *(4,6)= (8,6)

    encoder_decoder_Q = tf.reshape(encoder_decoder_Q,
                                   (tf.shape(Decoder_Block.decoder_embedding_input)[0], tf.shape(Decoder_Block.decoder_embedding_input)[1], -1))#(4,6) ==(2,2,6)
    encoder_decoder_K = tf.reshape(encoder_decoder_K, (tf.shape(encoder_outputs)[0], tf.shape(encoder_outputs)[1], -1))#(8,6) ==(2,4,6)
    encoder_decoder_V = tf.reshape(encoder_decoder_V, (tf.shape(encoder_outputs)[0], tf.shape(encoder_outputs)[1], -1))#(8,6) ==(2,4,6)

    encoder_decoder_Q_split = tf.split(encoder_decoder_Q, 2, axis=2) #[(2,2,3),(2,2,3])
    encoder_decoder_K_split = tf.split(encoder_decoder_K, 2, axis=2)#[(2,4,3),(2,4,3])
    encoder_decoder_V_split = tf.split(encoder_decoder_V, 2, axis=2)#[(2,4,3),(2,4,3])

    encoder_decoder_Q_concat = tf.concat(encoder_decoder_Q_split, axis=0)#(4,2,3)
    encoder_decoder_K_concat = tf.concat(encoder_decoder_K_split, axis=0)#(4,4,3)
    encoder_decoder_V_concat = tf.concat(encoder_decoder_V_split, axis=0)#(4,4,3)

    encoder_decoder_attention_map_raw = tf.matmul(encoder_decoder_Q_concat,
                                                  tf.transpose(encoder_decoder_K_concat, [0, 2, 1])) #(4,2,3) * (4,3,4) = (4,2,4)
    encoder_decoder_attention_map = encoder_decoder_attention_map_raw / 8#(4,2,4)

    encoder_decoder_attention_map = tf.nn.softmax(encoder_decoder_attention_map)##(4,2,4)

    weightedSumV = tf.matmul(encoder_decoder_attention_map, encoder_decoder_V_concat)#(4,2,4) *(4,4,3) = （4,2,3）

    encoder_decoder_outputs_z = tf.concat(tf.split(weightedSumV, 2, axis=0), axis=2)#[（2,2,3）,（2,2,3）] = (2,2,6)

    encoder_decoder_outputs = tf.matmul(
        tf.reshape(encoder_decoder_outputs_z, (-1, tf.shape(encoder_decoder_outputs_z)[2])), w_Z_decoder_sa2)#(2,2,6) ==>(4,6) * (6,4) = (4,4)

    encoder_decoder_attention_outputs = tf.reshape(encoder_decoder_outputs, (
    tf.shape(Decoder_Block.decoder_embedding_input)[0], tf.shape(Decoder_Block.decoder_embedding_input)[1], -1)) #(2,2,4)

    encoder_decoder_attention_outputs = encoder_decoder_attention_outputs + decoder_sa_outputs #(2,2,4)  #保留原有数据信息

    # todo :add BN
    W_f = tf.constant([[0.2, 0.3, 0.5, 0.4],
                       [0.2, 0.3, 0.5, 0.4],
                       [0.2, 0.3, 0.5, 0.4],
                       [0.2, 0.3, 0.5, 0.4]])

    decoder_ffn_outputs = tf.matmul(
        tf.reshape(encoder_decoder_attention_outputs, (-1, tf.shape(encoder_decoder_attention_outputs)[2])), W_f) #(2,2,4)-->(4,4)*(4,4) = (4,4)
    decoder_ffn_outputs = tf.reshape(decoder_ffn_outputs, (
    tf.shape(encoder_decoder_attention_outputs)[0], tf.shape(encoder_decoder_attention_outputs)[1], -1))#(4,4) -->(2,2,4)

    decoder_outputs = decoder_ffn_outputs + encoder_decoder_attention_outputs #(2,2,4) 这里保留 encoder - decoder阶段的信息
    # todo :add BN
    #=====全连接层及最终输出===========
    print("=========全连接层及最终输出===========")
    W_final = tf.constant([[0.2, 0.3, 0.5, 0.4],
                           [0.2, 0.3, 0.5, 0.4],
                           [0.2, 0.3, 0.5, 0.4],
                           [0.2, 0.3, 0.5, 0.4]])

    logits = tf.matmul(tf.reshape(decoder_outputs, (-1, tf.shape(decoder_outputs)[2])), W_final) #(4,4)*(4,4) = (4,4)
    logits = tf.reshape(logits, (tf.shape(decoder_outputs)[0], tf.shape(decoder_outputs)[1], -1))#(4,4) == >(2,2,4）

    logits = tf.nn.softmax(logits)#(2,2,4)

    y = tf.one_hot(Decoder_Block.decoder_input, depth=4) #（2,2）==>(2,2,4)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
with tf.Session() as sess:
    # print(sess.run(decoder_outputs))
    print(sess.run(train_op))
#-*- coding:utf-8 -*-
import os,time,datetime
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')

from utils.summary_utils import variable_summaries
from utils.conv_utils import conv_output_length


def build_ds2(args,
              inputX,
              cell_fn,
              activation_fn,
              seqLengths,
              time_major=True,
              is_training=True,
              convlayers=3,
              use_dropout=False,
              rnnlayers=7,
              use_bidirection_rnn=True):
    '''

    :param args:
    :param inputX:
    :param cell_fn:
    :param activation_fn:
    :param seqLengths:
    :param time_major:
    :param is_training:
    :param convlayers:
    :param use_dropout:
    :param rnnlayers:
    :param use_bidirection_rnn:
    :return: fc tensor with time_major,[maxtime,batchsize,feat]
    '''

    if time_major:
        x=tf.transpose(inputX,[1,2,0])
    else:
        x = inputX
    layer = tf.expand_dims(x,-1)

    conv2dlayer_stride=[[1,2,2,1],[1,2,1,1],[1,2,1,1]]
    conv2dlayer_filter=[(41,11,1,32),(21,11,32,32),(21,11,32,96)]

    for ci in range(convlayers):
        layer_filter = tf.get_variable('conv{}_filter'.format(ci+1),
                                       shape=conv2dlayer_filter[ci],
                                       dtype=tf.float32)
        layer = tf.nn.conv2d(layer, layer_filter,
                             conv2dlayer_stride[ci],padding='SAME',
                             name='conv{}'.format(ci+1))

        layer = tf.layers.batch_normalization(layer,training=is_training,name='conv{}/bn'.format(ci+1))
        if use_dropout:
            layer = tf.contrib.layers.dropout(layer,keep_prob=args.keep_prob,is_training=is_training)

    # shape adjust , rnn input must be [maxtime,batch_size,...] for time major
    # from 4d tensor to 3d tensor
    layer4d = tf.transpose(layer,[2,0,1,3])
    shape = layer4d.get_shape().as_list()
    layer = tf.reshape(layer4d, [shape[0] or -1, shape[1] or -1 , shape[2]*shape[3]])

    #rnn layers

    if use_bidirection_rnn:
        for ri in range(rnnlayers):
            forward_cell = cell_fn(args.hiddens, activation=activation_fn)
            backward_cell = cell_fn(args.hiddens, activation=activation_fn)
            layer, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=forward_cell,
                                                       cell_bw=backward_cell,
                                                       inputs=layer,
                                                       sequence_length=seqLengths,
                                                       dtype=tf.float32,
                                                       time_major=True,
                                                       scope='bidrnn{}'.format(ri+1))
            of,ob = layer
            ofb = tf.concat([of,ob],2)
            shape=ofb.get_shape().as_list()
            ofb2 = tf.reshape(ofb,[shape[0] or -1, shape[1] or -1, 2, int(shape[2]/2)])#or only concat it
            layer = tf.reduce_sum(ofb2,2)
            # batch norm
            layer = tf.layers.batch_normalization(layer, training=is_training, name='bidrnn{}/bn'.format(ri + 1))
            if use_dropout:
                layer = tf.contrib.layers.dropout(layer, keep_prob=args.keep_prob, is_training=is_training)
    else:
        for ri in range(rnnlayers):
            rnncell = cell_fn(args.hiddens, activation=activation_fn)
            layer, _ = tf.nn.dynamic_rnn(cell=rnncell,
                                         inputs=layer,
                                         seqLengths=seqLengths,
                                         dtype=tf.float32,
                                         time_major=True,
                                         scope='drnn{}'.format(ri+1))

            # batch norm
            layer = tf.layers.batch_normalization(layer, training=is_training, name='drnn{}/bn'.format(ri + 1))
            if use_dropout:
                layer = tf.contrib.layers.dropout(layer, keep_prob=args.keep_prob, is_training=is_training)
    # fc layer
    fc = tf.layers.dense(layer, args.classes, name='fc_out')

    return fc


class DeepSpeech2(object):
    def __init__(self, args, name='deepspeech2', server=None, device=None):
        self.args = args
        self.name = name
        if args.mode == 'train':
            self.is_training = True
        else:
            self.is_training = False
        if args.rnncell =='gru':
            self.cell_fn = tf.contrib.rnn.GRUCell
        elif args.rnncell == 'lstm':
            self.cell_fn = tf.contrib.rnn.BasicLSTMCell
        elif args.rnncell == 'rnn':
            self.cell_fn = tf.contrib.rnn.BasicRNNCell
        else:
            raise Exception('rnn cell not supported:{}'.format(args.rnncell))

        if args.activation == 'relu':
            self.activation = tf.nn.relu
        elif args.activation == 'tanh':
            self.activation = tf.nn.tanh
        else:
            raise Exception('activation not supported:{}'.format(args.activation))

        self.server = server
        self.device = device
        self.build_graph()

    def add_input_layer(self):
        # according to DeepSpeech2 paper, input is the spectrogram power of audio, but if you like,
        # you can also use mfcc feature as the input.

        # input x
        self.inputX = tf.placeholder(tf.float32, shape=(None, self.args.batch_size, self.args.features), name='inputX')

        # ground truth,y
        self.targetIxs = tf.placeholder(tf.int64, name='targetIxs')
        self.targetVals = tf.placeholder(tf.int32, name='targetVals')
        self.targetShape = tf.placeholder(tf.int64, name='targetShape')
        self.targetY = tf.SparseTensor(self.targetIxs, self.targetVals, self.targetShape)
        self.input_seq_length = tf.placeholder(tf.int32, shape=(self.args.batch_size), name='inputSeqLengths')
        # the length of all the output sequences
        self.target_seq_length = tf.placeholder(dtype=tf.int32,
                                                shape=(self.args.batch_size),
                                                name='target_seq_length')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)


    def add_ds2_layer(self):
        self.logits3d = build_ds2(self.args, self.inputX, self.cell_fn, self.activation, self.input_seq_length,
                                  is_training=self.is_training,
                                  convlayers=self.args.conv_layers,
                                  rnnlayers=self.args.rnn_layers,
                                  use_dropout=self.args.keep_prob > 0.0,  # less 0 don't use dropout
                                  use_bidirection_rnn=self.args.use_bidirectional_rnn == 'yes')

    def add_backward_path(self):
        # TBD: will move to trainer or tester class later!!!
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.targetY, self.logits3d, self.input_seq_length))
        self.var_op = tf.global_variables()
        self.var_trainable_op = tf.trainable_variables()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if self.args.grad_clip == -1:
            # not apply gradient clipping
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.args.learning_rate).minimize(self.loss,
                                                                                    global_step=self.global_step)
        else:
            # apply gradient clipping
            gradients = tf.gradients(self.loss, self.var_trainable_op)
            if self.args.use_summary == 'yes':
                tf.summary.scalar('global_gradient_norm', tf.global_norm(gradients))

            grads, _ = tf.clip_by_global_norm(gradients, self.args.grad_clip)
            opti = tf.train.AdamOptimizer(self.args.learning_rate)
            with tf.control_dependencies(update_ops):
                self.train_op = opti.apply_gradients(zip(grads, self.var_trainable_op), global_step=self.global_step)

        self.predictions = tf.to_int32(
            tf.nn.ctc_beam_search_decoder(self.logits3d, self.input_seq_length, merge_repeated=False)[0][0])
        self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=True))
        self.initial_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        # summary
        if self.args.use_summary == 'yes':
            tf.summary.scalar('train loss', self.loss)
            tf.summary.scalar('error rate', self.errorRate)
            for param in tf.trainable_variables():
                variable_summaries(param, name_scope=param.name)

            self.summary_op = tf.summary.merge_all()


    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if self.device is None:
                self.add_input_layer()
                self.add_ds2_layer()
                self.add_backward_path()
            else:
                with tf.device(self.device):
                    self.add_input_layer()
                    self.add_ds2_layer()
                    self.add_backward_path()


    def compute_conv_output_length(self, input_length):
        return conv_output_length(input_length=input_length,
                                  filter_size=1,
                                  padding='SAME',
                                  stride=2)

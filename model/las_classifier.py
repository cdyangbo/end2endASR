#-*- coding:utf-8 -*-
import os,time,datetime
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod

from seq2seq_layer import TSClassifier, Listener, Speller

class LAS_Classifier(TSClassifier):
    '''
    a Listen,attention,spell class for an encoder decoder system
    '''

    def __init__(self, conf, output_dim, name=None):
        '''
        LAS constructor
        Args:
            conf: The classifier configuration
            output_dim: the classifier output dimension
            name: the classifier name
        '''

        super(LAS_Classifier, self).__init__(conf, output_dim, name)

        self.is_chief = True

        # create the listener
        self.encoder = Listener(conf, name=self.scope.name + '/encoder')

        # create the speller
        self.decoder = Speller(conf, self.output_dim, name=self.scope.name + '/decoder')

    def _get_outputs(self, inputs, input_seq_length, targets=None,
                     target_seq_length=None, is_training=False, time_major=False):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor for time minor,else [t,b,f]
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] vector
            is_training: whether or not the network is in training mode
            time_major: True or False for [t,b,f] or [b,t,f]

        Returns:
            A pair containing:
                - output logits
                - the output logits sequence lengths as a vector
        '''

        # add input noise
        std_input_noise = float(self.conf['std_input_noise'])
        if is_training and std_input_noise > 0:
            noisy_inputs = inputs + tf.random_normal(inputs.get_shape(), stddev=std_input_noise)
        else:
            noisy_inputs = inputs

        # compute the high level features
        hlfeat = self.encoder(inputs=noisy_inputs,
                              sequence_lengths=input_seq_length,
                              is_training=is_training,
                              time_major=time_major)

        # prepend a sequence border label to the targets to get the encoder
        # inputs, the label is the last label
        # <sos> and <eos> all equal output_dim -1 ?
        batch_size = int(targets.get_shape()[0])
        s_labels = tf.constant(self.output_dim - 1,
                               dtype=tf.int32,
                               shape=[batch_size, 1])

        encoder_inputs = tf.concat([s_labels, targets], 1)

        # compute the output logits
        logits, _ = self.decoder(
            hlfeat=hlfeat,
            encoder_inputs=encoder_inputs,
            initial_state=self.decoder.zero_state(batch_size),
            first_step=True,
            is_training=is_training,
            time_major=time_major)

        return logits, target_seq_length + 1

    def compute_conv_output_length(self, input_length, padding='SAME', stride =1):
        '''
        :param input_length:
        :param padding:
        :param stride:
        :return:
        '''

        return input_length

    def build_graph(self, task_index=0, time_major=True):
        # self.max_input_length = dispenser.max_input_length

        if self.server is not None:
            cluster = tf.train.ClusterSpec(self.server.server_def.cluster)
            self.is_chief = task_index == 0
            num_replicas = len(cluster.as_dict()['worker'])
            device = tf.train.replica_device_setter(cluster=cluster, worker_device='/job:worker/task:%d' % task_index)
        else:
            self.is_chief = True
            num_replicas = 1
            device = None


        # create the graph
        self.graph = tf.Graph()

        # define the placeholders in the graph

        with self.graph.as_default():
            batch_size = self.conf['batch_size']
            with tf.device(device):
                # create the inputs placeholder, time_major, [time,batch_size,input_dim]
                # max_input_length and batch_size should be None for efficiency compute?
                if time_major:
                    self.inputX = tf.placeholder(dtype=tf.float32,
                                                 shape=[self.max_input_length, batch_size, self.input_dim],
                                                 name='inputX')
                else:
                    self.inputX = tf.placeholder(dtype=tf.float32,
                                                 shape=[batch_size, self.max_input_length, self.input_dim],
                                                 name='inputX')

                # reference labels
                self.targetY = tf.placeholder(dtype=tf.int32,
                                              shape=[batch_size, self.max_target_length],
                                              name='targetY')

                # the length of all the input sequences
                self.input_seq_length = tf.placeholder(dtype=tf.int32,
                                                       shape=[batch_size],
                                                       name='input_seq_length')

                # the length of all the output sequences
                self.target_seq_length = tf.placeholder(dtype=tf.int32,
                                                        shape=[batch_size],
                                                        name='target_seq_length')


                # compute the training outputs of the classifier
                self.trainlogits, self.logit_seq_length = self.__call__(inputs=self.inputX,
                                                                input_seq_length=self.input_seq_length,
                                                                targets=self.targetY,
                                                                target_seq_length=self.target_seq_length,
                                                                is_training=True,
                                                                time_major=time_major)



                # create variables for validation
                with tf.variable_scope('validation'):
                    self.vallogits, self.val_logit_seq_length = self.__call__(inputs=self.inputX,
                                                                      input_seq_length=self.input_seq_length,
                                                                      targets=self.targetY,
                                                                      target_seq_length=self.target_seq_length,
                                                                      is_training=False,
                                                                      time_major=time_major)
                    self.val_loss = self.compute_ce_loss(self.targetY,
                                                         self.vallogits,
                                                         self.val_logit_seq_length,
                                                         self.target_seq_length,
                                                         time_major=time_major)

                    self.predictions = tf.to_int32(tf.nn.ctc_greedy_decoder(self.vallogits, self.val_logit_seq_length, merge_repeated=False)[0][0])

                # a variable to hold the amount of steps already taken
                self.global_step = tf.get_variable(name='global_step',
                                                   shape=[],
                                                   dtype=tf.int32,
                                                   initializer=tf.constant_initializer(0),
                                                   trainable=False)

                with tf.variable_scope('train'):
                    # create the optimizer
                    if self.conf['optimizer'] == 'adam':
                        optimizer = tf.train.AdamOptimizer(self.conf['learning_rate'])
                    elif self.conf['optimizer'] == 'nm':#nestrov mometum
                        optimizer = tf.train.MomentumOptimizer(self.conf['learning_rate'], 0.99, use_nesterov=True)
                    else:
                        raise Exception('unsupported optimizer func' + self.conf['optimizer'])

                    # compute the loss
                    self.loss = self.compute_ce_loss(self.targetY,
                                                     self.trainlogits,
                                                     self.logit_seq_length,
                                                     self.target_seq_length,
                                                     time_major=time_major)

                    # compute the gradients
                    gradients, variables = zip(*optimizer.compute_gradients(self.loss))

                    with tf.variable_scope('clip'):
                        # clip the gradients,to test clib_by_gloabal_norm
                        if self.conf['write_summary'] == 'yes':
                            tf.summary.scalar('global_gradients_norm', tf.global_norm(gradients))
                        gradients, _ = tf.clip_by_global_norm(gradients,self.conf['grad_clip'] or 5)


                    # opperation to apply the gradients
                    apply_gradients_op = optimizer.apply_gradients(grads_and_vars=zip(gradients,variables),
                                                                   global_step=self.global_step,
                                                                   name='apply_gradients')

                    # all remaining operations with the UPDATE_OPS GraphKeys
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                    # create an operation to update the gradients, the batch_loss
                    # and do all other update ops
                    self.update_op = tf.group(*([apply_gradients_op] + update_ops),name='update')

                if self.conf['write_summary'] == 'yes':
                    # create the summaries for visualisation
                    tf.summary.scalar('validation loss', self.val_loss)
                    tf.summary.scalar('train loss', self.loss)
                    tf.summary.scalar('learning rate', self.learning_rate)

                    # create a histogram for all trainable parameters
                    for param in tf.trainable_variables():
                        tf.summary.histogram(param.name, param)

                    self.train_summary_writer=tf.summary.FileWriter(os.path.join(self.conf['savepath'],'log/train',self.conf['model']))
                    self.summary_op = tf.summary.merge_all()

                # create the saver
                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5, keep_checkpoint_every_n_hours=1)



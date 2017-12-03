#-*- coding:utf-8 -*-
import os,time,datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import tqdm

import sys
sys.path.append('../')

from utils.wer import wer

class Seq2Seq(object):
    def __init__(self,rnn_size, n_layers,
                 input_char2idx_dict, encoder_embedding_dim,
                 output_char2idx_dict, decoder_embedding_dim,
                 batch_size, modeldir, grad_clip):
        '''

        :param rnn_size:
        :param n_layers:
        :param input_char2idx_dict:
        :param encoder_embedding_dim:
        :param output_char2idx_dict:
        :param decoder_embedding_dim:
        :param batch_size:
        :param modeldir:
        :param sess:
        :param grad_clip:
        '''

        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.grad_clip = grad_clip
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.batch_size = batch_size


        self.modeldir = modeldir
        self.learning_rate=1e-3

        self.x_char2idx_dict = input_char2idx_dict #  char/word ->idx
        self.y_char2idx_dict = output_char2idx_dict # char/word ->idx
        self.x_idx2char_dict = {v:k for k,v in self.ix_char2idx_dict.iteritems()}
        self.y_idx2char_dict = {v:k for k,v in self.y_char2idx_dict.iteritems()}

        self.register_symbols()
        self.build_graph()


    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.add_input_layer()
            self.add_encoder_layer()
            self.add_decoder_layer()
            self.add_backward_path()

            self.saver = tf.train.Saver(tf.global_variables(),
                                        max_to_keep=5,
                                        keep_checkpoint_every_n_hours=1)

    def add_input_layer(self):
        self.inputX = tf.placeholder(tf.int32, [self.batch_size, None])
        self.targetY = tf.placeholder(tf.int32,[self.batch_size, None])
        self.input_seq_len = tf.placeholder(tf.int32, [self.batch_size])
        self.output_seq_len = tf.placeholder(tf.int32,[self.batch_size])

        self.global_step = tf.get_variable(name='global_step',
                                           shape=[],
                                           dtype=tf.int32,
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)


    def add_encoder_layer(self):
        with tf.variable_scope('encoder'):
            encoder_embedding = tf.get_variable('encoder_embed', [len(self.x_char2idx_dict), self.encoder_embedding_dim],
                                                tf.float32, tf.random_uniform_initializer(-1.0,1.0))

            cell = tf.nn.rnn_cell.MultiRNNCell([self.rnn_cell() for _ in range(self.n_layers)])

            embed = tf.nn.embedding_lookup(params=encoder_embedding, ids=self.inputX)

            self.encoder_out, self.encoder_state = tf.nn.dynamic_rnn(cell=cell,
                                                                     inputs=embed,
                                                                     sequence_length=self.input_seq_len,
                                                                     dtype=tf.float32)

    def add_decoder_layer(self):
        with tf.variable_scope('decoder'):
            decoder_embedding = tf.get_variable('decoder_embedding',
                                                [len(self.y_char2idx_dict), self.decoder_embedding_dim],
                                                tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
            decoder_cell = self._attention()
            embed = tf.nn.embedding_lookup(params=decoder_embedding, ids=self.processed_decoder_input())

            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=embed,
                sequence_length=self.output_seq_len,
                time_major=False)

            initial_state = decoder_cell.zero_state(self.batch_size,tf.float32).clone(cell_state=self.encoder_state)

            training_decoder=tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=training_helper,
                initial_state=initial_state,
                output_layer=Dense(len(self.y_char2idx_dict)))

            training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder,
                impute_finished=True,
                maximum_iterations=tf.reduce_max(self.output_seq_len))

            self.train_logits = training_decoder_outputs.rnn_out

        # prediction
        with tf.variable_scope('decoder',reuse=True):

            decoder_cell = self._attention(reuse=True)
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=tf.get_variable('decoder_embedding'),
                start_tokens=tf.tile(tf.constant([self._y_sos],dtype=tf.int32),[self.batch_size]),
                end_token=self._y_eos)

            initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_state)

            pred_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=pred_helper,
                initial_state=initial_state,
                output_layer=Dense(len(self.y_char2idx_dict),_reuse=True))

            pred_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=pred_decoder,
                impute_finished=True,
                maximum_iterations= 2*tf.reduce_max(self.input_seq_len))

            self.pred_logits = pred_decoder_outputs.sample_id


    def add_backward_path(self):
        masks = tf.sequence_mask(self.output_seq_len, tf.reduce_max(self.output_seq_len),dtype=tf.float32)
        # cross-enthopy loss
        self.loss = tf.contrib.seq2seq.sequence_loss(
            logitis=self.train_logits,
            targets=self.targetY,
            weights=masks)

        variables = tf.trainable_variables()
        gradients = tf.gradients(self.loss, variables)
        clip_gradients,_= tf.clib_by_global_norm(gradients,self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(clip_gradients,variables),global_step=self.global_step)

    def rnn_cell(self, reuse=False, rnn_type='lstm' ):
        return tf.nn.rnn_cell.LSTMCell(self.rnn_size,
                                       initializer=tf.orthogonal_initializer(),
                                       reuse=reuse)

    def _attention(self, reuse=False):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=self.rnn_size,
            memory=self.self.encoder_out,
            memory_sequence_length=self.input_seq_len)

        cell = tf.nn.rnn_cell.MultiRNNCell([self.rnn_cell(reuse=reuse) for _ in range(self.n_layers)])
        return tf.contrib.seq2seq.AttentionWrapper(cell=cell,
                                                   attention_mechanism=attention_mechanism,
                                                   attention_layer_size=self.rnn_size)

    def processed_decoder_input(self):
        # remove last char
        main = tf.stided_slice(self.targetY, [0,0],[self.batch_size,-1],[1,1])
        decoder_input=tf.concat([tf.fill([self.batch_size,1],self._y_sos),main],1)
        return decoder_input


    def register_symbols(self):
        self._x_sos = self.x_char2idx_dict['<SOS>']
        self._x_eos = self.x_char2idx_dict['<EOS>']
        self._x_pad = self.x_char2idx_dict['<PAD>']
        self._x_unk = self.x_char2idx_dict['<UNK>']


        self._y_sos = self.y_char2idx_dict['<SOS>']
        self._y_eos = self.y_char2idx_dict['<EOS>']
        self._y_pad = self.y_char2idx_dict['<PAD>']
        self._y_unk = self.y_char2idx_dict['<UNK>']


    def fit(self,epocs=20):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction =  1.0
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(graph=self.classifier.graph, config=config) as sess:
            self.initModelParams(sess)
            for epoch in range(epocs):
                self.run_train_epoch(sess, epoch, self.batch_size)


    def run_train_epoch(self, sess, epoch, batch_size=16, sort_samples=True):
        tq = tqdm.tqdm(self.train_batchgen.iterate_train(batch_size, sort_by_duration=False, shuffle=sort_samples,
                                                         allow_smaller_final_batch=False),
                       desc='Epoch %i' % epoch,
                       total=self.train_batchgen.get_batch_num(batch_size),
                       dynamic_ncols=True)
        for i, batch in enumerate(tq):
            feed_dict = self.get_feed_dict(batch)
            _, step, loss = sess.run([self.train_op, self.global_step,self.loss],
                                     feed_dict=feed_dict)
            tq.set_postfix(global_step=str(step), loss=loss)

            if step % 100:
                print('val 10 batchs loss')

        self.saveModelParams(sess, step=step)

        self.run_validate_epoch(sess, epoch, batch_size, True)

    def run_validate_epoch(self,sess,epoch,batch_size=16,decode=False):
        batch_num = self.dev_batchgen.get_batch_num(batch_size)

        self.log('Epoch:{},begin validation, batch_num:{}'.format(epoch,batch_num))

        total_err = 0
        total_num = 0

        val_batchs = self.dev_batchgen.iterate_validation(batch_size,allow_smaller_final_batch=False)
        for i, batch in enumerate(val_batchs):
            feed_dict = self.get_feed_dict(batch)
            pre, y = sess.run([self.pred_logits, self.classifier.targetY], feed_dict=feed_dict)

            pred_strip = [strip_zeros(s.tolist()) for s in pre]

            if decode:
                e,n=self.decode(y, pred_strip, i, batch_size)
                total_err += e
                total_num += n

        avg_cer = float(total_err*1.0 / total_num)

        self.log('Epoch:{},end {} validation,  cer:{}'.format(epoch, batch_num, avg_cer))

        # idx decode
        def decode(self, y, pred, batch_no, batch_size, debuglines=1, logfile='decode_dev.txt'):

            ground_truth = ''
            predstr = ''

            reallines = min(debuglines, len(ys))

            tn = te = 0
            self.log('validation batch:{},'.format(batch_no), logfile=logfile)
            for i in range(reallines):
                ground_truth = ' '.join([self.y_idx2char_dict[k].encode('utf-8') for k in y[i]
                predstr = ' '.join([self.y_idx2char_dict[k].encode('utf-8') for k in pred[i]
                e, n = wer(ground_truth.split(), predstr.split(), False)
                tn += n
                te += e
                self.log('Truth :' + ground_truth, logfile=logfile)
                self.log('Decode:' + predstr + ',cer={}'.format(e * 1.0 / n), logfile=logfile)
            return te, tn


def strip_zeros(a):
    if not isinstance(a,list):
        a = a.tolist()
    return [c for c in a if c != 0]



class Seq2SeqDataGen():
    def __init__(self,batch_size,pad_x,pad_y):
        self.batch_size=batch_size
        self.pad_x = pad_x
        self.pad_y = pad_y

    def pad_sentence(self, sentence_batch, pad_idx):
        padded_seqs=[]
        seq_lens=[]
        max_seq_lens = max([len(s) for s in sentence_batch])
        for s in sentence_batch:
            padded_seqs.append(s + [pad_idx]*(max_seq_lens-len(s)))
            seq_lens.append(len(s))
        return padded_seqs,seq_lens


    def next_batch(self,x,y):

        for i in range(0,len(x)-len(x)%self.batch_size, self.batch_size):
            xb = x[i:i+self.batch_size]
            yb = y[i:i+self.batch_size]
            px,xlens = self.pad_sentence(xb,self.pad_x)
            py,ylens = self.pad_sentence(yb,self.pad_y)

            yield (np.array(px),np.array(py),np.array(xlens),np.array(ylens))


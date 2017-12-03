#-*- coding:utf-8 -*-
'''
modified from nabu
created on 2017.11.28
@author: yangbo
'''
from __future__ import print_function
import os,time,datetime,math
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
import json
import argparse
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, wait
import random,tqdm

import sys
sys.path.append('../')

from utils.vocab_utils import loadvocabulary,output_to_sequence
from utils.wer import wer,cer
from utils.misc import log

from model.model_factory import model_factory
from preprocess.data_generator import DataGenerator

class Trainer(object):
    '''General class outlining the training environment of a classifier.'''
    __metaclass__ = ABCMeta

    def __init__(self,args,name,vocabfile=None):
        '''
        :param args:
        :param name:
        :param vocabfile:
        '''
        self.name = name
        self.args = args
        self.write_summary = args.use_summary == 'yes'

        if args.mode == 'train':
            self.is_training = True
        else:
            self.is_training = False

        self.time_major = True

        self.char_map, self.index_map = loadvocabulary(vocabfile or 'vocab.json',mode='alphabet')

        self.server = None
        self.device = None
        self.is_chief = self.args.task_index == 0
        self.workers = len(self.args.ws_hosts.split(','))

    def start_server(self):
        # distribute
        if self.args.ps_hosts != '' and self.args.ws_hosts != '':
            ps_hosts = self.args.ps_hosts.split(',')
            worker_hosts = self.args.ws_hosts.split(',')
            cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0 if self.args.job_name == 'ps' else self.args.gpu_fraction
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            #config.gpu_options.visible_device_list=[self.args.taskindex]
            self.server = tf.train.Server(cluster,
                                          job_name=self.args.job_name,
                                          task_index=self.args.task_index,
                                          config=config)

            if self.args.job_name == 'ps':  # ps server
                self.server.join()
            else:  # worker
                self.device = tf.train.replica_device_setter(worker_device='/job:worker/task:%d' % self.args.task_index,
                                                             cluster=cluster)

    def build_model(self, save_model_per_iters=500, val_loss_per_iters=100):
        self.classifier = model_factory(args=self.args, server=self.server, device=self.device)
        self.save_model_per_iters = save_model_per_iters
        self.val_loss_per_iters = val_loss_per_iters

        if self.write_summary:
            if self.is_training:
                self.train_summary_writer = tf.summary.FileWriter(
                    os.path.join(self.args.savepath,'log/train',self.args.model),self.classifier.graph)
            else:
                self.train_summary_writer = tf.summary.FileWriter(
                    os.path.join(self.args.savepath, 'log/test', self.args.model), self.classifier.graph)


    def init_batch_gen(self,train_jsons=[], dev_jsons=[],major_time=False,fit_samples=500):
        if len(train_jsons)>0:
            self.train_batchgen = DataGenerator(train_jsons,major_time = major_time)
            self.train_batchgen.fit_train(fit_samples)

        if len(dev_jsons)>0:
            self.dev_batchgen = DataGenerator(dev_jsons,major_time = major_time)
            self.dev_batchgen.fit_train(fit_samples)

        if len(train_jsons) == 0 and len(dev_jsons) == 0 :
            raise  Exception('no json desc files.')

    def get_feed_dict(self, batch):
        feed_dict = {}
        inputx = batch['x']
        sparsey = batch['sparsey']
        seqlengths = batch['input_lengths']

        max_input_seq_length = max(seqlengths)
        #print(max_input_seq_length)
        seqlens = [self.classifier.compute_conv_output_length(i) for i in seqlengths]

        target_ixs,target_vals,target_shape = sparsey

        feed_dict[self.classifier.inputX] = inputx
        feed_dict[self.classifier.targetIxs] = target_ixs
        feed_dict[self.classifier.targetVals] = target_vals
        feed_dict[self.classifier.targetShape] = target_shape
        feed_dict[self.classifier.seqLengths] = seqlens


        return feed_dict

    def train(self):

        self.train_costs = []
        self.val_costs = []
        self.cer_costs = []
        self.wer_costs = []

        if self.server is None:
            self.train_single()
        else:
            self.train_multi_replicas()

    def train_single(self):

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.args.gpu_fraction or 1.0
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True



        with tf.Session(graph=self.classifier.graph,config=config) as sess:
            self.initModelParams(sess)

            for epoch in range(self.args.epochs):
                self.run_train_epoch(sess,epoch+self.args.initial_epoch,self.args.batch_size)

    def train_multi_replicas(self):

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.args.gpu_fraction or 1.0
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        sv = tf.train.Supervisor(graph=self.classifier.graph,
                                 is_chief=self.is_chief,
                                 logdir=self.args.savepath,
                                 init_op= self.classifier.global_step,#tf.global_variables_initializer(),
                                 global_step=self.classifier.global_step,
                                 #saver=self.classifier.saver,
                                 #summary_op=self.classifier.summary_op,
                                 #save_model_secs=600
                                 )

        with sv.prepare_or_wait_for_session(self.server.target) as sess:
            sv.start_queue_runners(sess)

            self.initModelParams(sess, True)

            for epoch in range(self.args.epochs):
                self.run_train_epoch(sess, epoch+self.args.initial_epoch, self.args.batch_size,sv=sv)

        sv.stop()

    def run_train_epoch(self, sess, epoch, batch_size=16, sort_samples=True,sv=None):
        if sort_samples:
            shuffle = epoch !=0
            sortagrad = epoch == 0
        else:
            shuffle = True
            sortagrad = False

        if sv is not None and sv.should_stop():
            return

        tq = tqdm.tqdm(self.train_batchgen.iterate_train(batch_size,sort_by_duration=sortagrad,shuffle=shuffle,allow_smaller_final_batch=False),
                       desc='Epoch %i'%epoch,
                       total=self.train_batchgen.get_batch_num(batch_size),
                       dynamic_ncols=True)
        for i, batch in enumerate(tq):
            #if sv is not None and sv.should_stop():
            #    break

            feed_dict = self.get_feed_dict(batch)

            if self.write_summary:
                _, step, summary, loss, er,seqlens = sess.run([self.classifier.train_op,self.classifier.global_step,
                                                               self.classifier.summary_op,self.classifier.loss,
                                                               self.classifier.errorRate,self.classifier.seqLengths],
                                                              feed_dict=feed_dict)
                self.train_summary_writer.add_summary(summary,step)
            else:
                _, step, loss, er,seqlens= sess.run([self.classifier.train_op, self.classifier.global_step,
                                                     self.classifier.loss,self.classifier.errorRate,self.classifier.seqLengths],
                                                    feed_dict=feed_dict)

            max_input_seq_length = max(seqlens)

            self.train_costs.append(loss)


            # validation one  batch loss,not decode it
            if step % self.val_loss_per_iters == 0:
                val_cost = self.run_validate_loss_epoch(sess, batch_size)
                self.val_costs.append([step, val_cost])
                tq.set_postfix(refresh=False, step=str(step), acc=er, loss=loss,valoss=val_cost)
            else:
                tq.set_postfix(refresh=False, step=str(step), acc=er, loss=loss)
                #print('epoch:{},iters:{},global_step:{},loss:{},acc:{}'.format(epoch,i,step,loss,er))

            # save model params and loss
            if (step+1) % self.save_model_per_iters == 0:
                self.saveModelParams(sess, step=step)
                self.saveTrainCosts(self.train_costs, self.val_costs, self.cer_costs, self.wer_costs)
                self.train_costs=[]
                self.val_costs=[]
                self.cer_costs=[]
                self.wer_costs=[]


        # decode val dataset after one epoch
        if self.is_chief and epoch > 2:
            decode_cost,decode_cer, decode_wer = self.run_validate_decode_epoch(sess, epoch, batch_size)
            self.val_costs.append([step, decode_cost])
            self.cer_costs.append([step, decode_cer])
            self.wer_costs.append([step, decode_wer])

    def run_validate_loss_epoch(self,sess,batch_size=16,batch_num=1):

        avg_loss = 0.0

        real_batch_num = min(batch_num, self.dev_batchgen.get_batch_num(batch_size))


        val_batchs = self.dev_batchgen.iterate_validation(batch_size,
                                                          max_iters=real_batch_num,
                                                          shuffle=True,
                                                          allow_smaller_final_batch=False)

        for i, batch in enumerate(val_batchs):
            feed_dict = self.get_feed_dict(batch)
            loss = sess.run(self.classifier.loss,feed_dict=feed_dict)
            avg_loss += loss

        avg_loss /= real_batch_num

        #self.log('Epoch:{},end {} validation, avg_loss:{}, cer:{}'.format(epoch, real_val_batch_num, avg_loss, avg_cer))
        return avg_loss

    def run_validate_decode_epoch(self, sess, epoch, batch_size=16):

        batch_num = self.dev_batchgen.get_batch_num(batch_size)

        self.log('Epoch:{},begin validation, batch_num:{}'.format(epoch, batch_num))

        total_err = total_num = 0
        tchar_err = tchar_num = 0

        avg_loss = 0.0
        avg_cer = -1
        avg_wer = -1

        val_batchs = self.dev_batchgen.iterate_validation(batch_size,
                                                          shuffle=False,
                                                          allow_smaller_final_batch=False)
        real_batch_num = 0
        for i, batch in enumerate(val_batchs):
            real_batch_num += 1
            feed_dict = self.get_feed_dict(batch)
            loss, pre, y = sess.run([self.classifier.loss,
                                     self.classifier.predictions,
                                     self.classifier.targetY],
                                    feed_dict=feed_dict)
            avg_loss += loss

            we, wn, ce, cn = self.decode(y, pre, i, batch_size)
            total_err += we
            total_num += wn
            tchar_err += ce
            tchar_num += cn
            if total_num != 0:
                avg_wer = float(total_err * 1.0 / total_num)
            if tchar_num != 0:
                avg_cer = float(tchar_err * 1.0 / tchar_num)

        avg_loss /= real_batch_num

        self.log('Epoch:{},end {} validation, avg_loss:{}, cer:{},wer:{}'.format(epoch, real_batch_num, avg_loss, avg_cer,avg_wer))
        return avg_loss,avg_cer,avg_wer

    def saveTrainCosts(self, train_costs, val_costs, cer_costs, wer_costs):
        if self.is_chief:
            costfilename = os.path.join(self.args.savepath, 'costs.npz')
            if os.path.exists(costfilename):
                costs = np.load(costfilename)
                train_costs = costs['train'].tolist() + train_costs
                val_costs = costs['validation_with_step'].tolist() + val_costs
                cer_costs = costs['cer_with_step'].tolist() + cer_costs
                wer_costs = costs['wer_with_step'].tolist() + wer_costs

            np.savez(costfilename, train=train_costs,
                     validation_with_step=val_costs,
                     cer_with_step=cer_costs,
                     wer_with_step=wer_costs)


    def saveModelParams(self,sess, step):
        if self.is_chief:
            print('===========save model params =================')
            check_point_path=os.path.join(self.args.savepath, self.name+'.ckpt')
            self.classifier.saver.save(sess, check_point_path, global_step=step)
        else:
            print('===========slave  need no save model params =================')

    def initModelParams(self,sess,is_cluster=False):
        if self.is_chief:
            print('===========master initModelParams=================')
            if self.args.restoremodel == 'yes':
                ckpt = tf.train.get_checkpoint_state(self.args.savepath)
                if ckpt and ckpt.model_checkpoint_path:
                    self.classifier.saver.restore(sess, ckpt.model_checkpoint_path)
                    print('success to restore model params from ', ckpt.model_checkpoint_path)
                else:
                    print('fail to restore model params from ', self.args.savepath)
                    if not is_cluster:
                        sess.run(tf.global_variables_initializer())
            else:
                print('new model params from ')
                sess.run(tf.global_variables_initializer())
        else:
            print('=======wait master to initModelParams=================')

    #greedy decode
    def decode(self, y, pred, batch_no, batch_size, debuglines =1, logfile='decode_dev.txt'):
        ys = output_to_sequence(y, self.index_map)
        preds = output_to_sequence(pred, self.index_map)

        ground_truth = ''
        predstr =''

        reallines = min(debuglines, len(ys))

        twordn = tworde = 0
        tcharn = tchare = 0
        self.log('validation batch:{},'.format(batch_no),logfile=logfile)
        for i in range(reallines):
            ground_truth = ys[i].encode('utf-8')
            predstr = preds[i].encode('utf-8')
            we,wn = wer(ground_truth.split(), predstr.split(), False)
            ce,cn = cer(ground_truth, predstr, False)
            twordn += wn
            tworde += we
            tcharn += cn
            tchare += ce
            self.log('Truth=>:'+ground_truth, logfile=logfile)
            self.log('ASR===>:'+predstr+',wer:{},cer:{}'.format(we*1.0/wn,ce*1.0/cn),logfile=logfile)
        return tworde, twordn, tchare, tcharn


    def log(self,info, logfile='trainlog.txt'):
        log(info,logfilename = logfile, savepath = self.args.savepath)


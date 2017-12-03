#-*- coding:utf-8 -*-
import argparse
import os
import sys
import tensorflow as tf
sys.path.append('./')

from utils.misc import saveArgs
from trainer.trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='command line for run_train')
    parser.add_argument('-rc','--rnncell', choices=['gru','lstm','rnn'], default='gru', type= str,help='rnncell')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='rnncell')
    parser.add_argument('-n', '--hiddens', default=1024, type=int, help='rnncell')
    parser.add_argument('-f', '--features', choices=[13,39,81,161], default=81, type=int, help='rnncell')
    parser.add_argument('-c', '--classes', default=31, type=int, help='rnncell')
    parser.add_argument('-rl', '--rnn_layers', default=7, type=int, help='rnncell')
    parser.add_argument('-cl', '--conv_layers', default=3, type=int, help='rnncell')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='rnncell')
    parser.add_argument('-a', '--activation', choices=['relu', 'tanh', 'sigmod'], default='relu', type=str, help='rnncell')
    parser.add_argument('-o', '--optimizer', default='adam', type=str, help='rnncell')
    parser.add_argument('-lr', '--learning_rate', default=2e-5, type=float, help='rnncell')
    parser.add_argument('-k', '--keep_prob', default=0.9, type=float, help='rnncell')
    parser.add_argument('-gc', '--grad_clip', default=1.0, type=float, help='rnncell')
    parser.add_argument('-m', '--mode', default='train', type=str, help='rnncell')
    parser.add_argument('-r', '--restoremodel', default='yes', type=str, help='rnncell')
    parser.add_argument('-bn', '--batchnorm', default='yes', type=str, help='rnncell')
    parser.add_argument('-p', '--epochs', default=10, type=int, help='rnncell')
    parser.add_argument('-i', '--initial_epoch', default=0, type=int, help='rnncell')
    parser.add_argument('-t', '--trainfiles', nargs='+', required=True, type=str, help='rnncell')
    parser.add_argument('-d', '--devfiles', nargs='+', required=True, type=str, help='rnncell')
    parser.add_argument('-s', '--savepath', required=True, type=str, help='rnncell')

    parser.add_argument('-gf', '--gpu_fraction',default=1.0, type=float, help='rnncell')
    parser.add_argument('-md', '--model', default='ds2', type=str, help='rnncell')
    parser.add_argument('-ub', '--use_bidirectional_rnn', default='yes', type=str, help='rnncell')
    parser.add_argument('-us', '--use_summary', default='yes', type=str, help='rnncell')
    parser.add_argument('-v', '--vocabfile', default='conf/alphabet.txt', type=str, help='rnncell')


    #distributed training
    parser.add_argument('--ps_hosts', default='', type=str, help='comma-separated list of hostname:port pairs')
    parser.add_argument('--ws_hosts', default='', type=str, help='comma-separated list of hostname:port pairs')
    parser.add_argument('--job_name', default='ps', choices=['ps', 'worker'], type=str, help='ps or worker')
    parser.add_argument('--task_index', default=0, type=int, help='index of task within the job')
    args = parser.parse_args()

    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)


    tr = Trainer(args=args, vocabfile=args.vocabfile,name=args.model)
    tr.log('creat training server...')
    tr.start_server()

    tr.log('creat training model...')
    tr.build_model()

    tr.log('init batch generator...')
    tr.init_batch_gen(args.trainfiles, args.devfiles, major_time=args.features == 81)
    tr.log('starting training....')
    saveArgs(args, ' '.join(sys.argv), os.path.join(args.savepath,'args.conf'))

    tr.train()


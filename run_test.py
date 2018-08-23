#-*- coding:utf-8 -*-
"""
1. 08/12/2018, yb,change params pass mode, network params place in network.conf of the model save path,
               you can modify the conf/network(example).conf then copy it to the dest path!
              !!! the conf of network should consistent  with the model.ckpt,do not modify after the training many steps,
              otherwise you should train from begining.
"""
import argparse
import os
import sys
import tensorflow as tf
sys.path.append('./')

from utils.misc import saveArgs , loadJsonConfs, dotdict
from trainer.trainer import TestTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='command line for run_test ')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='rnncell')
    parser.add_argument('-t', '--testfiles', nargs='+', required=True, type=str, help='rnncell')
    parser.add_argument('-f', '--fitfiles', nargs='+', default=[], type=str, help='rnncell')
    parser.add_argument('-s', '--savepath', required=True, type=str, help='model parms path')


    #distributed test
    parser.add_argument('-gf', '--gpu_fraction', default=0.6, type=float, help='rnncell')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='rnncell')
    parser.add_argument('--ps_hosts', default='', type=str, help='comma-separated list of hostname:port pairs')
    parser.add_argument('--ws_hosts', default='', type=str, help='comma-separated list of hostname:port pairs')
    parser.add_argument('--job_name', default='ps', choices=['ps', 'worker'], type=str, help='ps or worker')
    parser.add_argument('--task_index', default=0, type=int, help='index of task within the job')
    args = parser.parse_args()

    if not os.path.exists(args.savepath):
        raise Exception('model path {} not found!'.formate(args.savepath))


    # load net graph parms from json file
    gp = loadJsonConfs(os.path.join(args.savepath,'graph.json'))

    ad = args.__dict__
    for k, v in ad.items():
        gp[k] = v

    args = dotdict(gp)

    print(args.rnn_layers)



    tr = TestTrainer(args=args)
    tr.log('creat test server...')
    tr.start_server()

    tr.log('creat test model graph...')
    tr.build_model(save_model_per_iters=600, val_loss_per_iters=200)

    tr.log('init model...')
    tr.init(major_time=args.features == 81, max_input_time_length=2000)
    tr.log('starting test....')

    tr.test()

    exit(0)
#-*- coding:utf-8 -*-
"""
Defines a class that is used to featurize audio clips, and provide
them to the network for training or testing.
"""

from __future__ import absolute_import, division, print_function
from functools import reduce,wraps
import time
import json
import logging
import numpy as np
import random

from concurrent.futures import ThreadPoolExecutor, wait



logger = logging.getLogger(__name__)


def describe(func):
    ''' wrap function,to add some descriptions for function and its running time
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__+'...')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(str(func.__name__+' in '+ str(end-start)+' s'))
        return result
    return wrapper


class Seq2SeqDataGenerator(object):
    def __init__(self, feat_label_descfile, major_time=False, max_time_length=None, max_label_length=None):
        pass



class DataGenerator(object):
    def __init__(self, feat_label_descfile, major_time=False, max_time_length=None,max_label_length=None):
        '''

        :param feat_label_descfile:
        :param major_time:
        :param max_time_length:
        :param max_label_length:
        '''

        RNG_SEED = int(time.time() * 1000 * 1000) % 10000
        self.rand_gen = random.Random(RNG_SEED)
        self.major_time=major_time
        self.max_time_length=max_time_length #default use batch max time length
        self.max_label_length=max_label_length

        self.load_desc_file(feat_label_descfile, max_duration= self.max_time_length or 10)


    @staticmethod
    @describe
    def create_desc_file(feat_label_path,save_json):
        for subdir, dirs, files in os.walk(feat_label_path):
            for f in files:
                fullpathfilename=os.path.join(subdir,f)
                if f.endswith('.npz'):
                    n = np.load(fullpathfilename)
                    feat = n['feature']
                    label= n['label']
                    ma = np.max(label)
                    dur = feat.shape[1] * 0.01 # should be win_step
                    with open(save_json,'a') as jf:
                        line = json.dumps({'key': fullpathfilename, 'duration': dur})
                        jf.write(line + '\n')

    @describe
    def load_desc_file(self, jsonfiles=[], max_duration=10.0):
        total_files = nouse_files = 0
        self.feat_label_paths = []
        self.durations = []
        for jsonfile in jsonfiles:
            with open(jsonfile) as jf:
                for line_num, json_line in enumerate(jf):
                    try:
                        total_files += 1
                        spec = json.loads(json_line)
                        if(float(spec['duration']) > max_duration):
                            #print('time too long :{}, {}'.format(spec['key'], spec['duration']))
                            nouse_files += 1
                            continue
                        self.feat_label_paths.append(spec['key'])
                        self.durations.append(spec['duration'])

                    except Exception as e:
                        print(str(e),'Error read line#{}:{}'.format(line_num,json_line))

        print('success load_desc_files:{}, >maxtime<{}>files:{}, total_files:{}'.format(total_files-nouse_files,max_duration,nouse_files,total_files))
        self.fit_train(k_samples=10)

    def get_batch_num(self,batch_size=16):
        return len(self.feat_label_paths) // batch_size

    def load_feat_label(self, filename):
        n = np.load(filename)
        label=n['label']
        feat=n['features']
        if not self.major_time: #[f,t]
            feat = np.swapaxes(feat,0,1)
        return feat,label

    def load_feat(self,filename):
        #print(filename)
        n = np.load(filename)
        #print(type(n))
        feat = n['features']
        if not self.major_time:  # [f,t]
            feat = np.swapaxes(feat, 0, 1)
        return feat

    @staticmethod
    def sort_by_duration(durations, feat_label_paths):
        return zip(*sorted(zip(durations, feat_label_paths),reverse=True))

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def prepare_minibatch(self, feat_label_paths):
        """ Featurize a minibatch of audio, zero pad them and return a dictionary
        Params:
            audio_paths (list(str)): List of paths to audio files
            texts (list(str)): List of texts corresponding to the audio files
        Returns:
            dict: See below for contents
        """
        features = []
        labels = []

        for f in feat_label_paths:
            feat,label = self.load_feat_label(f)
            features.append(feat)
            labels.append(label)
            if len(label) == 0:
                raise Exception('target label length==0, file:{}'.format(f))
            if self.max_label_length is not None:
                if len(label) > self.max_label_length:
                    print('target label length={},file:{}'.format(len(label),f))

        input_lengths = [f.shape[0] for f in features]
        max_length = max(input_lengths)

        feature_dim = features[0].shape[1]
        mb_size = len(features)

        # Pad all the inputs so that they are all the same length

        #x = np.zeros((mb_size, max_length, feature_dim))
        #[T,B,F]
        x = np.zeros((max_length, mb_size, feature_dim))
        y = []
        label_lengths = []
        pad_y = []
        for i in range(mb_size):
            feat = features[i]
            feat = self.normalize(feat)  # Center using means and std
            x[:feat.shape[0],i,  :] = feat
            label = list(labels[i])
            label_len = len(label)
            y.append(label)
            label_lengths.append(label_len)
        sparseY = self.list_to_sparse_tensor(y)
        # Flatten labels to comply with warp-CTC signature
        #y = reduce(lambda i, j: i + j, y)
        return {
            'x': x,  # (0-padded features of shape(mb_size,timesteps,feat_dim)
            'pady': pad_y,  # list(int) Flattened labels (integer sequences)
            'sparsey':sparseY,
            #'texts': texts,  # list(str) Original texts
            'input_lengths': input_lengths,  # list(int) Length of each input
            'label_lengths': label_lengths  # list(int) Length of each label
        }

    def list_to_sparse_tensor(self, targetList):
        indices =[]
        vals =[]
        for tI, target in enumerate(targetList):
            for seqI, val in enumerate(target):
                indices.append([tI,seqI])
                vals.append(val)
        shape=[len(targetList), np.asanyarray(indices).max(axis=0)[1]+1]
        return (np.array(indices), np.array(vals),np.array(shape))

    def iterate(self, feat_label_paths, minibatch_size, max_iters=None,allow_smaller_final_batch =True):
        if max_iters is not None:
            k_iters = max_iters
        else:
            k_iters = int(np.ceil(len(feat_label_paths) / minibatch_size))
            # discard the last batch ,if not allow_smaller_final_batch
            if k_iters*minibatch_size > len(feat_label_paths) and allow_smaller_final_batch == False:
                k_iters -= 1
        logger.info("Iters: {}".format(k_iters))
        pool = ThreadPoolExecutor(1)  # Run a single I/O thread in parallel
        future = pool.submit(self.prepare_minibatch,
                             feat_label_paths[:minibatch_size])
        start = minibatch_size
        for i in xrange(k_iters - 1):
            wait([future])
            minibatch = future.result()
            # While the current minibatch is being consumed, prepare the next
            future = pool.submit(self.prepare_minibatch,
                                 feat_label_paths[start: start + minibatch_size])
            yield minibatch
            start += minibatch_size
        # Wait on the last minibatch
        wait([future])
        minibatch = future.result()
        yield minibatch

    def iterate_train(self, minibatch_size=16, sort_by_duration=False,
                      shuffle=True,allow_smaller_final_batch=True):
        if sort_by_duration and shuffle:
            shuffle = False
            logger.warn("Both sort_by_duration and shuffle were set to True. "
                        "Setting shuffle to False")
        durations, feat_label_paths = (self.durations,self.feat_label_paths)
        if shuffle:
            temp = zip(durations, feat_label_paths)
            self.rand_gen.shuffle(temp)
            durations, feat_label_paths = zip(*temp)
        if sort_by_duration:
            durations, feat_label_paths =\
                DataGenerator.sort_by_duration(durations, feat_label_paths)
        return self.iterate(feat_label_paths, minibatch_size,allow_smaller_final_batch=allow_smaller_final_batch)

    def iterate_test(self, minibatch_size=16,max_iters=None,allow_smaller_final_batch=True):
        return self.iterate(self.feat_label_paths,
                            minibatch_size,max_iters=max_iters,
                            allow_smaller_final_batch=allow_smaller_final_batch)

    def iterate_validation(self, minibatch_size=16,shuffle=False,max_iters=None,allow_smaller_final_batch=True):
        durations, feat_label_paths = (self.durations, self.feat_label_paths)
        if shuffle:
            temp = zip(durations, feat_label_paths)
            self.rand_gen.shuffle(temp)
            durations, feat_label_paths = zip(*temp)
        return self.iterate(feat_label_paths, minibatch_size,
                            max_iters=max_iters,
                            allow_smaller_final_batch=allow_smaller_final_batch)

    def fit_train(self, k_samples=100):
        """ Estimate the mean and std of the features from the training set
        Params:
            k_samples (int): Use this number of samples for estimation
        """
        k_samples = min(k_samples, len(self.feat_label_paths))
        samples = self.rand_gen.sample(self.feat_label_paths, k_samples)
        feats = [self.load_feat(s) for s in samples]
        feats = np.vstack(feats)
        self.feats_mean = np.mean(feats, axis=0)
        self.feats_std = np.std(feats, axis=0)

if __name__ == '__main__':
    train_json=['../libri_test_clean.json', '../libri_test_other.json']


    t = DataGenerator(train_json, True)
    t.fit_train()

    for i ,batch in enumerate(t.iterate_train(16)):
        print(i, batch['x'].shape)




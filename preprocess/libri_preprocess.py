#-*- coding:utf-8 -*-
from __future__ import print_function,unicode_literals
import os
import argparse
import json
import sys
import scipy.io.wavfile as wav
from subprocess import check_call, CalledProcessError
from sklearn import preprocessing
import numpy as np
from wavprocess import spectrogram_from_file
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')


def preprocess(root_directory):
    """
    Function to walk through the directory and convert flac to wav files
    """
    try:
        check_call(['flac'])
    except OSError:
        raise OSError("""Flac not installed. Install using apt-get install flac""")
    print(root_directory)
    for subdir, dirs, files in os.walk(root_directory):
        for f in files:
            print('preprocess',f)
            filename = os.path.join(subdir, f)
            if f.endswith('.flac'):
                try:
                    check_call(['flac', '-d', filename])
                    os.remove(filename)
                except CalledProcessError as e:
                    print("Failed to convert file {}".format(filename))
            elif f.endswith('.TXT'):
                os.remove(filename)
            elif f.endswith('.txt'):
                with open(filename, 'r') as fp:
                    lines = fp.readlines()
                    for line in lines:
                        sub_n = line.split(' ')[0] + '.label'
                        subfile = os.path.join(subdir, sub_n)
                        sub_c = ' '.join(line.split(' ')[1:])
                        sub_c = sub_c.lower()
                        with open(subfile, 'w') as sp:
                            sp.write(sub_c)
            elif f.endswith('.wav'):
                if not os.path.isfile(os.path.splitext(filename)[0] +
                                      '.label'):
                    raise ValueError(".label file not found for {}".format(filename))
            else:
                pass


def wav2feature(root_directory, save_directory, name, win_len, win_step, mode, feature_len, save,jsonfile,split,seq2seq=False):
    count = 0
    dirid = 0

    max_lengths = 0.0
    total_lengths=0.0
    data_dir = os.path.join(root_directory, name)
    preprocess(data_dir)
    for subdir, dirs, files in os.walk(data_dir):
        for f in files:
            fullFilename = os.path.join(subdir, f)
            filenameNoSuffix =  os.path.splitext(fullFilename)[0]
            if f.endswith('.wav'):
                rate = None
                sig = None
                try:
                    (rate,sig)= wav.read(fullFilename)
                    dur=float(len(sig)) / rate
                except ValueError as e:
                    if e.message == "File format 'NIST'... not understood.":
                        sf = Sndfile(fullFilename, 'r')
                    nframes = sf.nframes
                    sig = sf.read_frames(nframes)
                    rate = sf.samplerate
                    dur = float(len(sig)) / rate
                max_lengths = max(max_lengths,dur)
                total_lengths += dur

                if(mode == 'log'):
                    if feature_len == 81:
                        maxfreq = 4000
                    else:
                        maxfreq = 8000
                    feat = spectrogram_from_file(filename=fullFilename,
                                                step=int(win_step*1000),
                                                window=int(win_len*1000),
                                                max_freq=maxfreq)
                else:
                    feat = calcfeat_delta_delta(sig,rate,win_length=win_len,win_step=win_step,mode=mode,feature_len=feature_len)

                feat = preprocessing.scale(feat)
                feat = np.transpose(feat)
                print(feat.shape)
                labelfilename = filenameNoSuffix + '.label'
                with open(labelfilename,'r') as lf:
                    characters = lf.readline().strip().lower()
                targets = []
                if seq2seq is True:
                    targets.append(28)
                for c in characters:
                    if c == ' ':
                        targets.append(0)
                    elif c == "'":
                        targets.append(27)
                    else:
                        targets.append(ord(c)-96)
                if seq2seq is True:
                    targets.append(29)
                print(targets)
                if save:
                    count+=1
                    if count%10000 == 0:
                        dirid += 1
                    print('file index:',count)
                    print('dir index:',dirid)
                    feat_label_dir = os.path.join(save_directory, name, str(dirid))
                    if not os.path.isdir(feat_label_dir):
                        os.makedirs(feat_label_dir)
                    feat_label_Filename = os.path.join(feat_label_dir, filenameNoSuffix.split('/')[-1] +'.npz')
                    np.savez(feat_label_Filename,features=feat, label=targets)
                    if jsonfile is not None:
                        with open(jsonfile,'a') as jf:
                            line = json.dumps({'key':feat_label_Filename, 'duration':dur, 'texts':characters})
                            jf.write(line+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='libri_preprocess',description='Script to preprocess libri data')
    parser.add_argument("path", help="Directory of LibriSpeech dataset", type=str)
    parser.add_argument("save", help="Directory where preprocessed arrays are to be saved",type=str)
    parser.add_argument("jsonfile", help="json file name where preprocessed arrays are to be saved to", type=str)
    parser.add_argument("-m", "--mode", help="Mode",choices=['mfcc', 'fbank','log'],type=str, default='fbank')
    parser.add_argument("-f", "--featlen", help='Features length', type=int, choices=[13,81,61],default=13)
    parser.add_argument("-wl", "--winlen", type=float, default=0.02, help="specify the window length of feature")
    parser.add_argument("-ws", "--winstep", type=float,default=0.01, help="specify the window step length of feature")
    parser.add_argument("-s", "--split", type=str, default=None, help="split to train/dev/test ratio of one path, None do nothing")
    parser.add_argument("-n", "--name", help="Name of the librispeech dataset",
                        choices=['dev-clean', 'dev-other', 'test-clean',
                                 'test-other', 'train-clean-100', 'train-clean-360',
                                 'train-other-500'], type=str, default='test-clean')
    args = parser.parse_args()
    root_directory = args.path
    save_directory = args.save
    mode = args.mode
    feature_len = args.featlen
    name = args.name
    win_len = args.winlen
    win_step = args.winstep

    if root_directory == '.':
        root_directory = os.getcwd()

    if save_directory == '.':
        save_directory = os.getcwd()

    if not os.path.isdir(root_directory):
        raise ValueError("LibriSpeech Directory does not exist!")

    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    if os.path.exists(args.jsonfile):
        os.rename(args.jsonfile, args.jsonfile+'.bak')

    wav2feature(root_directory, save_directory, name=name, win_len=win_len, win_step=win_step,
                mode=mode, feature_len=feature_len, jsonfile=args.jsonfile, save=True,split=args.split)
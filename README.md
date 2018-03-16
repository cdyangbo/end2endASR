# End-to-End Automatic Speech Recognition use tensorflow
use tensorflow to implement a end-to-end algorithm according baidu deepspeech paper[DeepSpeech](https://arxiv.org/abs/1412.5567),[DeepSpeech2](http://proceedings.mlr.press/v48/amodei16.pdf) and seq2seq listener-attention-speller model[Listen, Attend and Spell](https://arxiv.org/pdf/1508.01211.pdf) [Towards better decoding and language model integration in sequence to Sequence model](https://arxiv.org/pdf/1612.02695.pdf)

## 1 prerequests.
  see requirements.txt
  
## 2 data and preprocess
### 2.1 English corpus:[LibriSpeech](http://www.openslr.org/12)
### 2.2 Chinese corpus:[THCHS-30](http://www.openslr.org/18/)
### 2.3 preprocess
#### 1.librispeech
```
usage: libri_preprocess [-h] [-m {mfcc,fbank,log}] [-f {13,81,161}]
                        [-wl WINLEN] [-ws WINSTEP] [-s SPLIT]
                        [-n {dev-clean,dev-other,test-clean,test-other,train-clean-100,train-clean-360,train-other-500}]
                        path save jsonfile
param:
     -m, feature type, mfcc=13 dim,fbank=40dim,log=81or161 dim
     -f, feature dims, mfcc=13,log=81,161
     -n, librispeech corpus sub dirs
     path,corpus data dir
     save,feature save dir
     jsonfile, json file to index all wav feature and ground truth
```   
``` 
sample script
$python libri_preprocess.py -m log -f 81 -n dev-clean ~/asr_corpus/librispeech/LibriSpeech ~/asr_corpus/librispeech_feat ~/asr_corpus/dev-clean-featlabel.json
  
```
## 3 train
run_train.py
```
usage: run_train.py [-h] [-rc {gru,lstm,rnn}] [-b BATCH_SIZE] [-n HIDDENS]
                    [-f {13,39,81,161}] [-c CLASSES] [-rl RNN_LAYERS]
                    [-cl CONV_LAYERS] [-g GPUS] [-a {relu,tanh,sigmod}]
                    [-o OPTIMIZER] [-lr LEARNING_RATE] [-k KEEP_PROB]
                    [-gc GRAD_CLIP] [-m MODE] [-r RESTOREMODEL]
                    [-bn BATCHNORM] [-p EPOCHS] [-i INITIAL_EPOCH] -t
                    TRAINFILES [TRAINFILES ...] -d DEVFILES [DEVFILES ...] -s
                    SAVEPATH [-gf GPU_FRACTION] [-md MODEL]
                    [-ub USE_BIDIRECTIONAL_RNN] [-us USE_SUMMARY]
                    [-v VOCABFILE] [--ps_hosts PS_HOSTS] [--ws_hosts WS_HOSTS]
                    [--job_name {ps,worker}] [--task_index TASK_INDEX]

```
### 3.1 single machine training 
```
deepspeech2 model
$./run_libri_ds2_train.sh
las model 
$./run_libri_las_train.sh
```
### 3.2 clustering machine training 
```
two gpu cars.
deepspeech2 model
$./run_libri_ds2_train_dist.sh ps 0
$./run_libri_ds2_train_dist.sh worker 0
$./run_libri_ds2_train_dist.sh worker 1

las model 
$./run_libri_las_train_dist.sh ps 0
$./run_libri_las_train_dist.sh worker 0
$./run_libri_las_train_dist.sh worker 1

```

## 4 test

## 5 other

#### refs. [zzw922cn/Automatic_Speech_Recognition](https::/github.com/zzw922cn/Automatic_Speech_Recognition) 

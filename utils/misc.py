#-*- coding:utf-8 -*-
import os
import  time

def log(info, logfilename='log.txt',savepath = None, debug=True):
    t = time.localtime(int(time.time()))
    ts = time.strftime('%m-%d %H:%M:%S ', t)
    if savepath is not None:
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        fullfilename = os.path.join(savepath,logfilename)
    else:
        fullfilename = logfilename

    if debug:
        print(ts + info)

    with open(fullfilename,'a') as f:
        f.write(ts+info+'\n')


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def loadArgs(filename='args.conf'):
    args = {}
    with open(filename, 'r') as f:
        for l in f.readlines():
            kv = l.strip('\n').split('=')
            if kv[2] == 'str':
                args[kv[0]] = kv[1]
            elif kv[2] == 'int':
                args[kv[0]] = int(kv[1])
            elif kv[2] == 'float':
                args[kv[0]] = float(kv[1])
            elif kv[2] == 'bool':
                args[kv[0]] = bool(kv[1])
            elif kv[2] == 'list':
                args[kv[0]] = kv[1].split()
            else:
                raise Exception('no supported data type:{}'.format(kv[2]))

    return dotdict(args)


def saveArgs(args, cmdstr, filename='args.conf'):
    ad = args.__dict__
    t = time.localtime(int(time.time()))
    ts = time.strftime('%m-%d %H:%M:%S ', t)
    with open(filename, 'w') as f:
        f.write('created at:' + ts + 'cmd:<' + cmdstr + '>\n')
        for k, v in ad.items():
            if type(v) == int:
                f.write(k + '=' + str(v) + '=int\n')
            elif type(v) == str:
                f.write(k + '=' + str(v) + '=str\n')
            elif type(v) == float:
                f.write(k + '=' + str(v) + '=float\n')
            elif type(v) == bool:
                f.write(k + '=' + str(v) + '=bool\n')
            elif type(v) == list:
                f.write(k + '=' + ' '.join(v) + '=list\n')
            else:
                raise Exception('no supported data type:{}'.format(v))

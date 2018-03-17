#-*- coding:utf-8 -*-
import json
import os

def loadvocabulary(vocabfile,mode='jsondict'):
    char_map = {}
    index_map = {}

    if mode == 'jsondict': # jsondict {'key':k,'word':w}
        with open(vocabfile,'r') as f:
            for line in f.readlines():
                d = json.loads(line)
                index_map[d['key']] = d['word']
                char_map[d['word']] = d['key']
    elif mode == 'alphabet': # a-z
        with open(vocabfile, 'r') as f:
            for line in f.readlines():
                line.strip('\n').strip()
                ch, index = line.split()
                char_map[ch] = int(index)
                index_map[int(index)] = ch
            index_map[0] = ' '
    else:
        raise Exception('not supported mode:{}'.format(mode))

    #print char_map
    return char_map, index_map


def text_to_int_sequence(char_map, text):
    '''
    Use a character map and convert text to an integer sequence
    :param char_map:
    :param text: list of text
    :return: list of vocab's index
    '''
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence


def output_to_sequence(lmt, vocab, sp=None):
    ''' convert the output into sequences of characters
    '''
    sequences = []
    start = 0
    sequences.append([])

    if sp is None:
        sp = ''

    for i in range(len(lmt[0])):
        if lmt[0][i][0] == start:
            sequences[start].append(lmt[1][i])
        else:
            start = start + 1
            sequences.append([])
            sequences[start].append(lmt[1][i])

    out_seqs = []
    for i in range(len(sequences)):
        indexes = sequences[i]
        seq = []
        for ind in indexes:
            seq.append(vocab[ind])
        seq = sp.join(seq)
        out_seqs.append(seq)
    #print('output_to_sequence',len(lmt[0]),len(out_seqs))
    return out_seqs


def output_to_sequence_dense(lmt, vocab, sp=None, label_length=None):
    sequences = lmt
    if sp is None:
        sp = ''
    out_seqs = []

    for i in range(len(sequences)):
        indexes = sequences[i]
        if label_length is not None:
            indexes = indexes[:label_length[i]]
        seq = []
        for ind in indexes:
            seq.append(vocab[ind])
        seq = sp.join(seq)
        out_seqs.append(seq)
    #print('output_to_sequence_dense', len(lmt[0]), len(out_seqs))
    return out_seqs


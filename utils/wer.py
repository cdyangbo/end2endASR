# -*- coding: utf-8 -*-

import sys
import numpy

def cer(groundtruth_str, hypothesis_str, debug=False):
    '''
    This is a function that calculate the character error rate in ASR.
    You can use it like this: wer("what is it", "what is")
    :param groundtruth_str:
    :param hypothesis_str:
    :param debug:
    :return:
    '''

    ground_truth_list=[c for c in groundtruth_str]
    hypothesis_str = [c for c in hypothesis_str]

    return wer(ground_truth_list,hypothesis_str,debug)


def wer(groundtruth_list, hypothesis_list, debug=True):
    '''
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    :param groundtruth_list: list of ground truth
    :param hypothesis_list: list of hypsotesis
    :param debug: print debug info
    :return:
    '''

    r = groundtruth_list
    h = hypothesis_list
    # build the matrix
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    if debug:
        showwerinfo(r, h, d)
    return d[len(r)][len(h)], len(r)


def showwerinfo(r, h, d):
    # find out the manipulation steps
    x = len(r)
    y = len(h)
    list = []
    result = float(d[len(r)][len(h)]) / len(r) * 100
    result = str("%.2f" % result) + "%"
    while True:
        if x == 0 and y == 0:
            break
        else:
            if d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
                list.append("e")
                x = x - 1
                y = y - 1
            elif d[x][y] == d[x][y - 1] + 1:
                list.append("i")
                x = x
                y = y - 1
            elif d[x][y] == d[x - 1][y - 1] + 1:
                list.append("s")
                x = x - 1
                y = y - 1
            else:
                list.append("d")
                x = x - 1
                y = y
    list = list[::-1]
    # print d
    # print list

    # print the result in aligned way
    print "REF:",
    for i in range(len(list)):
        if list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1;
            index = i - count
            print " " * (len(h[index])),
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1;
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1;
            index2 = i - count2
            if len(r[index1]) < len(h[index2]):
                print r[index1] + " " * (len(h[index2]) - len(r[index1])),
            else:
                print r[index1],
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1;
            index = i - count
            print r[index],
    print
    print "HYP:",
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1;
            index = i - count
            print " " * (len(r[index])),
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1;
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1;
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print h[index2] + " " * (len(r[index1]) - len(h[index2])),
            else:
                print h[index2],
        else:
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1;
            index = i - count
            print h[index],
    print
    print "EVA:",
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1;
            index = i - count
            print "D" + " " * (len(r[index]) - 1),
        elif list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1;
            index = i - count
            print "I" + " " * (len(h[index]) - 1),
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1;
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1;
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print "S" + " " * (len(r[index1]) - 1),
            else:
                print "S" + " " * (len(h[index2]) - 1),
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1;
            index = i - count
            print " " * (len(r[index])),
    print
    print "WER: " + result


if __name__ == '__main__':
    # filename1 = sys.argv[1]
    # filename2 = sys.argv[2]
    r = "aa abcd aa".split()
    h = "aa a b c d aa".split()
    print wer(r, h, True)
    print cer("aa abcd aa","aa ab cd aa")
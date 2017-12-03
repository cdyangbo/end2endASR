#-*- coding:utf-8 -*-
import tensorflow as tf

def variable_summaries(var,name_scope='summaries'):
    vn = name_scope.replace(':','_')
    with tf.name_scope(vn):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('mean',mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
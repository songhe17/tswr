#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 08:57:09 2020

@author: songhewang
"""

import tensorflow as tf
tf.reset_default_graph()
print(tf.__version__)
tf.set_random_seed(1234)#1234
class case_model(object):
    
    def __init__(self, batch, k, optimizer):
        self.batch = batch
        self.k = k
        self.optimizer = optimizer
        self.X = tf.placeholder(tf.float32, [batch,k-1], 'X')
        self.C = tf.placeholder(tf.float32, [batch,k-1], 'C')
        self.x = tf.placeholder(tf.float32, [batch,1], 'x')
        self.c = tf.placeholder(tf.float32, [batch,1], 'c')
        self.gt = tf.placeholder(tf.float32, [batch,1], 'gt')
        self.wd = tf.placeholder(tf.float32, [batch,7], 'wd')
        
        with tf.variable_scope('init', reuse = tf.AUTO_REUSE):
            w_1 = tf.get_variable('w_1', [k-1,1], tf.float32,tf.random_uniform_initializer(minval=-1., maxval=1.))
            w_2 = tf.get_variable('w_2', [k-1,1], tf.float32,tf.random_uniform_initializer(minval=-1., maxval=1.))
            beta_1 = tf.get_variable('beta_1', [1,1], tf.float32,tf.random_uniform_initializer(minval=-1., maxval=1.))
            beta_2 = tf.get_variable('beta_2', [1,1], tf.float32,tf.random_uniform_initializer(minval=-1., maxval=1.))
            bias = tf.get_variable('bias', [1,1], tf.float32,tf.random_uniform_initializer(minval=-1., maxval=1.))
            theta = tf.get_variable('theta', [7,1], tf.float32,tf.random_uniform_initializer(minval=-1., maxval=1.))
        self.prediction = (tf.matmul(self.X, w_1)+self.x) * tf.tile(beta_1, [batch,1])
        self.prediction += (tf.matmul(self.C, w_2)+self.c) * tf.tile(beta_2, [batch,1])
        self.prediction += tf.tile(bias, [batch,1])
        self.prediction *= tf.matmul(self.wd, theta)
        self.loss = tf.reduce_mean(tf.squared_difference(self.gt, self.prediction))
        self.train_op = self.optimizer.minimize(self.loss)
        
    def predict(self, sess):
        return sess.run(self.prediction)
    
    def optimize(self, sess, feed_dict):
        _,loss,prediction = sess.run([self.train_op,self.loss, self.prediction], feed_dict)
        return loss, prediction
     
    def save(self, sess, path):
        self.saver = tf.train.Saver()
        self.saver.save(sess, path)
        
    def load(self, sess, path):
        self.saver = tf.train.Saver()
        self.saver.restore(sess, path)
        
        
        
if __name__ == '__main__':
    optimizer = tf.train.AdamOptimizer(0.01)
    model = case_model(10,20,optimizer)
    
    
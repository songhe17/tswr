#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:37:24 2020

@author: songhewang
"""

import tensorflow as tf
tf.reset_default_graph()
print(tf.__version__)

class TSWR(object):
    
    def __init__(self,n,k,opimizer):
        
        self.n = n
        self.k = k
        self.optimizer = optimizer
        self.x = tf.placeholder(tf.float32, [n,k-1], 'x')
        self.x_case = tf.placeholder(tf.float32, [n,1], 'x_case')
        self.t = tf.placeholder(tf.float32, [n,1], 't')
        self.gt = tf.placeholder(tf.float32, [n,1], 'gt')
        
        with tf.variable_scope('init', reuse = tf.AUTO_REUSE):
            self.w = tf.get_variable('w', [n,1], tf.float32)
            self.bias = tf.get_variable('bias', [n,1], tf.float32)
            self.alpha = tf.get_variable('alpha', [n,1], tf.float32)
            self.beta = tf.get_variable('beta', [k-1,1], tf.float32)
            
        self.y = self.w * (tf.matmul(self.x, self.beta) + self.x_case * self.alpha * self.t + self.bias)
        self.loss = tf.reduce_mean(tf.squared_difference(self.gt, self.y))
        self.train_op = self.optimizer.optimize(self.loss)
        
    def predict(self, sess):
        return sess.run(self.y)
    
    def optimize(self, sess, feed_dict):
        _,loss = sess.run([self.train_op,self.loss], feed_dict)
        

        
        
        
twsr = TSWR(10,20,None)
y = twsr.predict() 
print(y) 
        
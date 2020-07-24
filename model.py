#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:37:24 2020

@author: songhewang
"""

import tensorflow as tf
tf.reset_default_graph()
print(tf.__version__)
tf.set_random_seed(1234)#1234
class TSWR(object):
    
    def __init__(self,batch,n,k,optimizer):
        
        self.batch = batch
        self.n = n
        self.k = k
        self.optimizer = optimizer
        self.x_other = tf.placeholder(tf.float32, [batch,k,n-1], 'x_o')
        self.x = tf.placeholder(tf.float32, [batch,1,k], 'x')
        self.case_other = tf.placeholder(tf.float32, [batch,1,n-1], 'c_o')
        self.case = tf.placeholder(tf.float32, [batch,1,1], 'c')
        self.wd = tf.placeholder(tf.float32, [batch,1,7], 'wd')
        self.gt = tf.placeholder(tf.float32, [batch,1], 'gt')
        
        
        
        with tf.variable_scope('init', reuse = tf.AUTO_REUSE):
            self.w = tf.get_variable('w', [1,n-1,1], tf.float32,tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.bias = tf.get_variable('bias', [1], tf.float32,tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.alpha = tf.get_variable('alpha', [1,n-1,1], tf.float32,tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.beta = tf.get_variable('beta', [1,k,1], tf.float32,tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.eta = tf.get_variable('eta', [1,1,1], tf.float32,tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.theta = tf.get_variable('theta', [1,7,1], tf.float32,tf.random_uniform_initializer(minval=-1., maxval=1.))
        tf.tile(self.beta, [self.batch,1,1])
        
        self.y = tf.matmul(self.x, tf.tile(self.beta, [self.batch,1,1]))
        
        #print(self.y)
        
        self.y += tf.matmul(tf.tile(tf.transpose(self.beta, [0,2,1]), [self.batch,1,1]), tf.matmul(self.x_other, tf.tile(self.w, [self.batch,1,1])))
        
        #print(self.y)
        
        self.y += tf.tile(self.eta, [self.batch,1,1]) * self.case
        
        #print(self.y)
        
        self.y += tf.matmul(tf.tile(tf.transpose(self.eta, [0,2,1]), [self.batch,1,1]),tf.matmul(self.case_other, tf.tile(self.alpha,[self.batch,1,1])))
        
        #print(self.y)
        
        self.y *= tf.matmul(self.wd, tf.tile(self.theta,[self.batch,1,1]))
        
        #print(self.y)
        
        self.y = tf.reshape(self.y, [self.batch,-1])
        self.loss = tf.reduce_mean(tf.squared_difference(self.gt, self.y))
        self.train_op = self.optimizer.minimize(self.loss)
        
    def predict(self, sess):
        return sess.run(self.y)
    
    def optimize(self, sess, feed_dict):
        _,loss,prediction = sess.run([self.train_op,self.loss, self.y], feed_dict)
        return loss, prediction
     
    def save(self, sess, path):
        self.saver = tf.train.Saver()
        self.saver.save(sess, path)
        
    def load(self, sess, path):
        self.saver = tf.train.Saver()
        self.saver.restore(sess, path)
    
        
if __name__ == '__main__':    
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    
    with open('data/preprocessed/input_Los Angeles.pkl','rb') as f:
        data = pickle.load(f)
    gt = data['flow']
    x = data['x']
    x_other = data['x_other']
    case = data['c']
    case_other = data['c_other']
    wd = data['weekday']
    for key, value in data.items():
        print(np.shape(value))
    batch = np.shape(x_other)[0]
    n = np.shape(x_other)[2] + 1
    k = np.shape(x_other)[1]
    twsr = TSWR(batch,n,k,tf.train.AdamOptimizer(0.01))
    feed_dict = {twsr.x:x, twsr.x_other:x_other, twsr.case:case, twsr.case_other:case_other,twsr.wd:wd, twsr.gt:gt}
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            
            loss, prediction = twsr.optimize(sess, feed_dict)
            
            if i % 100 == 1:
                print(loss)
        theta, beta, w = sess.run([twsr.theta, twsr.beta, twsr.w], feed_dict)
        #print(prediction)
        #print(gt)
        twsr.save(sess, 'save_model/model_1')
        
        
    beta = np.reshape(beta, (-1))
    w = np.reshape(w,(-1))
    reg_keys = ['population_density_per_sqmi',
            'percent_smokers', 'percent_adults_with_obesity',
            'percent_excessive_drinking', 'percent_uninsured',
            'percent_unemployed_CHR',
            'violent_crime_rate','life_expectancy',
            'percent_65_and_over', 'per_capita_income', 'percent_below_poverty', 'case']
    all_counties = ['Alameda', 'Amador', 'Butte', 'Calaveras', 'Colusa', 'Contra Costa', 'Del Norte', 'El Dorado', 'Fresno', 'Glenn', 'Humboldt', 'Imperial', 'Inyo', 'Kern', 'Kings', 'Lake', 'Lassen', 'Los Angeles', 'Madera', 'Marin', 'Mariposa', 'Mendocino', 'Merced', 'Mono', 'Monterey', 'Napa', 'Nevada', 'Orange', 'Placer', 'Riverside', 'Sacramento', 'San Benito', 'San Bernardino', 'San Diego', 'San Francisco', 'San Joaquin', 'San Luis Obispo', 'San Mateo', 'Santa Barbara', 'Santa Clara', 'Santa Cruz', 'Shasta', 'Siskiyou', 'Solano', 'Sonoma', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Tuolumne', 'Ventura', 'Yolo', 'Yuba']
    for b, key in zip(beta, reg_keys):
        print(f'{key}: {b}')
        
    print('\n' * 4)
    temp = {}
    for weight, county in zip(w, all_counties):
        temp[weight] = county
        #print(f'{county}: {weight}')
    keys = sorted(temp)
    print(temp)
    for key in keys:
        value = temp[key]
        print(f'{value}: {key}')
    plt.figure(0)
    gt = np.reshape(gt,(-1))
    prediction = np.reshape(prediction, (-1))
    plt.plot(gt)
    plt.plot(prediction)
    plt.figure(1)
    theta = np.reshape(theta,(-1))
    plt.plot(theta)
# =============================================================================
#     plt.figure(2)
#     beta = np.reshape(beta, (-1))
#     plt.plot(beta)
#     plt.figure(3)
#     w = np.reshape(w,(-1))
#     plt.plot(w)
# =============================================================================
    
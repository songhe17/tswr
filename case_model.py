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
        self.w_1, self.w_2, self.beta_1, self.beta_2, self.bias, self.theta = w_1, w_2, beta_1, beta_2,bias, theta
        
        self.prediction = (tf.matmul(self.X, w_1)) * tf.tile(beta_1, [batch,1])
        #self.prediction = self.x * tf.tile(beta_1, [batch,1])
        #self.prediction += (tf.matmul(self.C, w_2)+self.c) * tf.tile(beta_2, [batch,1])
        self.prediction += tf.tile(bias, [batch,1])
        #self.prediction *= tf.matmul(self.wd, theta)
        self.loss = tf.reduce_mean(tf.squared_difference(self.gt, self.prediction))
        self.train_op = self.optimizer.minimize(self.loss)
        
    def predict(self, sess):
        return sess.run(self.prediction);
    
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

    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import pprint
    from sklearn.decomposition import PCA
    tf.set_random_seed(1234)

    pca = PCA()
    all_counties = ['Alameda', 'Amador', 'Butte', 'Calaveras', 'Colusa', 'Contra Costa', 'Del Norte', 'El Dorado', 'Fresno', 'Glenn', 'Humboldt', 'Imperial', 'Inyo', 'Kern', 'Kings', 'Lake', 'Lassen', 'Los Angeles', 'Madera', 'Marin', 'Mariposa', 'Mendocino', 'Merced', 'Mono', 'Monterey', 'Napa', 'Nevada', 'Orange', 'Placer', 'Riverside', 'Sacramento', 'San Benito', 'San Bernardino', 'San Diego', 'San Francisco', 'San Joaquin', 'San Luis Obispo', 'San Mateo', 'Santa Barbara', 'Santa Clara', 'Santa Cruz', 'Shasta', 'Siskiyou', 'Solano', 'Sonoma', 'Stanislaus', 'Sutter', 'Tehama', 'Tulare', 'Tuolumne', 'Ventura', 'Yolo']
    county_weights = {county:0 for county in all_counties}
    for index,selected_county in enumerate(all_counties):
        tf.reset_default_graph()
        print(selected_county)
        selected_county_weights = {}
        with open(f'data/preprocessed/input/{selected_county}_prev.pkl','rb') as f:
            data = pickle.load(f)
        X = data['X']
        X = np.reshape(X,(np.shape(X)[0],-1))
        #sns.heatmap(np.corrcoef(X))
        print(np.shape(X))
        X = pca.fit_transform(X)
        print(np.corrcoef(X))
        #print(pca.explained_variance_ratio_)

        x = data['x']
        C = data['C']
        c = data['c']
        gt = data['flow']
        wd = data['weekday']

        batch = np.shape(X)[0]
        k = np.shape(X)[1] + 1
        name_adam = selected_county.replace(' ', '') + '_adam'
        optimizer = tf.train.AdamOptimizer(0.01, name = name_adam)
        model = case_model(batch,k,optimizer)

        
        #feed_dict = {model.X:X, model.x:x, model.C:C, model.c:c, model.gt:gt, model.wd:wd}
        feed_dict = {model.X:X, model.gt:gt, model.wd:wd}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(100):
                
                loss, prediction = model.optimize(sess, feed_dict)
                
            theta, beta, w = sess.run([model.theta, model.beta_1, model.w_1], feed_dict)
            #print(prediction)
            #print(gt)
            model.save(sess, 'save_model/model_case')
        # plt.figure(index)
        # plt.plot(theta)
        # plt.savefig(f'figures/weekday_weights/{selected_county}_ww.png')
        plt.figure(index)
        plt.plot(gt)
        plt.plot(prediction)
        plt.plot(x)
        plt.savefig(f'figures/predictions/{selected_county}_pred.png')

        counties = [c for c in all_counties if c != selected_county]
        #print(w)
        for weight, county in zip(w, counties):
            county_weights[county] += weight[0]


            selected_county_weights[county] = weight[0]

        with open(f'results/{selected_county}_weights.pkl', 'wb') as f:
            pickle.dump(selected_county_weights,f)


        #print(beta)
        #print(w)
        plt.figure(0)
        gt = np.reshape(gt,(-1))
        prediction = np.reshape(prediction, (-1))
        plt.plot(gt)

        plt.plot(prediction)
        plt.show()
            
        break
    #print(county_weights)
    # with open('county_weights.pkl', 'wb') as f:
    #     pickle.dump(county_weights,f)


        
        
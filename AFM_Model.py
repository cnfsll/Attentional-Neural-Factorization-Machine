# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 17:19:12 2018

@author: minjiang

定义AFM模型
参考如下：
https://github.com/hexiangnan/attentional_factorization_machine
Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu and Tat-Seng Chua (2017).
Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks
IJCAI, Melbourne, Australia, August 19-25, 2017.
"""

import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.contrib.layers import batch_norm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from time import time

class AFM(BaseEstimator, TransformerMixin):
    
    def __init__(self, features_M, em_factor, attention_factor, epoch, batch_size, learning_rate,
                 keep_prob, optimizer_type, bn, activation_function, verbose, early_stop,
                 attention, fields, lamda_attention, lamda_em, decay, save_file, random_seed=2016):
        
        '''
        features_M:特征维度
        em_factor: embedding维度
        attention_factor: 注意力网络的隐藏层的节点数
        epoch：迭代次数
        batch_size:batch的大小
        learning_rate:学习率
        keep_prob:对embedding层和attention网络使用dropout,1:no dropout
        optimizer_type:优化方法选择
        bn:是否对隐藏层使用BN，1：使用
        activation_function: 激活函数
        verbose:显示运行结果，每X次迭代显示一次
        early_stop:是否应用早停策略
        attention:是否使用注意力机制
        fields：输入特征的类别数，对于frappe：类别数为10；对于：ml：类别数为3
        lamda_attention：对attention网络的参数使用正则化的系数，0则不使用
        lamda_em: 对embedding层的参数使用正则化的系数，0则不使用
        decay：BN操作中移动平均系数
        save_file：训练后的模型保存地址        
        '''

        self.features_M = features_M    
        self.em_factor = em_factor
        self.attention_factor = attention_factor
        self.epoch = epoch        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.optimizer_type = optimizer_type
        self.bn = bn
        self.activation_function = activation_function
        self.verbose = verbose
        self.early_stop = early_stop
        self.attention = attention
        self.fields = fields
        self.lamda_attention = lamda_attention
        self.lamde_em = lamda_em
        self.decay = decay
        self.save_file = save_file
        self.random_seed = random_seed
        
        # 迭代误差
        self.train_loss, self.valid_loss, self.test_loss = [], [], [] 
        
        # 初始化计算图
        self._init_graph()

    def _init_graph(self):
        #初始化Tensorflow计算图，包括输入数据，变量，模型，损失和优化
        
        self.graph = tf.Graph()
        with self.graph.as_default():  # 默认使用cpu:
            
            tf.set_random_seed(self.random_seed)
            # 输入数据
            self.train_features = tf.placeholder(tf.int32, shape=[None, None], name="train_features")  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1], name="train_labels")  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            # 变量
            self.weights = self._initialize_weights()        
        
        
            # 模型定义
            self.nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features) # None * M' * K; M'即fields, K即em_factor
            #Pair-wise Interation Layer
            element_wise_product_list = []
            for i in range(0, self.fields):
                for j in range(i+1, self.fields):
                    element_wise_product_list.append(tf.multiply(self.nonzero_embeddings[:,i,:], self.nonzero_embeddings[:,j,:]))
            #将一个list变为一个tensor，上述list由M'*(M'-1)个None * K的tensor组成
            self.element_wise_product = tf.stack(element_wise_product_list) # (M'*(M'-1)) * None * K
            self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1,0,2], name="element_wise_product") # None * (M'*(M'-1)) * K
            self.interactions = tf.reduce_sum(self.element_wise_product, 2, name="interactions")  # None * (M'*(M'-1))
            
            # _________ 注意力机制部分 _____________
            num_interactions = int(self.fields*(self.fields-1)/2)
            if self.attention:
                self.attention_mul = tf.reshape(tf.matmul(tf.reshape(self.element_wise_product, shape=[-1, self.em_factor]), \
                    self.weights['attention_W']), shape=[-1, num_interactions, self.attention_factor])
                #上式中第一个reshape的目的size由None * (M'*(M'-1)) * K 变为 (None*(M'*(M'-1))) * K, 因为后面的权重为二维tensor
                #第一个reshpae再讲size变回None * (M'*(M'-1)) * attention_factor
                self.attention_exp = tf.exp(tf.reduce_sum(tf.multiply(self.weights['attention_p'], tf.nn.relu(self.attention_mul + \
                    self.weights['attention_b'])), 2, keep_dims=True)) # None * (M'*(M'-1)) * 1
                self.attention_sum = tf.reduce_sum(self.attention_exp, 1, keep_dims=True) # None * 1 * 1
                self.attention_out = tf.div(self.attention_exp, self.attention_sum, name="attention_out") # None * (M'*(M'-1)) * 1
            #attention不使用dropout和bn处理，对该网络的权重使用L2正则化
            
            # _________ 基于注意力机制的池化层 _____________
            if self.attention:
                self.AFM = tf.reduce_sum(tf.multiply(self.attention_out, self.element_wise_product), 1, name="afm") # None * K
            else:
                self.AFM = tf.reduce_sum(self.element_wise_product, 1, name="afm") # None * K
            
            #对attention后的输出执行BN操作
            if self.bn:
                self.AFM = self.batch_norm_layer(self.AFM, train_phase=self.train_phase, scope_bn='bn_fm')                
            #对attention后的输出执行dropout操作
            self.AFM = tf.nn.dropout(self.AFM, self.dropout_keep[0]) # dropout
            
            # ___________ 输出层 ___________________
            self.Bilinear = tf.matmul(self.AFM, self.weights['prediction']) # None * 1
            #Bilinear = tf.reduce_sum(self.Bilinear, 1, keep_dims=True)  # None * 1
            self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features) , 1)  # None * 1
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n([self.Bilinear, self.Feature_bias, Bias], name="out_afm")  # None * 1   
        
            # 计算损失
            if self.attention and self.lamda_attention > 0:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(self.lamda_attention)(self.weights['attention_W'])  # regulizer
            else:
                self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))     
            
            if self.lamde_em > 0:
                self.loss = self.loss + tf.contrib.layers.l2_regularizer(self.lamda_em)(self.weights['feature_embeddings'])  # regulizer
                
            # 优化方法
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

            # 初始化
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # 参数数目
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape() # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print ("#params: %d" %total_parameters)              
        
    def _initialize_weights(self):
        #初始化权重
        all_weights = dict()
        all_weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.features_M, self.em_factor], 0.0, 0.01),
            name='feature_embeddings')  # features_M * K
        all_weights['feature_bias'] = tf.Variable(
            tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')  # features_M * 1
        all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1

        # 注意力网络权重
        if self.attention:
            glorot = np.sqrt(2.0 / (self.attention_factor+self.em_factor))
            all_weights['attention_W'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.em_factor, self.attention_factor)), dtype=np.float32, name="attention_W")  # K * AK
            all_weights['attention_b'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.attention_factor)), dtype=np.float32, name="attention_b")  # 1 * AK
            all_weights['attention_p'] = tf.Variable(
                np.random.normal(loc=0, scale=1, size=(self.attention_factor)), dtype=np.float32, name="attention_p") # AK

        # prediction layer
        all_weights['prediction'] = tf.Variable(np.ones((self.em_factor, 1), dtype=np.float32))  # K * 1

        return all_weights        
    
    #BN操作定义
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.decay, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.decay, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # 对一个batch的数据进行训练
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.dropout_keep: self.keep_prob, self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # 从训练数据中随机产生一个batch的数据
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X , Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}
    
    def get_ordered_block_from_data(self, data, batch_size, index):  # 从训练数据中产生一个有序的batch的数据，用于测试阶段
        start_index = index*batch_size
        X , Y = [], []
        # get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append(data['Y'][i])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        return {'X': X, 'Y': Y}

    def shuffle_in_unison_scary(self, a, b): # 将数据打乱，特征和标签执行相同的打乱
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        
    def train(self, Train_data, Validation_data, Test_data):
        # 初始化性能检测
        if self.verbose > 0:
            t2 = time()
            init_train = self.evaluate(Train_data)
            init_valid = self.evaluate(Validation_data)
            init_test = self.evaluate(Test_data)
            print("Init: \t train=%.4f, validation=%.4f, test=%.4f [%.1f s]" %(init_train, init_valid, init_test, time()-t2))

        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X'], Train_data['Y'])
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            for i in range(total_batch):
                # 产生一个batch数据
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # 训练
                self.partial_fit(batch_xs)
            t2 = time()

            # 执行完一个epoch后，评价训练集、验证集和测试集误差
            train_result = self.evaluate(Train_data)
            valid_result = self.evaluate(Validation_data)
            test_result = self.evaluate(Test_data)

            self.train_loss.append(train_result)
            self.valid_loss.append(valid_result)
            self.test_loss.append(test_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [train_time%.1f s]\ttrain=%.4f, validation=%.4f, test=%.4f [eval_time%.1f s]" 
                      %(epoch+1, t2-t1, train_result, valid_result, test_result, time()-t2))
            if self.early_stop > 0 and self.eva_termination(self.valid_loss):
                print ("Early stop at %d based on validation result." %(epoch+1))
                break
        
        self.saver.save(self.sess, self.save_file) #保存训练后的模型
        
        plt.plot(range(0, self.epoch), self.train_loss, 'r-', label = 'Train_loss')
        plt.plot(range(0, self.epoch), self.valid_loss, 'b-', label = 'Valid_loss')        
        plt.plot(range(0, self.epoch), self.test_loss, 'y-', label = 'Test_loss')
        plt.title('训练集、验证集、测试集误差变化')
        plt.ylabel('Rmse')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
            
    def eva_termination(self, valid):
        if len(valid) > 5:
            if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                return True
        return False

    def evaluate(self, data):  # 评价给定的数据集
        num_example = len(data['Y'])
        # 取第一个batch
        batch_index = 0
        batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)
        y_pred = None
        
        while len(batch_xs['X']) > 0:
            num_batch = len(batch_xs['Y'])
            feed_dict = {self.train_features: batch_xs['X'], self.train_labels: [[y] for y in batch_xs['Y']], self.dropout_keep: [1], self.train_phase: False}
            a_exp, a_sum, a_out, batch_out = self.sess.run((self.attention_exp, self.attention_sum, self.attention_out, self.out), feed_dict=feed_dict)
            
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))
            #然后依次取其它batch数据
            batch_index += 1
            batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)

        y_true = np.reshape(data['Y'], (num_example,))

        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # 输出的最低值截断
        predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # 输出的最高值截断
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        return RMSE        
        
        
        
        
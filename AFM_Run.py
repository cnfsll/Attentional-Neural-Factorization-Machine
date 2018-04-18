# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:19:40 2018

@author: minjiang
"""

import tensorflow as tf
import LoadData as DATA
from AFM_Model import AFM
import numpy as np
from time import time

#定义参数
Path = 'D:/Recommender System/attentional_factorization_machine-master/Min/data/frappe/'
Save_file = 'D:/Recommender System/attentional_factorization_machine-master/Min/model/'
Epoch = 100
Batch_size = 128
Em_factor = 64
Attention_factor = 64
Keep_prob = '[0.7]'
Lr = 0.05
Lamda_em = 0.0
Lamda_attention = 15.0
Optimizer = 'AdagradOptimizer'
Verbose = 1
Bn = 1
Activation = 'relu'
Early_stop = 1
Attention = 1
Fields = 10
Decay = 0.99

# 读取数据
data = DATA.LoadData(Path)

activation_function = tf.nn.relu
if Activation == 'sigmoid':
    activation_function = tf.sigmoid
elif Activation == 'tanh':
    activation_function == tf.tanh
elif Activation == 'identity':
    activation_function = tf.identity

# 训练
t1 = time()
model = AFM(data.features_M, Em_factor, Attention_factor, Epoch, Batch_size, Lr,
                 eval(Keep_prob), Optimizer, Bn, Activation, Verbose, Early_stop,
                 Attention, Fields, Lamda_attention, Lamda_em, Decay, Save_file)
model.train(data.Train_data, data.Validation_data, data.Test_data)

# 找到使验证集误差最小的迭代次数
best_epoch  = np.argmin(model.valid_loss)
print ("Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]" 
       %(best_epoch+1, model.train_loss[best_epoch], model.valid_loss[best_epoch], model.test_loss[best_epoch], time()-t1))

# train:0.0906；valid：0.3253；test:0.3291 vaild-train = 0.2347 keep_prob = 0.8 Em_factor = 64 Lamda_attention = 10.0
# train:0.1361；valid：0.3272；test:0.3400 vaild-train = 0.1911 Em_factor = 32 减少了gap，但是训练集误差减小得不够多，使得最终的验证集误差比较大
# train:0.1100；valid：0.3225；test:0.3252 vaild-train = 0.2225 keep_prob = 0.7 Lamda_attention = 10.0 减少了过拟合， 从误差曲线上看，训练饱和程度比上述小，还可继续训练
# train:0.1127；valid：0.3222；test:0.3281 vaild-train = 0.2095 keep_prob = 0.7 Lamda_attention = 15.0 Epoch = 50减少了过拟合，但测试集误差增大了，从误差曲线上看，训练饱和程度比上述小，还可继续训练
# train:0.0843；valid：0.3171；test:0.3206 vaild-train = 0.2328 keep_prob = 0.7 Lamda_attention = 15.0 Epoch = 100 训练更加充分
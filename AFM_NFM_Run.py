# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 16:19:23 2018

@author: minjiang
"""

#import tensorflow as tf
import LoadData as DATA
from AFM_NFM_Model import AFM_NFM
import numpy as np
from time import time

#定义参数
Path = 'D:/Recommender System/attentional_factorization_machine-master/Min/data/frappe/'
Save_file = 'D:/Recommender System/attentional_factorization_machine-master/Min/model2/'
Epoch = 100
Batch_size = 128
Em_factor = 64
Attention_factor = 64
Layers = '[64]'
Keep_prob = '[0.7, 0.8]'
Lr = 0.001
Lamda_em = 0.0
Lamda_layers = 0.0
Lamda_attention = 15.0
Optimizer = 'AdamOptimizer' #此时出现错误，Lr = 0.05
#Optimizer = 'AdagradOptimizer'
Verbose = 1
Bn = 1
Activation = 'relu'
Early_stop = 1
Attention = 1
Fields = 10
Decay = 0.99

# 读取数据
data = DATA.LoadData(Path)

# 训练
t1 = time()
model = AFM_NFM(data.features_M, Em_factor, Attention_factor,eval(Layers),Epoch, Batch_size, Lr,
                 eval(Keep_prob), Optimizer, Bn, Activation, Verbose, Early_stop,
                 Attention, Fields, Lamda_attention, Lamda_em, Lamda_layers, Decay, Save_file)
model.train(data.Train_data, data.Validation_data, data.Test_data)

# 找到使验证集误差最小的迭代次数
best_epoch  = np.argmin(model.valid_loss)
print ("Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]" 
       %(best_epoch+1, model.train_loss[best_epoch], model.valid_loss[best_epoch], model.test_loss[best_epoch], time()-t1))

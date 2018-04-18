# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:16:56 2018

@author: minjiang
"""

import tensorflow as tf
import LoadData as DATA
import numpy as np
from sklearn.metrics import mean_squared_error
import math

#定义参数
Path = 'D:/Recommender System/attentional_factorization_machine-master/Min/data/frappe/'
Save_file = 'D:/Recommender System/attentional_factorization_machine-master/Min/model/'

data = DATA.LoadData(Path).Test_data

# 载入计算图
weight_saver = tf.train.import_meta_graph(Save_file + '.meta')
train_graph = tf.get_default_graph()

# load tensors 
feature_embeddings = train_graph.get_tensor_by_name('feature_embeddings:0')
feature_bias = train_graph.get_tensor_by_name('feature_bias:0')
bias = train_graph.get_tensor_by_name('bias:0')
afm = train_graph.get_tensor_by_name('afm:0')
out_of_afm = train_graph.get_tensor_by_name('out_afm:0')
    
# placeholders for afm
train_features = train_graph.get_tensor_by_name('train_features:0')
train_labels = train_graph.get_tensor_by_name('train_labels:0')
dropout_keep = train_graph.get_tensor_by_name('dropout_keep:0')
train_phase = train_graph.get_tensor_by_name('train_phase:0')

# 恢复 session
sess = tf.Session()
weight_saver.restore(sess, Save_file)

# 评价数据集
num_example = len(data['Y'])
feed_dict = {train_features: data['X'], train_labels: [[y] for y in data['Y']], dropout_keep: [1.0], train_phase: False}
predictions = sess.run((out_of_afm), feed_dict=feed_dict)

# calculate rmse
y_pred_afm = np.reshape(predictions, (num_example,))
y_true = np.reshape(data['Y'], (num_example,))

predictions_bounded = np.maximum(y_pred_afm, np.ones(num_example) * min(y_true))  # bound the lower values
predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))

print("Test RMSE: %.4f"%(RMSE))

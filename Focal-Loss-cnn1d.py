#!/usr/bin/env python
# coding: utf-8

# In[1]:


exec('from __future__ import absolute_import, division, print_function, unicode_literals')

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Reading in and Preprocessing Data

# In[2]:


train = pd.read_csv('aps_failure_training_set_processed_8bit.csv')
test = pd.read_csv('aps_failure_test_set_processed_8bit.csv')


# In[3]:


test_set = test.copy()

train_copy = train.copy()
train_set = train_copy.sample(frac=0.9, random_state = 0)
val_set = train_copy.drop(train_set.index)


# In[4]:


train_set.head()


# In[5]:


train_set['class'] = (train_set['class'] > 0).astype(int)
val_set['class'] = (val_set['class'] > 0).astype(int)
test_set['class'] = (test_set['class'] > 0).astype(int)


# In[6]:


# SMOTE TO Balance Dataset

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 


print('Original dataset shape %s' % Counter(train_set['class'].to_numpy()))
sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(train_set.to_numpy(), train_set['class'].to_numpy())
print('Resampled dataset shape %s' % Counter(y_res))

print('Original dataset shape %s' % Counter(val_set['class'].to_numpy()))
sm = SMOTE(random_state=42)
X_val_res, y_val_res = sm.fit_resample(val_set.to_numpy(), val_set['class'].to_numpy())
print('Resampled dataset shape %s' % Counter(y_val_res))


# In[7]:


y_res_pd = pd.Series(y_res)
y_train = pd.get_dummies(y_res_pd)

y_val_res_pd = pd.Series(y_val_res)
y_val = pd.get_dummies(y_val_res_pd)

train_set = pd.DataFrame(X_res, columns=list(train_set.columns.values))
val_set = pd.DataFrame(X_val_res, columns=list(val_set.columns.values))


# In[8]:


y_test = pd.get_dummies(test_set['class'])



train_set.drop(columns=['class'],inplace=True)
val_set.drop(columns=['class'],inplace=True)
test_set.drop(columns=['class'],inplace=True)


# In[9]:


y_train.describe()


# In[10]:


train_set_np = train_set.to_numpy()
y_train_np = y_train.to_numpy().astype('float32')
val_set_np = val_set.to_numpy()
y_val_np = y_val.to_numpy().astype('float32')
test_set_np = test_set.to_numpy()
y_test_np = y_test.to_numpy().astype('float32')


# In[11]:


y_test_np.shape


# In[12]:


# Reshape for CNN1D 
train_set_cnn = train_set_np.reshape(106184, 170, 1)
val_set_cnn = val_set_np.reshape(11816, 170, 1)
test_set_cnn = test_set_np.reshape(16000, 170, 1)


# ## Focal Loss Implementation 

# In[13]:


# Focal Loss function

class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=5., alpha=.75,
                 reduction=keras.losses.Reduction.AUTO, name='focal_loss'):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})
        """
        super(FocalLoss, self).__init__(reduction=reduction,
                                        name=name)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        """
        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(
            tf.subtract(1., model_out), self.gamma))
        fl = tf.multiply(self.alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)


# ## Define Models (Focal Loss vs Cross Entropy) 

# In[14]:


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]
###############################################################################################################
# *************************************************************************************************************
###############################################################################################################

model_cnn_fl = tf.keras.Sequential()
model_cnn_fl.add(layers.Conv1D(10, 1, activation='relu', input_shape=(170, 1),data_format='channels_last'))
model_cnn_fl.add(layers.Flatten())
model_cnn_fl.add(layers.Dense(64, activation='relu', bias_initializer=tf.constant_initializer(0.01),
                              kernel_regularizer=regularizers.l2(0.0001)))
model_cnn_fl.add(layers.Dropout(0.5))
model_cnn_fl.add(layers.Dense(2, activation='softmax'))

model_cnn_fl.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=FocalLoss(),
              metrics=METRICS)

###############################################################################################################
# *************************************************************************************************************
###############################################################################################################

model_cnn_bc = tf.keras.Sequential()
model_cnn_bc.add(layers.Conv1D(10, 1, activation='relu', input_shape=(170, 1),data_format='channels_last'))
model_cnn_bc.add(layers.Flatten())
model_cnn_bc.add(layers.Dense(64, activation='relu', bias_initializer=tf.constant_initializer(0.01),
                              kernel_regularizer=regularizers.l2(0.0001)))
model_cnn_bc.add(layers.Dropout(0.5))
model_cnn_bc.add(layers.Dense(2, activation='softmax'))

model_cnn_bc.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='binary_crossentropy',
              metrics=METRICS)


# ## Training Models 

# In[15]:


model_cnn_fl.fit(train_set_cnn,y_train_np, batch_size=256, epochs=20, validation_data=(val_set_cnn,y_val_np))


# In[16]:


model_cnn_bc.fit(train_set_cnn,y_train_np, batch_size=256, epochs=20, validation_data=(val_set_cnn,y_val_np))


# ## Results 

# In[17]:


def final_cost(result, loss_name):
    Cost_1 = 10
    Cost_2 = 500
    tn, fp, fn, tp = confusion_matrix(y_test_ori.to_numpy(), result.to_numpy()).ravel()
    cost = Cost_1*fp + Cost_2*fn
    print('tn: ',tn, 'fp: ',fp,'fn: ',fn,'tp: ',tp)
    print('Final Cost using ',loss_name,' is: ',cost)
    print('Total number of misclassificactions: ', fp+fn)


# In[20]:


from sklearn.metrics import confusion_matrix

result_cnn_fl = model_cnn_fl.predict(test_set_cnn)
result_cnn_bc = model_cnn_bc.predict(test_set_cnn)
# print('FL CNN1D Output Shape:', result_cnn_fl.shape)
# print('BC CNN1D Output Shape:', result_cnn_bc.shape)

assert result_cnn_fl.shape == result_cnn_bc.shape

result_cnn_fl_pd = pd.DataFrame(np.rint(result_cnn_fl))
result_cnn_fl_stack_pd = result_cnn_fl_pd.stack()
result_cnn_fl_ori = pd.Series(pd.Categorical(result_cnn_fl_stack_pd[result_cnn_fl_stack_pd!=0].index.get_level_values(1)))

result_cnn_bc_pd = pd.DataFrame(np.rint(result_cnn_bc))
result_cnn_bc_stack_pd = result_cnn_bc_pd.stack()
result_cnn_bc_ori = pd.Series(pd.Categorical(result_cnn_bc_stack_pd[result_cnn_bc_stack_pd!=0].index.get_level_values(1)))

y_test_stack = y_test.stack()
y_test_ori = pd.Series(pd.Categorical(y_test_stack[y_test_stack!=0].index.get_level_values(1)))

print('FL: ',confusion_matrix(y_test_ori.to_numpy(),result_cnn_fl_ori.to_numpy()))
print('BC: ',confusion_matrix(y_test_ori.to_numpy(),result_cnn_bc_ori.to_numpy()))

final_cost(result_cnn_fl_ori, 'Focal Loss - CNN1D')
final_cost(result_cnn_bc_ori, 'Binary Cross Entropy - CNN1D')


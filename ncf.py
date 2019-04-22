#!/usr/bin/env python
# coding: utf-8


import sys
import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from ncf_singlenode import NCF
from dataset import Dataset as NCFDataset
from notebook_utils import is_jupyter
from python_splitters import python_chrono_split
from python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, recall_at_k, get_top_k_items)

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))


# In[15]:


# top k items to recommend
TOP_K = 30

# Select Movielens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Model parameters
EPOCHS = 30
BATCH_SIZE = 256


# In[16]:


# In[33]:


df=pd.read_csv("video_games_ratings_5_fact.csv").reset_index()
print(df.head(1))


# In[34]:


train, test = python_chrono_split(df, 0.90)


# In[35]:


data = NCFDataset(train=train, test=test, seed=42)


# In[36]:


model = NCF (
    n_users=data.n_users, 
    n_items=data.n_items,
    model_type="NeuMF",
    n_factors=4,
    layer_sizes=[8,8,4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=10,
)


# In[ ]:


start_time = time.time()

model.fit(data)

train_time = time.time() - start_time

print("Took {} seconds for training.".format(train_time))


# In[ ]:


start_time = time.time()

users, items, preds = [], [], []
item = list(train.itemID.unique())
for user in train.userID.unique():
    user = [user] * len(item) 
    users.extend(user)
    items.extend(item)
    preds.extend(list(model.predict(user, item, is_list=True)))

all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

all_predictions_include_training = merged.drop('rating', axis=1)

test_time = time.time() - start_time
print("Took {} seconds for prediction.".format(test_time))


# In[ ]:


eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')
print(time.time())


# In[ ]:


eval_map = map_at_k(train, all_predictions_include_training, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(train, all_predictions_include_training, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(train, all_predictions_include_training, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(train, all_predictions_include_training, col_prediction='prediction', k=TOP_K)

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')
print(time.time())


# In[ ]:





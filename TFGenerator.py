#/usr/bin/python

#imports
import numpy as np 
import tensorflow as tf


#Load normalized train/test data and labels. Stored in npy files.
data_train=np.load('/mnt/home/dembskic/data_train.npy')
data_test=np.load('/mnt/home/dembskic/data_test.npy')
labels_train=np.load('/mnt/home/dembskic/labels_train.npy')
labels_test=np.load('/mnt/home/dembskic/labels_test.npy')

#Python generator zips together data and label images
def train_generator():
    for i,j in zip(data_train, labels_train):
        yield i,j
        
#Ibid for test Generator        
def test_generator():
    for i,j in zip(data_test, labels_test):
        yield i,j

#Set batch size, buffer for shuffling 
BUFFER_SIZE=10
BATCH_SIZE=12

#Create and return shuffled, batched Tensorflow datasets for train and test. Contain both data and labels. 
def Get_Data():
    train_dataset = tf.data.Dataset.from_generator(train_generator, (tf.float32, tf.float32)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_generator(test_generator, (tf.float32, tf.float32)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    return train_dataset, test_dataset

#Get_Data() is called in the main training script to create the data and labels for the model. 

#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from skimage.util import view_as_windows #for 
from skimage.measure import block_reduce #For maximum pooling
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pytictoc import TicToc #timer purpose
import pickle #Save trained model purpose

#Provided code
from cross_entropy import Cross_Entropy
from lag import LAG
from llsr import LLSR as myLLSR
from pixelhop2 import Pixelhop2

def CIFAR10DatasetCreator():
    '''
    Create train and test data set from CIFAR-10 data base and return them
    Input: None
    
    Output: 
    1.X_train : 50000 images from CIFAR-10 Data base. (50000,32,32,3) size numpy
    2.Y_train : 50000 corresponding labels to the 50000 images. 1x50000 size numpy. labels of 10 classes are 0 to 9
    3.X_test : 10000 images from CIFAR-10 Data base. (10000,32,32,3) size numpy
    4.Y_test : 10000 corresponding labels to the 10000 images. 1x10000 size numpy
    '''
    #This code is motivated by training a classifier tutorial. Available Online:
    #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    
    #compose ToTensor and Normalize transfroms
    transformInfo = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
    #load train set
    trainSet_CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transformInfo)
    #load test set
    testSet_CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transformInfo)
    
    X_train = trainSet_CIFAR10.data
    Y_train = np.asarray(trainSet_CIFAR10.targets)
    X_test = testSet_CIFAR10.data
    Y_test = np.asarray(testSet_CIFAR10.targets)
    
    return X_train, Y_train, X_test, Y_test

def datasetPreprocessor(X):
    '''
    Preprocess X to convert data type from uint to float and rescale within 0-1
    input: dataset X (numOfSample x height x width x numChannel). e.g., 50000 x 32 x 32 x 3 
    output: preprocessed dataset X
    '''
    maxIntensity = 255
    preProcessedX = X.astype(float) #convert data type from uint to float
    preProcessedX = np.true_divide(preProcessedX,maxIntensity) #rescale intensity from 0-255 to 0-1
    
    return preProcessedX

# example callback function for collecting patches and its inverse
def Shrink(X, shrinkArg):
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    depth = shrinkArg['depth']
    ch = X.shape[-1]
    
    if(depth == 1):
        #print('no max pooling')
        X = view_as_windows(X, (1,win,win,ch), (1,stride,stride,ch))
    else:
        # at this point X is from previous layer(i.e., depth) e.g. (#img,28,28,K1)
        X = MaxpoolingOperator(X) #Need to maxpooling before construct neighborhoold for the next depth
        X = view_as_windows(X, (1,win,win,ch), (1,stride,stride,ch))
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

# example callback function for how to concate features from different hops
def Concat(X, concatArg):
    return X

def MaxpoolingOperator(X):
    #print('Maxpooling')
    numOfImg = int(X.shape[0]) #e.g. 10000 images
    numOfChannel = int(X.shape[-1]) #negative indexing #e.g. K1 = 41 channels 
    pooledHeight = int(X.shape[1] / 2) #e.g. 28/2 = 14
    pooledWidth = int(X.shape[2] / 2) #e.g. 28/2 = 14
    
    maxpooledX = np.zeros((numOfImg,pooledHeight,pooledWidth,numOfChannel))
    print(maxpooledX.shape)
    
    for imgIndex in range(0,numOfImg):
        for channelIndex in range(0,numOfChannel):
            maxpooledX[imgIndex,:,:,channelIndex] = block_reduce(X[imgIndex,:,:,channelIndex], block_size=(2,2), func = np.max)
    
    return maxpooledX


def pixelHop2TransBatchProcessor(X,pixelHop2Model):
    '''
    perform pixelHop2 transfrom batchwise. 
    input:
    X: train or test data set. e.g. (50000,32,32,3) or (10000,32,32,3)
    output:
    totalOut:transfromed data set at each unit. e.g. [out_at_1st_unit,out_at_2nd_unit,out_at_3rd_unit]
    out_at_nth_unit: c/w saab output at each unit. e.g. (10000,28,28,42) for first unit for test data set
    '''
    
    
    numOfBatch = int(X.shape[0]/1000) #e.g. 50000/1000=50 batches
    #First batch
    firstBatchLowIndex = int(0)
    firstBatchUpperIndex = int(1000)
    firstBatch = X[firstBatchLowIndex:firstBatchUpperIndex,:,:,:]
    
    firstBatchOut = pixelHop2Model.transform(firstBatch)
    out_at_1st_unit = firstBatchOut[0]
    out_at_2nd_unit = firstBatchOut[1]
    out_at_3rd_unit = firstBatchOut[2]
    
    
    #Rest batch
    for batchCounter in range(2,numOfBatch+1):
        crrLowIndex = int( 1000*(batchCounter - 1) )
        crrUpperIndex = int( 999 + 1000*(batchCounter - 1) + 1 )
        
        crrBatch = X[crrLowIndex:crrUpperIndex,:,:,:]
        print('currentBatch.shape: ')
        print(crrBatch.shape)
        
        crrOut = pixelHop2Model.transform(crrBatch)
        out_at_1st_unit = np.concatenate((out_at_1st_unit,crrOut[0]),axis=0)
        out_at_2nd_unit = np.concatenate((out_at_2nd_unit,crrOut[1]),axis=0)
        out_at_3rd_unit = np.concatenate((out_at_3rd_unit,crrOut[2]),axis=0)
        print("Current batch: ")
        print(batchCounter)
        
    totalOut = [out_at_1st_unit, out_at_2nd_unit, out_at_3rd_unit]
    
    return totalOut

def pixelHop2TransBatchProcessor_Unit1(X,pixelHop2Model):
    '''
    perform pixelHop2 transfrom batchwise. 
    input:
    X: train or test data set. e.g. (50000,32,32,3) or (10000,32,32,3)
    output:
    totalOut:transfromed data set at each unit. e.g. [out_at_1st_unit,out_at_2nd_unit,out_at_3rd_unit]
    out_at_nth_unit: c/w saab output at each unit. e.g. (10000,28,28,42) for first unit for test data set
    '''
    
    batchSize = 50
    numOfBatch = int(X.shape[0]/batchSize) #e.g. 50000/50=1000 batches
    #First batch
    firstBatchLowIndex = int(0)
    firstBatchUpperIndex = int(batchSize)
    firstBatch = X[firstBatchLowIndex:firstBatchUpperIndex,:,:,:]
    
    firstBatchOut = pixelHop2Model.transform(firstBatch)
    out_at_1st_unit = firstBatchOut[0]
    out_at_2nd_unit = firstBatchOut[1]
    out_at_3rd_unit = firstBatchOut[2]
    
    
    
    #Rest batch
    for batchCounter in range(2,numOfBatch+1):
        crrLowIndex = int( batchSize*(batchCounter - 1) )
        crrUpperIndex = int( 49 + batchSize*(batchCounter - 1) + 1 )#49 is from batch size 50
        
        crrBatch = X[crrLowIndex:crrUpperIndex,:,:,:]
        
        
        crrOut = pixelHop2Model.transform(crrBatch)
        out_at_1st_unit = np.concatenate((out_at_1st_unit,crrOut[0]),axis=0)
        #out_at_2nd_unit = np.concatenate((out_at_2nd_unit,crrOut[1]),axis=0)
        #out_at_3rd_unit = np.concatenate((out_at_3rd_unit,crrOut[2]),axis=0)
        print("Current batch_unit1: ")
        print(batchCounter)
        
    
    
    return out_at_1st_unit

def pixelHop2TransBatchProcessor_Unit2(X,pixelHop2Model):
    '''
    perform pixelHop2 transfrom batchwise. 
    input:
    X: train or test data set. e.g. (50000,32,32,3) or (10000,32,32,3)
    output:
    totalOut:transfromed data set at each unit. e.g. [out_at_1st_unit,out_at_2nd_unit,out_at_3rd_unit]
    out_at_nth_unit: c/w saab output at each unit. e.g. (10000,28,28,42) for first unit for test data set
    '''
    
    batchSize = 50
    numOfBatch = int(X.shape[0]/batchSize) #e.g. 50000/50=1000 batches
    #First batch
    firstBatchLowIndex = int(0)
    firstBatchUpperIndex = int(batchSize)
    firstBatch = X[firstBatchLowIndex:firstBatchUpperIndex,:,:,:]
    
    firstBatchOut = pixelHop2Model.transform(firstBatch)
    out_at_1st_unit = firstBatchOut[0]
    out_at_2nd_unit = firstBatchOut[1]
    out_at_3rd_unit = firstBatchOut[2]
    
    
    #Rest batch
    for batchCounter in range(2,numOfBatch+1):
        crrLowIndex = int( batchSize*(batchCounter - 1) )
        crrUpperIndex = int( 49 + batchSize*(batchCounter - 1) + 1 )#49 is from batch size 50
        
        crrBatch = X[crrLowIndex:crrUpperIndex,:,:,:]
        
        
        crrOut = pixelHop2Model.transform(crrBatch)
        #out_at_1st_unit = np.concatenate((out_at_1st_unit,crrOut[0]),axis=0)
        out_at_2nd_unit = np.concatenate((out_at_2nd_unit,crrOut[1]),axis=0)
        #out_at_3rd_unit = np.concatenate((out_at_3rd_unit,crrOut[2]),axis=0)
        print("Current batch_unit2: ")
        print(batchCounter)
        
    
    
    return out_at_2nd_unit

def pixelHop2TransBatchProcessor_Unit3(X,pixelHop2Model):
    '''
    perform pixelHop2 transfrom batchwise. 
    input:
    X: train or test data set. e.g. (50000,32,32,3) or (10000,32,32,3)
    output:
    totalOut:transfromed data set at each unit. e.g. [out_at_1st_unit,out_at_2nd_unit,out_at_3rd_unit]
    out_at_nth_unit: c/w saab output at each unit. e.g. (10000,28,28,42) for first unit for test data set
    '''
    
    batchSize = 50
    numOfBatch = int(X.shape[0]/batchSize) #e.g. 50000/50=1000 batches
    #First batch
    firstBatchLowIndex = int(0)
    firstBatchUpperIndex = int(batchSize)
    firstBatch = X[firstBatchLowIndex:firstBatchUpperIndex,:,:,:]
    
    firstBatchOut = pixelHop2Model.transform(firstBatch)
    out_at_1st_unit = firstBatchOut[0]
    out_at_2nd_unit = firstBatchOut[1]
    out_at_3rd_unit = firstBatchOut[2]
    
    
    #Rest batch
    for batchCounter in range(2,numOfBatch+1):
        crrLowIndex = int( batchSize*(batchCounter - 1) )
        crrUpperIndex = int( 49 + batchSize*(batchCounter - 1) + 1 ) #49 is from batch size 50
        
        crrBatch = X[crrLowIndex:crrUpperIndex,:,:,:]
        
        
        crrOut = pixelHop2Model.transform(crrBatch)
        #out_at_1st_unit = np.concatenate((out_at_1st_unit,crrOut[0]),axis=0)
        #out_at_2nd_unit = np.concatenate((out_at_2nd_unit,crrOut[1]),axis=0)
        out_at_3rd_unit = np.concatenate((out_at_3rd_unit,crrOut[2]),axis=0)
        print("Current batch_unit3: ")
        print(batchCounter)
        
    
    
    return out_at_3rd_unit


def pixelHop2TransBatchProcessor_Unit1_V2(X,pixelHop2Model):
    '''
    perform pixelHop2 transfrom batchwise. 
    input:
    X: train or test data set. e.g. (50000,32,32,3) or (10000,32,32,3)
    output:
    totalOut:transfromed data set at each unit. e.g. [out_at_1st_unit,out_at_2nd_unit,out_at_3rd_unit]
    out_at_nth_unit: c/w saab output at each unit. e.g. (10000,28,28,42) for first unit for test data set
    '''
    
    batchSize = 500
    numOfBatch = int(X.shape[0]/batchSize) #e.g. 50000/50=1000 batches
    #First batch
    firstBatchLowIndex = int(0)
    firstBatchUpperIndex = int(batchSize)
    firstBatch = X[firstBatchLowIndex:firstBatchUpperIndex,:,:,:]
    
    firstBatchOut = pixelHop2Model.transform(firstBatch)
    out_at_1st_unit = firstBatchOut[0]
    out_at_2nd_unit = firstBatchOut[1]
    out_at_3rd_unit = firstBatchOut[2]
    
    out_rest_list = [] #it will contain outputs from 2nd batch to 1000th batch
    
    #Rest batches
    for batchCounter in range(2,numOfBatch+1):
        crrLowIndex = int( batchSize*(batchCounter - 1) )
        crrUpperIndex = int( (batchSize-1) + batchSize*(batchCounter - 1) + 1 ) #49 is from batch size 50 - 1
        
        crrBatch = X[crrLowIndex:crrUpperIndex,:,:,:]
        
        
        crrOut = pixelHop2Model.transform(crrBatch)
        #out_at_1st_unit = np.concatenate((out_at_1st_unit,crrOut[0]),axis=0)
        #out_at_2nd_unit = np.concatenate((out_at_2nd_unit,crrOut[1]),axis=0)
        #out_at_3rd_unit = np.concatenate((out_at_3rd_unit,crrOut[2]),axis=0)
        out_rest_list.append(crrOut[0])
        
        print("Current batch_unit1: ")
        print(batchCounter)
        
    #Concatenating process
    print('Concatenating process')
    for batchCounter1 in range(0,numOfBatch-1):
        #2~1000
        out_at_1st_unit = np.concatenate((out_at_1st_unit,out_rest_list[batchCounter1]),axis=0)
        print("Current batch_unit1_concatenating process: ")
        print(batchCounter1)
    print('Concatenating process is done')
    
    return out_at_1st_unit

def pixelHop2TransBatchProcessor_Unit2_V2(X,pixelHop2Model):
    '''
    perform pixelHop2 transfrom batchwise. 
    input:
    X: train or test data set. e.g. (50000,32,32,3) or (10000,32,32,3)
    output:
    totalOut:transfromed data set at each unit. e.g. [out_at_1st_unit,out_at_2nd_unit,out_at_3rd_unit]
    out_at_nth_unit: c/w saab output at each unit. e.g. (10000,28,28,42) for first unit for test data set
    '''
    
    batchSize = 500
    numOfBatch = int(X.shape[0]/batchSize) #e.g. 50000/50=1000 batches
    #First batch
    firstBatchLowIndex = int(0)
    firstBatchUpperIndex = int(batchSize)
    firstBatch = X[firstBatchLowIndex:firstBatchUpperIndex,:,:,:]
    
    firstBatchOut = pixelHop2Model.transform(firstBatch)
    out_at_1st_unit = firstBatchOut[0]
    out_at_2nd_unit = firstBatchOut[1]
    out_at_3rd_unit = firstBatchOut[2]
    
    out_rest_list = [] #it will contain outputs from 2nd batch to 1000th batch
    
    #Rest batches
    for batchCounter in range(2,numOfBatch+1):
        crrLowIndex = int( batchSize*(batchCounter - 1) )
        crrUpperIndex = int( (batchSize-1) + batchSize*(batchCounter - 1) + 1 ) #49 is from batch size 50 - 1
        
        crrBatch = X[crrLowIndex:crrUpperIndex,:,:,:]
        
        
        crrOut = pixelHop2Model.transform(crrBatch)
        #out_at_1st_unit = np.concatenate((out_at_1st_unit,crrOut[0]),axis=0)
        #out_at_2nd_unit = np.concatenate((out_at_2nd_unit,crrOut[1]),axis=0)
        #out_at_3rd_unit = np.concatenate((out_at_3rd_unit,crrOut[2]),axis=0)
        out_rest_list.append(crrOut[1])
        
        print("Current batch_unit2: ")
        print(batchCounter)
        
    #Concatenating process
    print('Concatenating process')
    for batchCounter1 in range(0,numOfBatch-1):
        #2~1000
        out_at_2nd_unit = np.concatenate((out_at_2nd_unit,out_rest_list[batchCounter1]),axis=0)
        print("Current batch_unit2_concatenating process: ")
        print(batchCounter1)
    print('Concatenating process is done')
    
    return out_at_2nd_unit

def pixelHop2TransBatchProcessor_Unit3_V2(X,pixelHop2Model):
    '''
    perform pixelHop2 transfrom batchwise. 
    input:
    X: train or test data set. e.g. (50000,32,32,3) or (10000,32,32,3)
    output:
    totalOut:transfromed data set at each unit. e.g. [out_at_1st_unit,out_at_2nd_unit,out_at_3rd_unit]
    out_at_nth_unit: c/w saab output at each unit. e.g. (10000,28,28,42) for first unit for test data set
    '''
    
    batchSize = 500
    numOfBatch = int(X.shape[0]/batchSize) #e.g. 50000/50=1000 batches
    #First batch
    firstBatchLowIndex = int(0)
    firstBatchUpperIndex = int(batchSize)
    firstBatch = X[firstBatchLowIndex:firstBatchUpperIndex,:,:,:]
    
    firstBatchOut = pixelHop2Model.transform(firstBatch)
    out_at_1st_unit = firstBatchOut[0]
    out_at_2nd_unit = firstBatchOut[1]
    out_at_3rd_unit = firstBatchOut[2]
    
    out_rest_list = [] #it will contain outputs from 2nd batch to 1000th batch
    
    #Rest batches
    for batchCounter in range(2,numOfBatch+1):
        crrLowIndex = int( batchSize*(batchCounter - 1) )
        crrUpperIndex = int( (batchSize-1) + batchSize*(batchCounter - 1) + 1 ) #49 is from batch size 50 - 1
        
        crrBatch = X[crrLowIndex:crrUpperIndex,:,:,:]
        
        
        crrOut = pixelHop2Model.transform(crrBatch)
        #out_at_1st_unit = np.concatenate((out_at_1st_unit,crrOut[0]),axis=0)
        #out_at_2nd_unit = np.concatenate((out_at_2nd_unit,crrOut[1]),axis=0)
        #out_at_3rd_unit = np.concatenate((out_at_3rd_unit,crrOut[2]),axis=0)
        out_rest_list.append(crrOut[2])
        
        print("Current batch_unit3: ")
        print(batchCounter)
        
    #Concatenating process
    print('Concatenating process')
    for batchCounter1 in range(0,numOfBatch-1):
        #2~1000
        out_at_3rd_unit = np.concatenate((out_at_3rd_unit,out_rest_list[batchCounter1]),axis=0)
        print("Current batch_unit3_concatenating process: ")
        print(batchCounter1)
    print('Concatenating process is done')
    
    return out_at_3rd_unit


def featureSelector(X,Y):
    '''
    Select 50% features that has the lowest value of cross entropy
    input:
    X: output from max-pooling. Typically it has (numOfImages,pooled dimension, pooled dimension, K) size. e.g., (50000,14,14,41) at unit 1
    Y: labels corresponding to X
    output:
    selectedFeatures: selected 50% features 
    selectedFeaturesIndeces: indeces of the selected features 
    '''
    
    reshapedPooledX = X.reshape((len(X),-1))
    
    crossEntropyObj = Cross_Entropy(num_class = 10, num_bin = 10) 
    crossEntropyValStorage = np.zeros(reshapedPooledX.shape[-1])
    for storageIndex in range(reshapedPooledX.shape[-1]): #12 minutes for 100 images and 8036(14*14*41) dimension
        crossEntropyValStorage[storageIndex] = crossEntropyObj.KMeans_Cross_Entropy(reshapedPooledX[:,storageIndex].reshape(-1,1), Y)
    print("calculation is done \n")
    
    sortedIndex = np.argsort(crossEntropyValStorage) #return indeces that would be the indeces when the array is sorted
    numOfNs = int(sortedIndex.shape[0]/2) #select 50 persent of lowest values
    #numOfNs = int(1000) #Select 1000 lowest features 
    selectedFeaturesIndex = sortedIndex[0:numOfNs]
    
    
    numOfImgs = int(reshapedPooledX.shape[0]) #e.g.50000 images
    reducedFeatureImgs = np.zeros((numOfImgs,numOfNs))
    
    #Select features for each image
    for imgIndex in range(numOfImgs):
        for selectedFeatureCount in range(numOfNs):
            curr_selectedFeatureIndex = selectedFeaturesIndex[selectedFeatureCount]
            reducedFeatureImgs[imgIndex,selectedFeatureCount] = reshapedPooledX[imgIndex,curr_selectedFeatureIndex]
        
    return selectedFeaturesIndex, reducedFeatureImgs, numOfNs


def featureSelectorBinMethod(X,Y):
    '''
    Select 50% features that has the lowest value of cross entropy with bin method
    input:
    X: output from max-pooling. Typically it has (numOfImages,pooled dimension, pooled dimension, K) size. e.g., (50000,14,14,41) at unit 1
    Y: labels corresponding to X
    output:
    selectedFeatures: selected 50% features 
    selectedFeaturesIndeces: indeces of the selected features 
    '''
    
    reshapedPooledX = X.reshape((len(X),-1))
    
    crossEntropyObj = Cross_Entropy(num_class = 10, num_bin = 10) 
    crossEntropyValStorage = np.zeros(reshapedPooledX.shape[-1])
    for storageIndex in range(reshapedPooledX.shape[-1]): #12 minutes for 100 images and 8036(14*14*41) dimension
        crossEntropyValStorage[storageIndex] = crossEntropyObj.bin_process(reshapedPooledX[:,storageIndex].reshape(-1,1), Y)
    print("calculation is done \n")
    
    sortedIndex = np.argsort(crossEntropyValStorage) #return indeces that would be the indeces when the array is sorted
    numOfNs = int(sortedIndex.shape[0]/2) #select 50 persent of lowest values
    #numOfNs = int(1000) #Select 1000 lowest features 
    selectedFeaturesIndex = sortedIndex[0:numOfNs]
    
    
    numOfImgs = int(reshapedPooledX.shape[0]) #e.g.50000 images
    reducedFeatureImgs = np.zeros((numOfImgs,numOfNs))
    
    #Select features for each image
    for imgIndex in range(numOfImgs):
        for selectedFeatureCount in range(numOfNs):
            curr_selectedFeatureIndex = selectedFeaturesIndex[selectedFeatureCount]
            reducedFeatureImgs[imgIndex,selectedFeatureCount] = reshapedPooledX[imgIndex,curr_selectedFeatureIndex]
        
    return selectedFeaturesIndex, reducedFeatureImgs, numOfNs


def featureSelectorWithoutCrossEnt(pooledOut,selectedFeaturesIndeces,numOfNs):
    '''
    Flatten spatial and spectral features and select features that gave low cross entropy value in training 
    inputs:
    pooledOut: output from max-pooling process
    selectedIndeces: selected indeces in training
    numONs: number of selected indeces in training
    
    output:
    reducedFeatureImgs: images with reduced number of features
    '''
    
    reshapedPooledOut = pooledOut.reshape( (len(pooledOut),-1) )
    
    numOfImgs = int(reshapedPooledOut.shape[0])
    reducedFeatureImgs = np.zeros((numOfImgs,numOfNs))
    
    #Select features for each image
    for imgIndex in range(numOfImgs):
        for selectedFeatureCount in range(numOfNs):
            curr_selectedFeatureIndex = selectedFeaturesIndeces[selectedFeatureCount]
            reducedFeatureImgs[imgIndex,selectedFeatureCount] = reshapedPooledOut[imgIndex,curr_selectedFeatureIndex]
            
    return reducedFeatureImgs


def featureConcatenator(finalFeature_unit1, finalFeature_unit2, finalFeature_unit3):
    '''
    Concatenate features colunmwise and make it final feature
    input: 
    finalFeature_unit1: final feature from unit1 after LAG
    finalFeature_unit2: final feature from unit2 after LAG
    finalFeature_unit3: final feature from unit3 after LAG
    
    output:
    finalFeature: concatenated final feature. It will be used in classification task
    '''
    
    concatenated2FeatureVectors = np.concatenate((finalFeature_unit1,finalFeature_unit2), axis = 1)
    finalFeature = np.concatenate((concatenated2FeatureVectors, finalFeature_unit3), axis = 1)
    
    return finalFeature


##########################################Main##############################
#Create train and test data set from CIFAR-10 data base
X_train, Y_train, X_test, Y_test = CIFAR10DatasetCreator() #50000 images for training, 10000 images for test #fixed split

#Pre process images(data type converting and rescaling)
pre_X_train = datasetPreprocessor(X_train) #pre_ means pre-processed_. 
pre_X_test = datasetPreprocessor(X_test)

#Define Saab, Shrink, and Concat arguments
SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None, 'cw':False},
           {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw': True},
           {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw': True}]
shrinkArgs = [{'func':Shrink, 'win':5 , 'stride':1 ,'depth':1},
             {'func': Shrink, 'win':5, 'stride': 1 ,'depth':2},
             {'func': Shrink, 'win':5, 'stride': 1 ,'depth':3}]
concatArg = {'func':Concat}

timerObj = TicToc() #create timer class

#Extract features from 50000 train images->Use only 20000 train image, my computer cannot carry all
loaded_pixelHop2_6000 = pickle.load(open('pixelHop2_Cla_6000.sav','rb'))


#This part depends on the number of data samples used in Module 2
trainDataSize = 1/4
testDataSize = 1 - trainDataSize
subXTrain4, _ , subYTrain4 , _ = train_test_split(pre_X_train, Y_train, test_size = testDataSize, random_state = 0)
#subXTrain4 = pre_X_train
#subYTrain4 = Y_train



#train set
#Use only 20000 train image, my computer cannot carry all (it takes 27 mins = Elapsed time is 1655.533442 seconds.)
#timerObj.tic()
#Use batch processing for a large number of data set. Otherwise, use loaded_pixelHop2_6000 directly.
#module1OutStorage_train = pixelHop2TransBatchProcessor(subXTrain4,loaded_pixelHop2_6000)
#module1OutStorage_train = loaded_pixelHop2_6000.transform(subXTrain4)
#timerObj.toc()
#Elapsed time is 56.481034 seconds.

#Seperate extracted features from each hop
timerObj.tic()
module1Out1_train = pixelHop2TransBatchProcessor_Unit1_V2(subXTrain4,loaded_pixelHop2_6000)
print("transform is done for train for unit1:")#Elapsed time is 283.387480 seconds.
timerObj.toc()


timerObj.tic()
module1Out2_train = pixelHop2TransBatchProcessor_Unit2_V2(subXTrain4,loaded_pixelHop2_6000)
print("transform is done for train for unit2:")#Elapsed time is 240.888650 seconds.
timerObj.toc()

timerObj.tic()
module1Out3_train = pixelHop2TransBatchProcessor_Unit3_V2(subXTrain4,loaded_pixelHop2_6000)
print("transform is done for train for unit3:")#Elapsed time is 202.611487 seconds.
timerObj.toc()


####Test set #Run only once. 
#timerObj.tic()
#module1OutStorage_test = pixelHop2TransBatchProcessor(pre_X_test,loaded_pixelHop2_6000) #Elapsed time is 298.628781 seconds.
#timerObj.toc()

file = open('TestVariables.pckl', 'rb')
pre_X_test, Y_test, module1OutStorage_test = pickle.load(file)
file.close()

#Test
module1Out1_test = module1OutStorage_test[0]
module1Out2_test = module1OutStorage_test[1]
module1Out3_test = module1OutStorage_test[2]


##1st unit(depth 1)
##Module 2
#Max-pooling
pooled_module1Out1_train = MaxpoolingOperator(module1Out1_train)
pooled_module1Out1_test = MaxpoolingOperator(module1Out1_test)

#Feature Selection
timerObj.tic()
selectedFeaturesIndeces_unit1_train , reducedFeatureImgs_unit1_train, numOfNs_unit1_train = featureSelector(pooled_module1Out1_train,subYTrain4)
reducedFeatureImgs_unit1_test = featureSelectorWithoutCrossEnt(pooled_module1Out1_test, selectedFeaturesIndeces_unit1_train, numOfNs_unit1_train)
print("feature selection is done for train and test::")#Elapsed time is 1094.496263 seconds.
timerObj.toc()

#LAG Unit
lag_unit1 = LAG(encode='distance', num_clusters=[5,5,5,5,5,5,5,5,5,5], alpha=10, learner=myLLSR(onehot=False))
lag_unit1.fit(reducedFeatureImgs_unit1_train,subYTrain4)
timerObj.tic()
finalFeature_unit1_train = lag_unit1.transform(reducedFeatureImgs_unit1_train)
print("LAG is done for train set:")#Elapsed time is 0.296036 seconds.
timerObj.toc()

timerObj.tic()
finalFeature_unit1_test = lag_unit1.transform(reducedFeatureImgs_unit1_test)
print("LAG is done for test set:")#Elapsed time is 0.244330 seconds.
timerObj.toc()


##2nd unit(depth 2)
##Module 2
#Max-pooling
pooled_module1Out2_train = MaxpoolingOperator(module1Out2_train)
pooled_module1Out2_test = MaxpoolingOperator(module1Out2_test)

#Feature Selection
timerObj.tic()
selectedFeaturesIndeces_unit2_train , reducedFeatureImgs_unit2_train, numOfNs_unit2_train = featureSelector(pooled_module1Out2_train,subYTrain4)
reducedFeatureImgs_unit2_test = featureSelectorWithoutCrossEnt(pooled_module1Out2_test, selectedFeaturesIndeces_unit2_train, numOfNs_unit2_train)
print("feature selection is done for train and test:") #Elapsed time is 955.182213 seconds
timerObj.toc()

#LAG Unit
lag_unit2 = LAG(encode='distance', num_clusters=[5,5,5,5,5,5,5,5,5,5], alpha=10, learner=myLLSR(onehot=False))
lag_unit2.fit(reducedFeatureImgs_unit2_train,subYTrain4)
timerObj.tic()
finalFeature_unit2_train = lag_unit2.transform(reducedFeatureImgs_unit2_train)
print("LAG is done for train set:")#Elapsed time is 0.265465 seconds.
timerObj.toc()

timerObj.tic()
finalFeature_unit2_test = lag_unit2.transform(reducedFeatureImgs_unit2_test)
print("LAG is done for test set:")#Elapsed time is 0.222479 seconds.
timerObj.toc()

##3rd unit(depth 3)
##Module 2
#Feature Selection
timerObj.tic()
selectedFeaturesIndeces_unit3_train , reducedFeatureImgs_unit3_train, numOfNs_unit3_train = featureSelector(module1Out3_train,subYTrain4)
reducedFeatureImgs_unit3_test = featureSelectorWithoutCrossEnt(module1Out3_test, selectedFeaturesIndeces_unit3_train, numOfNs_unit3_train)
print("feature selection is done for train and test:") #Elapsed time is 69.942067 seconds.
timerObj.toc()

#LAG Unit
lag_unit3 = LAG(encode='distance', num_clusters=[5,5,5,5,5,5,5,5,5,5], alpha=10, learner=myLLSR(onehot=False))
lag_unit3.fit(reducedFeatureImgs_unit3_train,subYTrain4)
finalFeature_unit3_train = lag_unit3.transform(reducedFeatureImgs_unit3_train)
print("LAG is done for train set:") #Elapsed time is 74.189534 seconds.
timerObj.toc()

timerObj.tic()
finalFeature_unit3_test = lag_unit3.transform(reducedFeatureImgs_unit3_test)
print("LAG is done for test set:") #Elapsed time is 0.019078 seconds
timerObj.toc()


##Module 3
#Concatenate all features from unit 1, 2, 3 to make 3M - Dimension feature vector
finalFeature_train = featureConcatenator(finalFeature_unit1_train,finalFeature_unit2_train,finalFeature_unit3_train)
finalFeature_test = featureConcatenator(finalFeature_unit1_test,finalFeature_unit2_test,finalFeature_unit3_test)

#Train a classifier
timerObj.tic()
Clf = RandomForestClassifier(n_estimators = 100, bootstrap=True, max_features=3)
Clf.fit(finalFeature_train, subYTrain4)
print("Training Classifier is done:")
timerObj.toc()


#Predict
pred_train = Clf.predict(finalFeature_train)
pred_test = Clf.predict(finalFeature_test)
err_train = 1 - accuracy_score(subYTrain4, pred_train)
err_test = 1 - accuracy_score(Y_test,pred_test)
print('err_train:')
print(err_train)
print('err_test:')
print(err_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:





# In[ ]:





# In[11]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





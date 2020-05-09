#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

import pandas as pd
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

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

def randomForestExecuter(X_tr, y_tr, X_te, y_te):
    '''
    Input:
    1.X_tr & y_tr : training data set(i.e., features and labels respectively)  , type: data frame or numpy array
    2.X_te & y_te : test data set(i.e., features and labels respectively) , type: data frame or numpy array
    
    Output:
    1.final_num_estimators : chosen number of estimators  , type: integer
    2.ave_accuracy_with_chosen_num_estimators : average accuracy with the number of the estimators. It is found by cross validation , type: float  
    3.fin_accuracy_on_train_RF : final accuracy on the train data set , type: float
    4.fin_accuracy_on_test_RF : final accuracy on the test data set , type: float
    5.fin_RF_clf : a chosen model after cross validation , type: object
    
    '''
    
    '''
    RF = Random Forest
    '''
    
    
    
    #create possible number of estimators that will be used in cross validation to find best number of estimators
    n_estimators_values_list = []
    for i_RF in range(50,92,6):
        n_estimators_values_list.append(i_RF*10)
    n_estimators_values_arr = np.array(n_estimators_values_list) #range of number of estimators : from 10 to 1000, step size 10 
    
    num_of_fold_RF = 5
    kf_RF = KFold(n_splits = num_of_fold_RF, shuffle = True)
    
    #storage for average accuracy for each number of estimator
    stor_ave_acc_RF = []
    
    #Cross-Validation
    for j_RF in range(0,len(n_estimators_values_arr)):
        temp_n_estimator = n_estimators_values_arr[j_RF]
        
        #create a storage for 5 accuracies for same number of estimators and different train and test set from the whole train set
        temp_acc_list_RF = []
        
        print('Cross Validation Processing : current number of estimators is ', temp_n_estimator )
        
        for k_RF in range(0,5):
            
            for train_index_RF , test_index_RF in kf_RF.split(X_tr):
                X_val_tr_RF , X_val_test_RF = X_tr[train_index_RF] , X_tr[test_index_RF]
                y_var_tr_RF , y_val_test_RF = y_tr[train_index_RF] , y_tr[test_index_RF]
                
                X_val_tr_RF_arr = np.array(X_val_tr_RF)
                X_val_test_RF_arr = np.array(X_val_test_RF)
                
                y_val_tr_RF_arr = np.array(y_var_tr_RF)
                y_val_test_RF_arr = np.array(y_val_test_RF)
                
                #create D_bag with bag size 1 
                X_train_bag  = X_val_tr_RF_arr
                y_train_bag  = y_val_tr_RF_arr
                #Random Forest Process
                temp_RF_clf = RandomForestClassifier(n_estimators = temp_n_estimator, bootstrap=True)
                #training and predict on test validation set
                temp_RF_clf.fit(X_train_bag, np.ravel(y_train_bag,order='C'))
                predicted_label_on_test_val_data_RF = temp_RF_clf.predict(X_val_test_RF_arr)
                
                #evaluate accuracy on test data set
                accuracy_on_test_val_data_RF = accuracy_score(y_val_test_RF_arr, predicted_label_on_test_val_data_RF)
                
                #save accuracy for each number of estimators 5 times total
                temp_acc_list_RF.append(accuracy_on_test_val_data_RF)
                
        #save average of 5 accuracies for each number of estimators
        average_acc_RF = averageCalculator(temp_acc_list_RF)
        stor_ave_acc_RF.append(average_acc_RF)
    
    #find location of number of estimator value
    num_estimator_location = stor_ave_acc_RF.index(max(stor_ave_acc_RF))
    final_num_of_estimators = n_estimators_values_arr[num_estimator_location]
    print('Chosen number of estimators is ' , final_num_of_estimators , '\n' )
    print('Average Acurracy with the chosen number of estimators is ' , max(stor_ave_acc_RF) , '\n')
    
    #Final training
    fin_RF_clf = RandomForestClassifier(n_estimators = final_num_of_estimators, bootstrap = True)
    fin_RF_clf.fit(X_tr,np.ravel(y_tr,order='C'))
    fin_predicted_label_on_train_data_RF = fin_RF_clf.predict(X_tr)
    
    #calculate accuracy on whole train data set
    fin_accuracy_on_train_RF = accuracy_score(y_tr, fin_predicted_label_on_train_data_RF)
    print('Accuracy on whole train data with chosen number of estimator is : ', fin_accuracy_on_train_RF)
    
    #calculate accuracy on whole test data set
    fin_predicted_label_on_test_data_RF = fin_RF_clf.predict(X_te)
    fin_accuracy_on_test_RF = accuracy_score(y_te, fin_predicted_label_on_test_data_RF)
    print('Accuracy on whole test data with chosen number of estimator is : ' , fin_accuracy_on_test_RF)
    
    
    
    return final_num_of_estimators, max(stor_ave_acc_RF) , fin_accuracy_on_train_RF , fin_accuracy_on_test_RF , fin_RF_clf, fin_predicted_label_on_test_data_RF

def averageCalculator(accuracy_list):
    average = sum(accuracy_list) / len(accuracy_list)
    return average

##########################################Main##############################
#Create train and test data set from CIFAR-10 data base
X_train, Y_train, X_test, Y_test = CIFAR10DatasetCreator() #50000 images for training, 10000 images for test

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
trainDataSize = 1/16
testDataSize = 1 - trainDataSize
subXTrain4, _ , subYTrain4 , _ = train_test_split(pre_X_train, Y_train, test_size = testDataSize, random_state = 0)


#train set
#Use only 20000 train image, my computer cannot carry all (it takes 27 mins = Elapsed time is 1655.533442 seconds.)
timerObj.tic()
#Use batch processing for a large number of data set. Otherwise, use loaded_pixelHop2_6000 directly.
#module1OutStorage_train = pixelHop2TransBatchProcessor(subXTrain4,loaded_pixelHop2_6000)
module1OutStorage_train = loaded_pixelHop2_6000.transform(subXTrain4)
timerObj.toc()
#Elapsed time is 56.481034 seconds.


#Test set
#timerObj.tic()
#module1OutStorage_test = pixelHop2TransBatchProcessor(pre_X_test,loaded_pixelHop2_6000) #Elapsed time is 298.628781 seconds.
#timerObj.toc()

file = open('TestVariables.pckl', 'rb')
pre_X_test, Y_test, module1OutStorage_test = pickle.load(file)
file.close()

#Seperate extracted features from each hop
module1Out1_train = module1OutStorage_train[0]
module1Out2_train = module1OutStorage_train[1]
module1Out3_train = module1OutStorage_train[2]
#module1OutStorage_train = 'guru99' #free up variable

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
print("feature selection is done for train and test::")#Elapsed time is 782.632993 seconds.
timerObj.toc()

#LAG Unit
lag_unit1 = LAG(encode='distance', num_clusters=[5,5,5,5,5,5,5,5,5,5], alpha=10, learner=myLLSR(onehot=False))
lag_unit1.fit(reducedFeatureImgs_unit1_train,subYTrain4)
timerObj.tic()
finalFeature_unit1_train = lag_unit1.transform(reducedFeatureImgs_unit1_train)
print("LAG is done for train set:")#Elapsed time is 0.099313 seconds.
timerObj.toc()

timerObj.tic()
finalFeature_unit1_test = lag_unit1.transform(reducedFeatureImgs_unit1_test)
print("LAG is done for test set:")#Elapsed time is 0.296141 seconds.
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
print("feature selection is done for train and test:") #Elapsed time is 724.441072 seconds.
timerObj.toc()

#LAG Unit
lag_unit2 = LAG(encode='distance', num_clusters=[5,5,5,5,5,5,5,5,5,5], alpha=10, learner=myLLSR(onehot=False))
lag_unit2.fit(reducedFeatureImgs_unit2_train,subYTrain4)
timerObj.tic()
finalFeature_unit2_train = lag_unit2.transform(reducedFeatureImgs_unit2_train)
print("LAG is done for train set:")#Elapsed time is 0.083610 seconds.
timerObj.toc()

timerObj.tic()
finalFeature_unit2_test = lag_unit2.transform(reducedFeatureImgs_unit2_test)
print("LAG is done for test set:")#Elapsed time is 0.246903 seconds.
timerObj.toc()

##3rd unit(depth 3)
##Module 2
#Feature Selection
timerObj.tic()
selectedFeaturesIndeces_unit3_train , reducedFeatureImgs_unit3_train, numOfNs_unit3_train = featureSelector(module1Out3_train,subYTrain4)
reducedFeatureImgs_unit3_test = featureSelectorWithoutCrossEnt(module1Out3_test, selectedFeaturesIndeces_unit3_train, numOfNs_unit3_train)
print("feature selection is done for train and test:") #Elapsed time is 59.921232 seconds.
timerObj.toc()

#LAG Unit
lag_unit3 = LAG(encode='distance', num_clusters=[5,5,5,5,5,5,5,5,5,5], alpha=10, learner=myLLSR(onehot=False))
lag_unit3.fit(reducedFeatureImgs_unit3_train,subYTrain4)
finalFeature_unit3_train = lag_unit3.transform(reducedFeatureImgs_unit3_train)
print("LAG is done for train set:") #Elapsed time is 63.926254 seconds.
timerObj.toc()

timerObj.tic()
finalFeature_unit3_test = lag_unit3.transform(reducedFeatureImgs_unit3_test)
print("LAG is done for test set:") #Elapsed time is 0.023285 seconds.
timerObj.toc()

print(reducedFeatureImgs_unit1_train.shape)
print(reducedFeatureImgs_unit2_train.shape)
print(reducedFeatureImgs_unit3_train.shape)


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





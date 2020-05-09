#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Basic library for handling data set
import numpy as np

#Basic pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F

#Load data sets(Train data set, Test data set)
import torchvision as tv
import torchvision.transforms as transforms

#library for optimization
import torch.optim as optimization

#library for plotting
import matplotlib.pyplot as plt

from pytictoc import TicToc #timer purpose

#Define archtecture of the Nueral Network
class DeeperNeuralNet(nn.Module):
    def __init__(self):
        super(DeeperNeuralNet,self).__init__()
        #First convolution operation
        #input image has 3 channels due to color image, 10 output channels, convolution with 5x5 square based on the architecture
        numOfChannels_in_conv1 = 3 
        numOfChannels_out_conv1 = 10 #increased number of filter from 6 to 10
        sizeOfFilter_conv1 = 5
        self.conv1 = nn.Conv2d(numOfChannels_in_conv1,numOfChannels_out_conv1,sizeOfFilter_conv1,padding=2) #perform padding
        #Second convolution operation
        #input image has 10 channels, 20 output channels, convolution with 5x5 square filter based on the architecture
        numOfChannels_in_conv2 = 10 #increased number of filter from 6 to 10
        numOfChannels_out_conv2 = 20 #increased number of filter from 16 to 20 
        sizeOfFilter_conv2 = 5
        self.conv2 = nn.Conv2d(numOfChannels_in_conv2,numOfChannels_out_conv2,sizeOfFilter_conv2,padding=2) #perform padding
        #Third convolution operation
        #input has 20 channels, 30 ouput channels, convolution with 5x5 square filter based on the architecture I design
        numOfChannels_in_conv3 = 20
        numOfChannels_out_conv3 = 30
        sizeOfFilter_conv3 = 5
        self.conv3 = nn.Conv2d(numOfChannels_in_conv3,numOfChannels_out_conv3,sizeOfFilter_conv3,padding=2) #perform padding
        
        
        
        #Build fully connected layers structures.
        #1st
        sizeOf2ndMaxpoolingResult = 4 #structure is in competition 1 sketch paper page (7)  
        in_features_fc1 = numOfChannels_out_conv3 * sizeOf2ndMaxpoolingResult * sizeOf2ndMaxpoolingResult  # e.g.4x4x30 = 480
        out_features_fc1 = 130 #Based on the structure I design
        self.fc1 = nn.Linear(in_features_fc1, out_features_fc1)
        
        #2nd
        in_features_fc2 = out_features_fc1
        out_features_fc2 = 94 #Based on the structure I design
        self.fc2 = nn.Linear(in_features_fc2,out_features_fc2)
        
        #3rd(a.k.a output layer)
        in_features_fc3 = out_features_fc2
        out_features_fc3 = 10 #Based on the given structure in HW5
        self.fc3 = nn.Linear(in_features_fc3,out_features_fc3)
        
    def feedForward(self, x_input):
        #Perform Max pooling and apply non linear activation function
        sizeOfMaxpoolingWindow = 2
        #Max pooling after the first convolution, elu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.elu(self.conv1(x_input), alpha=1.0), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Max pooling after the second convolution, elu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.elu(self.conv2(x_input), alpha=1.0), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Max pooling after the third convolution, elu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.elu(self.conv3(x_input), alpha=1.0), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        
        #Flatten 4D tensor to 1D vector to use it at FC1 layer(Reshaping process)
        x_input = x_input.view(-1, self.numOfFlattenFeaturesCalculator(x_input))
        
        #Apply non  activation fuctions to the Fully connected layers
        #1st fc
        x_input = F.elu(self.fc1(x_input),alpha=1.0)
        
        #2nd fc
        x_input = F.elu(self.fc2(x_input),alpha=1.0)
        
        #output layer without non linear activation function
        x_input = self.fc3(x_input)
        return x_input
    
    def numOfFlattenFeaturesCalculator(self, x_input):
        sizeOfx_input = x_input.size()[1:]
        numOfFeatures = 1
        for sizeOrder in sizeOfx_input:
            numOfFeatures *= sizeOrder
        return numOfFeatures
    
def dataLoader():
    '''
    dataLoader creates train & test data set loader for CIFAR-10 data base
    
    Args:
        None
    
    Returns:
        trainDataSetLoader: generated train data set loader from CIFAR-10 data base
        testDataSetLoader: generated test data set loader from CIFAR-10 data base
    '''
    
    #each channel's mean and std in CIFAR-10 train data set
    #These values are given on "https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"
    meanChannelOne = 0.5
    meanChannelTwo = 0.5
    meanChannelThree = 0.5
    
    stdChannelOne = 0.5
    stdChannelTwo = 0.5
    stdChannelThree = 0.5
    
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((meanChannelOne,meanChannelTwo,meanChannelThree),(stdChannelOne,stdChannelTwo,stdChannelThree))])
    
    #set train data set
    trainDataSet = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    #Load train data
    batchSizeOfTrainData = 64 #it needs for SGD process
    trainDataSetLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=batchSizeOfTrainData, shuffle=True,num_workers=2)

    #set test data set
    testDataSet = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    #load test data
    batchSizeOfTestData = 1000 #whole data set is to be a batch for test
    testDataSetLoader = torch.utils.data.DataLoader(testDataSet, batch_size=batchSizeOfTestData, shuffle=False, num_workers=2)
    
    return trainDataSetLoader, testDataSetLoader

def accuracyCalculator(dataSetLoader,crrNeuralNet):
    '''
    Calculate and return accuracy on the given data set using current trained Neural Net
    
    Args: 
        dataSetLoader: given data set loader
        crrNeuralNet: given current trained neural net. 
        
    Returns:
        accuracyForCurrentNet: current accuracy on the given data set by trained net at current epoch
        
    '''
    
    numOfCorrectPrediction = 0
    numOfTotalDataPoints = 0
    timerObj = TicToc()
    timerObj.tic()
    with torch.no_grad():
        for crrData in dataSetLoader:
            crrInputData, crrLabels = crrData
            predictedOutputs = crrNeuralNet.feedForward(crrInputData)
            _, predictedLabels = torch.max(predictedOutputs.data, 1)
            numOfTotalDataPoints += crrLabels.size(0)
            numOfCorrectPrediction += (predictedLabels==crrLabels).sum().item()
    
    accuracyForCurrentNet = numOfCorrectPrediction / numOfTotalDataPoints
    print('Accuracy of the network on the images(train or test): %d %%' % (
    100 * numOfCorrectPrediction / numOfTotalDataPoints))
    timerObj.toc()
    return accuracyForCurrentNet     

def trainProcessor(trainDataSetLoader, chosenOptimizer, NeuralNet, chosenLossFunc, testDataSetLoader, orderOfEpoch):
    '''
    Calculate and return accuracy on train data set and test data set 
    while training the given Neural Net with chosen optimizer and loss function
    
    Args: 
        trainDataSetLoader: train data set loader
        chosenOptimizer: chosen optimizer. It will be used for training the net. e.g. SGD 
        NeuralNet: given neural net. e.g. LeNet5
        chosenLossFunc: chosen loss function(a.k.a criterion function). optimizer will try to lower value of loss function. e.g. cross entropy, MSE error
        testDataSetLoader: test data set loader
        orderOfEpoch: current order of epoch. It can be used to display value of loss function at current epoch. 
        
    Returns:
        currentAcc_train: current accuracy on train data set by trained net at current epoch
        currentAcc_test: current accuracy on test data set by trained net at current epoch
    '''
    #Train the given Neural net 
    #initialize current loss 
    currentLoss = 0.0
    
    for crrIteration, currentData in enumerate(trainDataSetLoader, 0):
        
        #get small input and labels of current batch
        crrBatch, labelsOfcrrBatch = currentData
        
        #need to make parameter gradients zero
        chosenOptimizer.zero_grad()
        
        #Perform Feed Forward to get output
        crrOutputs = NeuralNet.feedForward(crrBatch)
        
        #Calculate current loss using the chosen loss function and current outputs
        crrLoss = chosenLossFunc(crrOutputs, labelsOfcrrBatch)
        
        #Perform backpropagation and update optimizer's parameter
        crrLoss.backward()
        chosenOptimizer.step()
        
        #Debugging purpose to see value of loss function decreases while number of iteration and order of epoch increase
        currentLoss += crrLoss.item()
        unitIteration = 100
        if crrIteration % unitIteration == (unitIteration-1):
            print('[%d, %5d] current loss: %.3f' %
                  (orderOfEpoch + 1, crrIteration + 1, currentLoss / unitIteration))
            currentLoss = 0
        
        #Calculate accuracy on train and test data set
        
        
        
    #dummy accuracies
    crrTrainingAcc = accuracyCalculator(trainDataSetLoader,NeuralNet)
    crrTestACC = accuracyCalculator(testDataSetLoader,NeuralNet)
    return crrTrainingAcc, crrTestACC
def wholeTrainProcessor(trainDataSetLoader,momentumMag,eta,weightDecay,chosenNeuralNet,testDataSetLoader,numOfEpoch):
    '''
    Calculate and return accuracyStorage on train data set and test data set based on the given number of Epoch 
    while training the given Neural Net with chosen optimizer and loss function
    
    Args: 
        trainDataSetLoader: train data set loader
        modmentumMag: given momentum magnitude
        eta: learning rate
        weightDecay: regurization factor
        chosenNeuralNet: given neural net. It might have constructed by different intialization method e.g. LeNet5 with random initialized filter weights
        testDataSetLoader: test data set loader
        numOfEpoch: number of epoch
        
    Returns:
        accStorage_train: accuracies storage for each epoch for train data set
        accStorage_test: accuracies storage for each epoch for test data set
    
        
        '''
    
    
    #Construct optimizer
    chosenOptimizer = optimization.SGD(chosenNeuralNet.parameters(),lr=eta, momentum= momentumMag, weight_decay = weightDecay)
    
    #choose loss function
    chosenLossFunc = nn.CrossEntropyLoss()
    
    #create accuracy storages for train and test data set
    accStorage_train = np.zeros((1,numOfEpoch)) #Accuracies storage for each epoch
    accStorage_test = np.zeros((1,numOfEpoch))
    
    for orderOfEpoch in range(numOfEpoch):
        tempAcc_train, tempACC_test = trainProcessor(trainDataSetLoader, chosenOptimizer, chosenNeuralNet, chosenLossFunc, testDataSetLoader, orderOfEpoch)
    
        #Save accuracy on train and test data set for each epoch
        accStorage_train[0,orderOfEpoch] = tempAcc_train
        accStorage_test[0,orderOfEpoch] = tempACC_test

    return accStorage_train, accStorage_test

def dataLoaderV1(numOfTrainingSamples):
    '''
    dataLoader creates train & test data set loader for CIFAR-10 data base based on the given number of training Samples 
    e.g. 1562, 3125, 6250, 12500, 20000, 50000 
    
    Args:
        numOfTrainingSamples: desired number of training samples from CIFAR-10 data base
    
    Returns:
        trainDataSetLoader: generated train data set loader from CIFAR-10 data base
        testDataSetLoader: generated test data set loader from CIFAR-10 data base
    '''
    
    #each channel's mean and std in CIFAR-10 train data set
    #These values are given on "https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"
    meanChannelOne = 0.5
    meanChannelTwo = 0.5
    meanChannelThree = 0.5
    
    stdChannelOne = 0.5
    stdChannelTwo = 0.5
    stdChannelThree = 0.5
    
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((meanChannelOne,meanChannelTwo,meanChannelThree),(stdChannelOne,stdChannelTwo,stdChannelThree))])
    
    #set train data set
    trainDataSet = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    #Draw part of training samples at random
    portionOfTrainDataSet = torch.utils.data.random_split(trainDataSet, [numOfTrainingSamples, len(trainDataSet)-numOfTrainingSamples])[0]
    
    #Check if dataset is correctly slited 
    print(np.asarray(portionOfTrainDataSet.indices).shape)
   
    
    #Load train data
    batchSizeOfTrainData = 64 #it needs for SGD process
    trainDataSetLoader = torch.utils.data.DataLoader(portionOfTrainDataSet, batch_size=batchSizeOfTrainData, shuffle=True,num_workers=2)

    #set test data set
    testDataSet = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    #load test data
    batchSizeOfTestData = 1000 #whole data set is to be a batch for test
    testDataSetLoader = torch.utils.data.DataLoader(testDataSet, batch_size=batchSizeOfTestData, shuffle=False, num_workers=2)
    
    return trainDataSetLoader, testDataSetLoader

#Load CIFAR-10 data sets
trainDataSetLoader, testDataSetLoader = dataLoader()


bestLearningRate = 0.02 #from HW5 pr1
bestDecay = 0.01 #from HW5 pr1
bestMomentum = 0.7 #from HW5 pr1

DeeprNeuralNet = DeeperNeuralNet()
numOfEpochFinal = 80

#Takes 3 hours
timerObj = TicToc() #create timer class
timerObj.tic()
final_accStorage_train, final_accStorage_test = wholeTrainProcessor(trainDataSetLoader,bestMomentum,bestLearningRate,bestDecay,DeeprNeuralNet,testDataSetLoader,numOfEpochFinal)
timerObj.toc()


#plotting final learning curve
epochArr_final = []
final_accStorage_train_list = []
final_accStorage_test_list = []

for k in range(numOfEpochFinal):
    epochArr_final.append(k+1)
    #need to convert data type to list for plotting. 
    final_accStorage_train_list.append(final_accStorage_train[0,k])
    final_accStorage_test_list.append(final_accStorage_test[0,k])
    
plt.plot(epochArr_final, final_accStorage_train_list, label="Accuracy on train set")
plt.plot(epochArr_final, final_accStorage_test_list, label="Accuracy on test set")
plt.xlabel('Number of epoches')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

#Dropping a number of traning samples
numOfTrainingSamples20000 = 20000
numOfTrainingSamples12500 = 12500
numOfTrainingSamples6250 = 6250
numOfTrainingSamples3125 = 3125
numOfTrainingSamples1562 = 1562

trainDataSetLoader20000, testDataSetLoader20000 = dataLoaderV1(numOfTrainingSamples20000)
trainDataSetLoader12500, testDataSetLoader12500 = dataLoaderV1(numOfTrainingSamples12500)
trainDataSetLoader6250, testDataSetLoader6250 = dataLoaderV1(numOfTrainingSamples6250)
trainDataSetLoader3125, testDataSetLoader3125 = dataLoaderV1(numOfTrainingSamples3125)
trainDataSetLoader1562, testDataSetLoader1562 = dataLoaderV1(numOfTrainingSamples1562)


timerObj = TicToc() #create timer class

#20000
DeeprNeuralNet20000 = DeeperNeuralNet()
timerObj.tic()
final_accStorage_train20000, final_accStorage_test20000 = wholeTrainProcessor(trainDataSetLoader20000,bestMomentum,bestLearningRate,bestDecay,DeeprNeuralNet20000,testDataSetLoader20000,numOfEpochFinal)
timerObj.toc()
#Elapsed time is 5246.893781 seconds.

#12500
DeeprNeuralNet12500 = DeeperNeuralNet()
timerObj.tic()
final_accStorage_train12500, final_accStorage_test12500 = wholeTrainProcessor(trainDataSetLoader12500,bestMomentum,bestLearningRate,bestDecay,DeeprNeuralNet12500,testDataSetLoader12500,numOfEpochFinal)
timerObj.toc()
#Elapsed time is 3753.619140 seconds.

#6250
DeeprNeuralNet6250 = DeeperNeuralNet()
timerObj.tic()
final_accStorage_train6250, final_accStorage_test6250 = wholeTrainProcessor(trainDataSetLoader6250,bestMomentum,bestLearningRate,bestDecay,DeeprNeuralNet6250,testDataSetLoader6250,numOfEpochFinal)
timerObj.toc()
#Elapsed time is 2917.334948 seconds.

#3125
DeeprNeuralNet3125 = DeeperNeuralNet()
timerObj.tic()
final_accStorage_train3125, final_accStorage_test3125 = wholeTrainProcessor(trainDataSetLoader3125,bestMomentum,bestLearningRate,bestDecay,DeeprNeuralNet3125,testDataSetLoader3125,numOfEpochFinal)
timerObj.toc()
#Elapsed time is 2320.642540 seconds.

#1562
DeeprNeuralNet1562 = DeeperNeuralNet()
timerObj.tic()
final_accStorage_train1562, final_accStorage_test1562 = wholeTrainProcessor(trainDataSetLoader1562,bestMomentum,bestLearningRate,bestDecay,DeeprNeuralNet1562,testDataSetLoader1562,numOfEpochFinal)
timerObj.toc()
#Elapsed time is 1756.230697 seconds.

#Plotting accuracy for each number of training samples
xVals = [1562,3125,6250,12500,20000, 50000]
accuracies = [final_accStorage_test1562[0,79], final_accStorage_test3125[0,79],final_accStorage_test6250[0,79] ,final_accStorage_test12500[0,79] ,final_accStorage_test20000[0,79] ,final_accStorage_test[0,79]]
plt.plot(xVals,accuracies)
plt.xlabel('number of training samples for training')
plt.ylabel('accuracy')
plt.title('accuracy for each number of training samples')
plt.grid()
plt.show()


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





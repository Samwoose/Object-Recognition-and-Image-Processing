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

#Define archtecture of the Nueral Network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        #First convolution operation
        #input image has 3 channels due to color image, 6 output channels, convolution with 5x5 square based on the architecture
        numOfChannels_in_conv1 = 3 
        numOfChannels_out_conv1 = 6
        sizeOfFilter_conv1 = 5
        self.conv1 = nn.Conv2d(numOfChannels_in_conv1,numOfChannels_out_conv1,sizeOfFilter_conv1)
        #Second convolution operation
        #input image has 6 channels, 16 output channels, convolution with 5x5 square filter based on the architecture
        numOfChannels_in_conv2 = 6
        numOfChannels_out_conv2 = 16
        sizeOfFilter_conv2 = 5
        self.conv2 = nn.Conv2d(numOfChannels_in_conv2,numOfChannels_out_conv2,sizeOfFilter_conv2)
        
        #Build fully connected layers structures.
        #1st
        sizeOf2ndMaxpoolingResult = 5 
        in_features_fc1 = numOfChannels_out_conv2 * sizeOf2ndMaxpoolingResult * sizeOf2ndMaxpoolingResult  # e.g.16x5x5 = 400
        out_features_fc1 = 120 #Based on the given structure in HW5
        self.fc1 = nn.Linear(in_features_fc1, out_features_fc1)
        
        #2nd
        in_features_fc2 = out_features_fc1
        out_features_fc2 = 84 #Based on the given structure in HW5
        self.fc2 = nn.Linear(in_features_fc2,out_features_fc2)
        
        #3rd(a.k.a output layer)
        in_features_fc3 = out_features_fc2
        out_features_fc3 = 10 #Based on the given structure in HW5
        self.fc3 = nn.Linear(in_features_fc3,out_features_fc3)
        
    def feedForward(self, x_input):
        #Perform Max pooling and apply non linear activation function
        sizeOfMaxpoolingWindow = 2
        #Max pooling after the first convolution, relu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.relu(self.conv1(x_input)), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Max pooling after the second convolution, relu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.relu(self.conv2(x_input)), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Flatten 4D tensor to 1D vector to use it at FC1 layer(Reshaping process)
        x_input = x_input.view(-1, self.numOfFlattenFeaturesCalculator(x_input))
        
        #Apply non  activation fuctions to the Fully connected layers
        #1st fc
        x_input = F.relu(self.fc1(x_input))
        
        #2nd fc
        x_input = F.relu(self.fc2(x_input))
        
        #output layer without non linear activation function
        x_input = self.fc3(x_input)
        return x_input
    
    def numOfFlattenFeaturesCalculator(self, x_input):
        sizeOfx_input = x_input.size()[1:]
        numOfFeatures = 1
        for sizeOrder in sizeOfx_input:
            numOfFeatures *= sizeOrder
        return numOfFeatures
 
#Define archtecture of the Nueral Network
class NeuralNet_normal(nn.Module):
    def __init__(self):
        super(NeuralNet_normal,self).__init__()
        #First convolution operation
        #input image has 3 channels due to color image, 6 output channels, convolution with 5x5 square based on the architecture
        numOfChannels_in_conv1 = 3 
        numOfChannels_out_conv1 = 6
        sizeOfFilter_conv1 = 5
        self.conv1 = nn.Conv2d(numOfChannels_in_conv1,numOfChannels_out_conv1,sizeOfFilter_conv1)
        #Second convolution operation
        #input image has 6 channels, 16 output channels, convolution with 5x5 square filter based on the architecture
        numOfChannels_in_conv2 = 6
        numOfChannels_out_conv2 = 16
        sizeOfFilter_conv2 = 5
        self.conv2 = nn.Conv2d(numOfChannels_in_conv2,numOfChannels_out_conv2,sizeOfFilter_conv2)
        
        #Build fully connected layers structures.
        #1st
        sizeOf2ndMaxpoolingResult = 5 
        in_features_fc1 = numOfChannels_out_conv2 * sizeOf2ndMaxpoolingResult * sizeOf2ndMaxpoolingResult  # e.g.16x5x5 = 400
        out_features_fc1 = 120 #Based on the given structure in HW5
        self.fc1 = nn.Linear(in_features_fc1, out_features_fc1)
        #initialize weight with normal distribution at fc1
        nn.init.normal_(self.fc1.weight, mean=0, std=1)
        
        #2nd
        in_features_fc2 = out_features_fc1
        out_features_fc2 = 84 #Based on the given structure in HW5
        self.fc2 = nn.Linear(in_features_fc2,out_features_fc2)
        #initialize weight with normal distribution at fc1
        nn.init.normal_(self.fc2.weight, mean=0, std=1)
        
        #3rd(a.k.a output layer)
        in_features_fc3 = out_features_fc2
        out_features_fc3 = 10 #Based on the given structure in HW5
        self.fc3 = nn.Linear(in_features_fc3,out_features_fc3)
        
    def feedForward(self, x_input):
        #Perform Max pooling and apply non linear activation function
        sizeOfMaxpoolingWindow = 2
        #Max pooling after the first convolution, relu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.relu(self.conv1(x_input)), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Max pooling after the second convolution, relu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.relu(self.conv2(x_input)), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Flatten 4D tensor to 1D vector to use it at FC1 layer(Reshaping process)
        x_input = x_input.view(-1, self.numOfFlattenFeaturesCalculator(x_input))
        
        #Apply non  activation fuctions to the Fully connected layers
        #1st fc
        x_input = F.relu(self.fc1(x_input))
        
        #2nd fc
        x_input = F.relu(self.fc2(x_input))
        
        #output layer without non linear activation function
        x_input = self.fc3(x_input)
        return x_input
    
    def numOfFlattenFeaturesCalculator(self, x_input):
        sizeOfx_input = x_input.size()[1:]
        numOfFeatures = 1
        for sizeOrder in sizeOfx_input:
            numOfFeatures *= sizeOrder
        return numOfFeatures
 
#Define archtecture of the Nueral Network
class NeuralNet_uniform(nn.Module):
    def __init__(self):
        super(NeuralNet_uniform,self).__init__()
        #First convolution operation
        #input image has 3 channels due to color image, 6 output channels, convolution with 5x5 square based on the architecture
        numOfChannels_in_conv1 = 3 
        numOfChannels_out_conv1 = 6
        sizeOfFilter_conv1 = 5
        self.conv1 = nn.Conv2d(numOfChannels_in_conv1,numOfChannels_out_conv1,sizeOfFilter_conv1)
        #Second convolution operation
        #input image has 6 channels, 16 output channels, convolution with 5x5 square filter based on the architecture
        numOfChannels_in_conv2 = 6
        numOfChannels_out_conv2 = 16
        sizeOfFilter_conv2 = 5
        self.conv2 = nn.Conv2d(numOfChannels_in_conv2,numOfChannels_out_conv2,sizeOfFilter_conv2)
        
        #Build fully connected layers structures.
        #1st
        sizeOf2ndMaxpoolingResult = 5 
        in_features_fc1 = numOfChannels_out_conv2 * sizeOf2ndMaxpoolingResult * sizeOf2ndMaxpoolingResult  # e.g.16x5x5 = 400
        out_features_fc1 = 120 #Based on the given structure in HW5
        self.fc1 = nn.Linear(in_features_fc1, out_features_fc1)
        #initialize weight with uniform distribution at fc1
        nn.init.uniform_(self.fc1.weight)
        
        #2nd
        in_features_fc2 = out_features_fc1
        out_features_fc2 = 84 #Based on the given structure in HW5
        self.fc2 = nn.Linear(in_features_fc2,out_features_fc2)
        #initialize weight with uniform distribution at fc2
        nn.init.uniform_(self.fc2.weight)
        
        #3rd(a.k.a output layer)
        in_features_fc3 = out_features_fc2
        out_features_fc3 = 10 #Based on the given structure in HW5
        self.fc3 = nn.Linear(in_features_fc3,out_features_fc3)
        
    def feedForward(self, x_input):
        #Perform Max pooling and apply non linear activation function
        sizeOfMaxpoolingWindow = 2
        #Max pooling after the first convolution, relu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.relu(self.conv1(x_input)), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Max pooling after the second convolution, relu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.relu(self.conv2(x_input)), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Flatten 4D tensor to 1D vector to use it at FC1 layer(Reshaping process)
        x_input = x_input.view(-1, self.numOfFlattenFeaturesCalculator(x_input))
        
        #Apply non  activation fuctions to the Fully connected layers
        #1st fc
        x_input = F.relu(self.fc1(x_input))
        
        #2nd fc
        x_input = F.relu(self.fc2(x_input))
        
        #output layer without non linear activation function
        x_input = self.fc3(x_input)
        return x_input
    
    def numOfFlattenFeaturesCalculator(self, x_input):
        sizeOfx_input = x_input.size()[1:]
        numOfFeatures = 1
        for sizeOrder in sizeOfx_input:
            numOfFeatures *= sizeOrder
        return numOfFeatures
 
#Define archtecture of the Nueral Network
class NeuralNet_constant(nn.Module):
    def __init__(self):
        super(NeuralNet_constant,self).__init__()
        #First convolution operation
        #input image has 3 channels due to color image, 6 output channels, convolution with 5x5 square based on the architecture
        numOfChannels_in_conv1 = 3 
        numOfChannels_out_conv1 = 6
        sizeOfFilter_conv1 = 5
        self.conv1 = nn.Conv2d(numOfChannels_in_conv1,numOfChannels_out_conv1,sizeOfFilter_conv1)
        #Second convolution operation
        #input image has 6 channels, 16 output channels, convolution with 5x5 square filter based on the architecture
        numOfChannels_in_conv2 = 6
        numOfChannels_out_conv2 = 16
        sizeOfFilter_conv2 = 5
        self.conv2 = nn.Conv2d(numOfChannels_in_conv2,numOfChannels_out_conv2,sizeOfFilter_conv2)
        
        #Build fully connected layers structures.
        #1st
        sizeOf2ndMaxpoolingResult = 5 
        in_features_fc1 = numOfChannels_out_conv2 * sizeOf2ndMaxpoolingResult * sizeOf2ndMaxpoolingResult  # e.g.16x5x5 = 400
        out_features_fc1 = 120 #Based on the given structure in HW5
        self.fc1 = nn.Linear(in_features_fc1, out_features_fc1)
        #initialize weight with constant 0.2 at fc1
        nn.init.constant_(self.fc1.weight, 0.2)
        
        #2nd
        in_features_fc2 = out_features_fc1
        out_features_fc2 = 84 #Based on the given structure in HW5
        self.fc2 = nn.Linear(in_features_fc2,out_features_fc2)
        #initialize weight with constant 0.2 at fc2
        nn.init.constant_(self.fc2.weight, 0.2)
        
        #3rd(a.k.a output layer)
        in_features_fc3 = out_features_fc2
        out_features_fc3 = 10 #Based on the given structure in HW5
        self.fc3 = nn.Linear(in_features_fc3,out_features_fc3)
        
    def feedForward(self, x_input):
        #Perform Max pooling and apply non linear activation function
        sizeOfMaxpoolingWindow = 2
        #Max pooling after the first convolution, relu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.relu(self.conv1(x_input)), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Max pooling after the second convolution, relu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.relu(self.conv2(x_input)), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Flatten 4D tensor to 1D vector to use it at FC1 layer(Reshaping process)
        x_input = x_input.view(-1, self.numOfFlattenFeaturesCalculator(x_input))
        
        #Apply non  activation fuctions to the Fully connected layers
        #1st fc
        x_input = F.relu(self.fc1(x_input))
        
        #2nd fc
        x_input = F.relu(self.fc2(x_input))
        
        #output layer without non linear activation function
        x_input = self.fc3(x_input)
        return x_input
    
    def numOfFlattenFeaturesCalculator(self, x_input):
        sizeOfx_input = x_input.size()[1:]
        numOfFeatures = 1
        for sizeOrder in sizeOfx_input:
            numOfFeatures *= sizeOrder
        return numOfFeatures
 
#Define archtecture of the Nueral Network
class NeuralNet_ones(nn.Module):
    def __init__(self):
        super(NeuralNet_ones,self).__init__()
        #First convolution operation
        #input image has 3 channels due to color image, 6 output channels, convolution with 5x5 square based on the architecture
        numOfChannels_in_conv1 = 3 
        numOfChannels_out_conv1 = 6
        sizeOfFilter_conv1 = 5
        self.conv1 = nn.Conv2d(numOfChannels_in_conv1,numOfChannels_out_conv1,sizeOfFilter_conv1)
        #Second convolution operation
        #input image has 6 channels, 16 output channels, convolution with 5x5 square filter based on the architecture
        numOfChannels_in_conv2 = 6
        numOfChannels_out_conv2 = 16
        sizeOfFilter_conv2 = 5
        self.conv2 = nn.Conv2d(numOfChannels_in_conv2,numOfChannels_out_conv2,sizeOfFilter_conv2)
        
        #Build fully connected layers structures.
        #1st
        sizeOf2ndMaxpoolingResult = 5 
        in_features_fc1 = numOfChannels_out_conv2 * sizeOf2ndMaxpoolingResult * sizeOf2ndMaxpoolingResult  # e.g.16x5x5 = 400
        out_features_fc1 = 120 #Based on the given structure in HW5
        self.fc1 = nn.Linear(in_features_fc1, out_features_fc1)
        #initialize weight with ones at fc1
        nn.init.ones_(self.fc1.weight)
        
        #2nd
        in_features_fc2 = out_features_fc1
        out_features_fc2 = 84 #Based on the given structure in HW5
        self.fc2 = nn.Linear(in_features_fc2,out_features_fc2)
        #initialize weight with ones at fc2
        nn.init.ones_(self.fc2.weight)
        
        #3rd(a.k.a output layer)
        in_features_fc3 = out_features_fc2
        out_features_fc3 = 10 #Based on the given structure in HW5
        self.fc3 = nn.Linear(in_features_fc3,out_features_fc3)
        
    def feedForward(self, x_input):
        #Perform Max pooling and apply non linear activation function
        sizeOfMaxpoolingWindow = 2
        #Max pooling after the first convolution, relu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.relu(self.conv1(x_input)), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Max pooling after the second convolution, relu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.relu(self.conv2(x_input)), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Flatten 4D tensor to 1D vector to use it at FC1 layer(Reshaping process)
        x_input = x_input.view(-1, self.numOfFlattenFeaturesCalculator(x_input))
        
        #Apply non  activation fuctions to the Fully connected layers
        #1st fc
        x_input = F.relu(self.fc1(x_input))
        
        #2nd fc
        x_input = F.relu(self.fc2(x_input))
        
        #output layer without non linear activation function
        x_input = self.fc3(x_input)
        return x_input
    
    def numOfFlattenFeaturesCalculator(self, x_input):
        sizeOfx_input = x_input.size()[1:]
        numOfFeatures = 1
        for sizeOrder in sizeOfx_input:
            numOfFeatures *= sizeOrder
        return numOfFeatures
#Define archtecture of the Nueral Network
class NeuralNet_zeros(nn.Module):
    def __init__(self):
        super(NeuralNet_zeros,self).__init__()
        #First convolution operation
        #input image has 3 channels due to color image, 6 output channels, convolution with 5x5 square based on the architecture
        numOfChannels_in_conv1 = 3 
        numOfChannels_out_conv1 = 6
        sizeOfFilter_conv1 = 5
        self.conv1 = nn.Conv2d(numOfChannels_in_conv1,numOfChannels_out_conv1,sizeOfFilter_conv1)
        #Second convolution operation
        #input image has 6 channels, 16 output channels, convolution with 5x5 square filter based on the architecture
        numOfChannels_in_conv2 = 6
        numOfChannels_out_conv2 = 16
        sizeOfFilter_conv2 = 5
        self.conv2 = nn.Conv2d(numOfChannels_in_conv2,numOfChannels_out_conv2,sizeOfFilter_conv2)
        
        #Build fully connected layers structures.
        #1st
        sizeOf2ndMaxpoolingResult = 5 
        in_features_fc1 = numOfChannels_out_conv2 * sizeOf2ndMaxpoolingResult * sizeOf2ndMaxpoolingResult  # e.g.16x5x5 = 400
        out_features_fc1 = 120 #Based on the given structure in HW5
        self.fc1 = nn.Linear(in_features_fc1, out_features_fc1)
        #initialize weight with zeros at fc1
        nn.init.zeros_(self.fc1.weight)
        
        #2nd
        in_features_fc2 = out_features_fc1
        out_features_fc2 = 84 #Based on the given structure in HW5
        self.fc2 = nn.Linear(in_features_fc2,out_features_fc2)
        #initialize weight with zeros at fc2
        nn.init.zeros_(self.fc2.weight)
        
        #3rd(a.k.a output layer)
        in_features_fc3 = out_features_fc2
        out_features_fc3 = 10 #Based on the given structure in HW5
        self.fc3 = nn.Linear(in_features_fc3,out_features_fc3)
        
    def feedForward(self, x_input):
        #Perform Max pooling and apply non linear activation function
        sizeOfMaxpoolingWindow = 2
        #Max pooling after the first convolution, relu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.relu(self.conv1(x_input)), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Max pooling after the second convolution, relu is a type of non linear activation fuction
        x_input = F.max_pool2d(F.relu(self.conv2(x_input)), (sizeOfMaxpoolingWindow,sizeOfMaxpoolingWindow))
        
        #Flatten 4D tensor to 1D vector to use it at FC1 layer(Reshaping process)
        x_input = x_input.view(-1, self.numOfFlattenFeaturesCalculator(x_input))
        
        #Apply non  activation fuctions to the Fully connected layers
        #1st fc
        x_input = F.relu(self.fc1(x_input))
        
        #2nd fc
        x_input = F.relu(self.fc2(x_input))
        
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



########################Main part###################
#Load CIFAR-10 data sets
trainDataSetLoader, testDataSetLoader = dataLoader()

#Construct NeuralNet
NeuralNet_LeNet5 = NeuralNet()

#Select optimizer and loss function that will be used in Backpropagation
#Optimizer: SGD will be used
eta1 = 0.001 #a.k.a learning rate
eta2 = 0.020 #a.k.a learning rate
eta3 = 0.250 #a.k.a learning rate
eta4 = 0.500 #a.k.a learning rate
eta5 = 1.000 #a.k.a learning rate

momentumMag1 = 0.0
momentumMag2 = 0.3
momentumMag3 = 0.5
momentumMag4 = 0.7
momentumMag5 = 0.9


#values of weight decay(recularizor)
weightDecay1 = 0
weightDecay2 = 0.01
weightDecay3 = 0.03
weightDecay4 = 0.09
weightDecay5 = 0.27


#Training Procedure
numOfEpoch = 20 #number of epoches



#observing weight decay's role part
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_w1, accStorage_test_w1 = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta1,weightDecay1,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_w2, accStorage_test_w2 = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta1,weightDecay2,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_w3, accStorage_test_w3 = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta1,weightDecay3,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_w4, accStorage_test_w4 = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta1,weightDecay4,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_w5, accStorage_test_w5 = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta1,weightDecay5,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)



#observing momentum's role part
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_m1, accStorage_test_m1 = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta1,weightDecay1,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_m2, accStorage_test_m2 = wholeTrainProcessor(trainDataSetLoader,momentumMag2,eta1,weightDecay1,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_m3, accStorage_test_m3 = wholeTrainProcessor(trainDataSetLoader,momentumMag3,eta1,weightDecay1,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_m4, accStorage_test_m4 = wholeTrainProcessor(trainDataSetLoader,momentumMag4,eta1,weightDecay1,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_m5, accStorage_test_m5 = wholeTrainProcessor(trainDataSetLoader,momentumMag5,eta1,weightDecay1,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)


#observing learning rate's role part
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_e1, accStorage_test_e1 = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta1,weightDecay1,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_e2, accStorage_test_e2 = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta2,weightDecay1,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_e3, accStorage_test_e3 = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta3,weightDecay1,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_e4, accStorage_test_e4 = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta4,weightDecay1,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)
NeuralNet_LeNet5 = NeuralNet()
accStorage_train_e5, accStorage_test_e5 = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta5,weightDecay1,NeuralNet_LeNet5,testDataSetLoader,numOfEpoch)


#observing filter weight initialization's role part
#initialize with normal disribution
#Construct NeuralNet
NeuralNet_LeNet5_normal = NeuralNet_normal()

accStorage_train_normal, accStorage_test_normal = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta1,weightDecay1,NeuralNet_LeNet5_normal,testDataSetLoader,numOfEpoch)

#initialize with uniform disribution
#Construct NeuralNet
NeuralNet_LeNet5_uniform = NeuralNet_uniform()

accStorage_train_uniform, accStorage_test_uniform = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta1,weightDecay1,NeuralNet_LeNet5_uniform,testDataSetLoader,numOfEpoch)

#initialize with constant 0.2
#Construct NeuralNet
NeuralNet_LeNet5_constant = NeuralNet_constant()

accStorage_train_constant, accStorage_test_constant = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta1,weightDecay1,NeuralNet_LeNet5_constant,testDataSetLoader,numOfEpoch)

#initialize with ones
#Construct NeuralNet
NeuralNet_LeNet5_ones = NeuralNet_ones()

accStorage_train_ones, accStorage_test_ones = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta1,weightDecay1,NeuralNet_LeNet5_ones,testDataSetLoader,numOfEpoch)

#initialize with zeros
#Construct NeuralNet
NeuralNet_LeNet5_zeros = NeuralNet_zeros()

accStorage_train_zeros, accStorage_test_zeros = wholeTrainProcessor(trainDataSetLoader,momentumMag1,eta1,weightDecay1,NeuralNet_LeNet5_zeros,testDataSetLoader,numOfEpoch)


#Search a pair of parameters that gives the best accuracy. 
numOfEta = 3
numOfWeightDecay = 2
numOfMomentum = 2
numOfAccuracies = numOfEta * numOfWeightDecay * numOfMomentum

etaStorage = np.array([eta1, eta2, eta3])
weightDecayStorage = np.array([weightDecay2,weightDecay3])
momentumStorage = np.array([momentumMag4, momentumMag5])

finalAccStorage_train = np.zeros((1,numOfAccuracies))
finalAccStorage_test = np.zeros((1,numOfAccuracies))
parameterValueStorage = np.zeros((numOfAccuracies,3))

extraIndex = 0
for orderOfEta in range(numOfEta):
    for orderOfDecay in range(numOfWeightDecay):
        for orderOfMomentum in range(numOfMomentum):
            crr_LeNet5 = NeuralNet()
            crr_eta = etaStorage[orderOfEta]
            crr_weightDecay = weightDecayStorage[orderOfDecay]
            crr_momentum = momentumStorage[orderOfMomentum]
            
            crr_accStorage_train, crr_accStorage_test = wholeTrainProcessor(trainDataSetLoader,crr_momentum,crr_eta,crr_weightDecay,crr_LeNet5,testDataSetLoader,numOfEpoch)
            
            crr_final_acc_train = crr_accStorage_train[0][numOfEpoch-1]
            crr_final_acc_test = crr_accStorage_test[0][numOfEpoch-1]
            
            #Save final accuracy computed at the last epoch
            finalAccStorage_train[0][extraIndex] = crr_final_acc_train
            finalAccStorage_test[0][extraIndex] = crr_final_acc_test
            
            #save value pair
            parameterValueStorage[extraIndex][0] = crr_eta 
            parameterValueStorage[extraIndex][1]=crr_weightDecay 
            parameterValueStorage[extraIndex][2]=crr_momentum
            
            extraIndex = extraIndex + 1
            
            print('currnet index:')
            print(extraIndex)
            print('current eta:')
            print(crr_eta)
            print('current decay:')
            print(crr_weightDecay)
            print('current modemtum:')
            print(crr_momentum)
            

#Search maximum accuracy on test set
indexOfMaxAcc = np.argmax(finalAccStorage_test, axis=1)
bestParameterSetting =  parameterValueStorage[indexOfMaxAcc]
print(bestParameterSetting)

#final training
bestLearningRate = bestParameterSetting[0][0]
bestDecay = bestParameterSetting[0][1]
bestMomentum = bestParameterSetting[0][2]

NeuralNet_LeNet5_final = NeuralNet()
numOfEpochFinal = 80

final_accStorage_train, final_accStorage_test = wholeTrainProcessor(trainDataSetLoader,bestMomentum,bestLearningRate,bestDecay,NeuralNet_LeNet5_final,testDataSetLoader,numOfEpochFinal)


#plotting final learning curve
epochArr_final = np.zeros((1,numOfEpochFinal))
for k in range(numOfEpochFinal):
    epochArr_final[0][k] = k+1
plt.plot(epochArr_final, final_accStorage_train,'g^' , epochArr_final, final_accStorage_test, 'bs' )


#other plotting 

epochArr = np.zeros((1,numOfEpoch))
for i in range(numOfEpoch):
    epochArr[0][i] = i+1
#eta1
plt.plot(epochArr, accStorage_train_e1,'g^' , epochArr, accStorage_test_e1, 'bs')

#eta2
plt.plot(epochArr, accStorage_train_e2, 'g^' , epochArr, accStorage_test_e2, 'bs' )
#eta3
plt.plot(epochArr, accStorage_train_e3, 'g^' , epochArr, accStorage_test_e3, 'bs' )
#eta4
plt.plot(epochArr, accStorage_train_e4, 'g^' , epochArr, accStorage_test_e4, 'bs' )
#eta5
plt.plot(epochArr, accStorage_train_e5, 'g^' , epochArr, accStorage_test_e5, 'bs' )

#momentum1
plt.plot(epochArr, accStorage_train_m1, 'g^' , epochArr, accStorage_test_m1, 'bs' )
#momentum2
plt.plot(epochArr, accStorage_train_m2, 'g^' , epochArr, accStorage_test_m2, 'bs' )
#momentum3
plt.plot(epochArr, accStorage_train_m3, 'g^' , epochArr, accStorage_test_m3, 'bs' )
#momentum4
plt.plot(epochArr, accStorage_train_m4, 'g^' , epochArr, accStorage_test_m4, 'bs' )
#momentum5
plt.plot(epochArr, accStorage_train_m5, 'g^' , epochArr, accStorage_test_m5, 'bs' )

#Decay1
plt.plot(epochArr, accStorage_train_w1, 'g^' , epochArr, accStorage_test_w1, 'bs' )
#Decay2
plt.plot(epochArr, accStorage_train_w2, 'g^' , epochArr, accStorage_test_w2, 'bs' )
#Decay3
plt.plot(epochArr, accStorage_train_w3, 'g^' , epochArr, accStorage_test_w3, 'bs' )
#Decay4
plt.plot(epochArr, accStorage_train_w4, 'g^' , epochArr, accStorage_test_w4, 'bs' )
#Decay5
plt.plot(epochArr, accStorage_train_w5, 'g^' , epochArr, accStorage_test_w5, 'bs' )

#################normal

plt.plot(epochArr, accStorage_train_normal,'g^' , epochArr, accStorage_test_normal, 'bs')


#################Uniform

plt.plot(epochArr, accStorage_train_uniform,'g^' , epochArr, accStorage_test_uniform, 'bs')

#############constant

plt.plot(epochArr, accStorage_train_constant,'g^' , epochArr, accStorage_test_constant, 'bs')


############Ones

plt.plot(epochArr, accStorage_train_ones,'g^' , epochArr, accStorage_test_ones, 'bs')

#############Zeros

plt.plot(epochArr, accStorage_train_zeros,'g^' , epochArr, accStorage_test_zeros, 'bs')












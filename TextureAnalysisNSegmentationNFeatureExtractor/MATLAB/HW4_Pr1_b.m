%add paths for train and test data set
% addpath('C:\Users\tjdtk\Desktop1\EE 569\HW4\HW4_MATLAB\HW4_Pr1_a\dataSets\train')
% addpath('C:\Users\tjdtk\Desktop1\EE 569\HW4\HW4_MATLAB\HW4_Pr1_a\dataSets\test')
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW4\HW4_MATLAB\HW4_Pr1_b\dataSet_V2\train')
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW4\HW4_MATLAB\HW4_Pr1_b\dataSet_V2\test')


%Load train and test 15D and 3D data sets
%train
load('X_train_15D_V2')
load('reducedX_train_3D_V2')
%test
load('X_test_15D_V2')
load('reducedX_test_3D_V2')

%Construct y_train(i.e. label vector 36x1)
%blanket: 1, brick: 2, grass: 3, rice: 4
%9 blankets, 9 bricks, 9 grasses, 9 rices
y_train = [1;1;1;1;1;1;1;1;1;2;2;2;2;2;2;2;2;2;3;3;3;3;3;3;3;3;3;4;4;4;4;4;4;4;4;4];
y_test = [3;1;1;2;4;3;2;4;4;2;1;3]; %by human eyes
%% Unsupervised: K-mean clustering
numOfClusters = 4;
%train
estimated_y_kmean_by_15D = kmeans(X_train_15D,numOfClusters);
estimated_y_kmean_by_3D = kmeans(reducedX_train_3D,numOfClusters);

%test
estimated_y_kmean_by_15D_test = kmeans(X_test_15D,numOfClusters);
estimated_y_kmean_by_3D_test = kmeans(reducedX_test_3D,numOfClusters);



%Calculate error rate for train
%error rate by 15D
errorRateBy15D = errorRateCalculater(y_train,estimated_y_kmean_by_15D);

%error rate by 3D
errorRateBy3D = errorRateCalculater(y_train,estimated_y_kmean_by_3D);

%% Supervised: (1) Random Forest (2)Support Vector Machine
%(1)Random Forest
%Train model(i.e. grow trees)
NumTrees = 80; %Refered from EE660 homework 12
B = TreeBagger(NumTrees,reducedX_train_3D,y_train); %Train random forest model B

%Predict
predictedY_3DNRandomForest = cell2DoubleConverter( predict(B,reducedX_test_3D) );
errorRateBy3DNRandomForest = errorRateCalculater(y_test,predictedY_3DNRandomForest);

%(2)Support Vector Machine
t = templateSVM('Standardize',true);
SVMModel = fitcecoc(reducedX_train_3D,y_train,'Learners',t,...
    'ClassNames',{'1','2','3','4'});



predictedY_3DNSVM = cell2DoubleConverter(predict(SVMModel,reducedX_test_3D));
errorRateBy3DNSVM = errorRateCalculater(y_test,predictedY_3DNSVM);



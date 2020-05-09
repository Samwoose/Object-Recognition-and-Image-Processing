function reducedAveragedFeatureVector = featureValAverager(pixelwiseFeatures)
%FEATUREVALAVERAGER Summary of this function goes here
%Average pixelwise feature vector to 1 feature vector corresponding one image
%Furthermore, remove feature values that have the same value. This is because
%Same valued features doesn't help when it comes to machine learning
%training process.(i.e. same valued features are correlated each other, we
%want uncorrelated features which give us discriminant power)
%   Detailed explanation goes here

%Take absolute value for all pixelwise features
absPixelwiseFeatures = abs(pixelwiseFeatures);

numOfFeaturesBeforeReduction = 25;

numOfPixelsPerImg = size(pixelwiseFeatures,3); % 128x128 = 16384

averagedFeatureVector = zeros(1,numOfFeaturesBeforeReduction); %1x25 size

%Averaging each feature. resulting vector will be 1x25 size
for featureOrder = 1 : numOfFeaturesBeforeReduction
    averagedFeatureVector(1,featureOrder) = sum(absPixelwiseFeatures(1,featureOrder,:))/numOfPixelsPerImg;
end

%Remove correlated features. Resulting feature vector will be 1x15 size 
%filter orders in my implementation
%[1.L5L5 2.L5E5 3.L5S5 4.L5W5 5.L5R5 6.E5L5 7.E5E5 8.E5S5 9.E5W5 10.E5R5 11.S5L5 12.S5E5 13.S5S5 14.S5W5 15.S5R5 16.W5L5 17.W5E5 18.W5S5 19.W5W5 20.W5R5 21.R5L5 22.R5E5 23.R5S5 24.R5W5 25.R5R5 ]
%extract only 1, 2, 8, 7, 3, 9, 13, 4, 10, 19, 5, 14, 25, 20, 15 from the
%1x25 feature vector by averaging a pair of 2 "same" averaged feature values
%10 Pairs (2,6), (8,12), (3,11), (9,17), (4,16), (10,22), (5,21), (14,18),
%(20,24),  (15,23)
aveFeatureValBy_L5L5 = averagedFeatureVector(1,1);
aveFeatureValBy_L5E5 = (averagedFeatureVector(1,2)+ averagedFeatureVector(1,6))/2;
aveFeatureValBy_E5S5 = (averagedFeatureVector(1,8)+ averagedFeatureVector(1,12))/2;
aveFeatureValBy_E5E5 = averagedFeatureVector(1,7);
aveFeatureValBy_L5S5 = (averagedFeatureVector(1,3)+ averagedFeatureVector(1,11))/2;
aveFeatureValBy_E5W5 = (averagedFeatureVector(1,9)+ averagedFeatureVector(1,17))/2;
aveFeatureValBy_S5S5 = averagedFeatureVector(1,13);
aveFeatureValBy_L5W5 = (averagedFeatureVector(1,4)+ averagedFeatureVector(1,16))/2;
aveFeatureValBy_E5R5 = (averagedFeatureVector(1,10)+ averagedFeatureVector(1,22))/2;
aveFeatureValBy_W5W5 = averagedFeatureVector(1,19);
aveFeatureValBy_L5R5 = (averagedFeatureVector(1,5)+ averagedFeatureVector(1,21))/2;
aveFeatureValBy_S5W5 = (averagedFeatureVector(1,14)+ averagedFeatureVector(1,18))/2;
aveFeatureValBy_R5R5 = averagedFeatureVector(1,25);
aveFeatureValBy_W5R5 = (averagedFeatureVector(1,20)+ averagedFeatureVector(1,24))/2;
aveFeatureValBy_S5R5 = (averagedFeatureVector(1,15)+ averagedFeatureVector(1,23))/2;


reducedFeatureVector = [aveFeatureValBy_L5L5, aveFeatureValBy_L5E5, aveFeatureValBy_E5S5, aveFeatureValBy_E5E5, aveFeatureValBy_L5S5, aveFeatureValBy_E5W5, aveFeatureValBy_S5S5, aveFeatureValBy_L5W5, aveFeatureValBy_E5R5, aveFeatureValBy_W5W5, aveFeatureValBy_L5R5, aveFeatureValBy_S5W5, aveFeatureValBy_R5R5, aveFeatureValBy_W5R5, aveFeatureValBy_S5R5 ];

%Normalize features by averaged feature value by L5L5' filter
%Resulting feature vector will be 1x15 size but first element value would
%be 1 => e.g. [1, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]


reducedAveragedFeatureVector = elementwiseDividerByNumber(reducedFeatureVector,aveFeatureValBy_L5L5);    


end


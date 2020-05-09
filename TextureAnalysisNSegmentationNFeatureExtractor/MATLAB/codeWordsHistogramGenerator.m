function dummyOutput = codeWordsHistogramGenerator(inputImg1,inputImg2,labels_inputImg1,Centroids_inputImg1,orderOfFigure1,orderOfFigure2)
%CODEWORDSHISTOGRAMGENERATOR Summary of this function goes here
%Plot 2 histograms of code words based on input image1.
%e.g. inputImg1: Husky_3, inputImg2: Husky_2, Husky1, or Puppy1 in this
%practice
%   Detailed explanation goes here

dummyOutput = 1;

%Compute descriptor for each image by SIFT

%inputImg2
[featureFrame_inputImg2,descriptor_inputImg2] = modifiedSIFT_V1(inputImg2);

%Compute 8 centroids for each image by K means clustering
%need to transpose descriptors variable matrix to make it a proper
%parameter form
numOfClusters = 8;
%inputImg2
descriptor_inputImg2_tr = double(descriptor_inputImg2');
[labels_inputImg2,Centroids_inputImg2] = kmeans(descriptor_inputImg2_tr,numOfClusters);

%Find the closest corresponding each cluster(centroid) of inputImg2 to cluster from inputImg1
%and construct centroid conversion table
centroidConversionTable = zeros(1,numOfClusters);

for orderOfCentroid_img2 = 1:numOfClusters
    currentCentroid_img2 = Centroids_inputImg2(orderOfCentroid_img2,:);
    currentCorrespondingCentroid = correspondingCentroidFinder(currentCentroid_img2,Centroids_inputImg1);
    %save current corresponding centroid to the table
    centroidConversionTable(1,orderOfCentroid_img2) = currentCorrespondingCentroid;
end

%using the table, convert labels_inputImg2
convertedLabels_inputImg2 = labelConverter(labels_inputImg2,centroidConversionTable);

%Count number of each label
numberOfEachLabelStorage_inputImg1 = labelCounter(labels_inputImg1);
numberOfEachLabelStorage_inputImg2 = labelCounter(convertedLabels_inputImg2);

%plot histogram of code words
x=[1,2,3,4,5,6,7,8]; % orders of centrouds
figure(orderOfFigure1)
stem(x,numberOfEachLabelStorage_inputImg1)

figure(orderOfFigure2)
stem(x,numberOfEachLabelStorage_inputImg2)


end


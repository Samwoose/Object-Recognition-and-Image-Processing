%Load feature vectors calculated at Problem 1 (c)
load('transformedFeatureVectors_window25.mat');

%Apply PCA
[coeff_train, newdata_train, latent_train, tsquared_train, explained_train] = pca(transformedFeatureVectors);
%Take first 3 feature elements
reducedX_train_3D = [newdata_train(:,1),newdata_train(:,2),newdata_train(:,3)];

%Perform K means
numofClusters =6 ;
[clusterIndex, centroids] = kmeans(reducedX_train_3D,numofClusters);

%Converte clusterIndex to image
convertedImgHeight = 450;
convertedImgWidth = 600;
segmentImg_labelVersion = cluster2ImgConverter(clusterIndex,convertedImgHeight,convertedImgWidth);
%gray scale conversion
convertedImg = grayScaleConverter(segmentImg_labelVersion);

imshow(uint8(convertedImg))
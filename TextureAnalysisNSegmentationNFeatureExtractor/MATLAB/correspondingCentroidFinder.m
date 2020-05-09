function currentCorrespondingCentroid = correspondingCentroidFinder(currentCentroid_img2,Centroids_inputImg1)
%CORRESPONDINGCENTROIDFINDER Summary of this function goes here
%find a corresponding centroid among 8 centroids of image 1 to currnet centroid of image 1.  
%   Detailed explanation goes here

numOfCentroids = size(Centroids_inputImg1,1); % e.g. 8
distanceStorage = zeros(1,numOfCentroids);

%Calculate distance between each centroid from image1 and current centroid
%of image2 and save the value to the storage
for orderOfCentroid = 1: numOfCentroids
    currentCentroid_img1 = Centroids_inputImg1(orderOfCentroid,:);
    currentDistance = distanceCalculator(currentCentroid_img2,currentCentroid_img1);
    distanceStorage(1,orderOfCentroid) = currentDistance;
end

%find min and its index
[minDistance, indexOfMin] = min(distanceStorage);
%the index is the same as order of corresponding centroid.
currentCorrespondingCentroid = indexOfMin;
end


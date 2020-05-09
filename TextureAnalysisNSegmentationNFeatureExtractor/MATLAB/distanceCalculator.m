function currentDistance = distanceCalculator(currentCentroid_img2,currentCentroid_img1)
%DISTANCECALCULATOR Summary of this function goes here
%Calculate distance between each centroid from image1 and current centroid
%of image2 by euclidean distance
%   Detailed explanation goes here

%Use euclidean distance formula

acc_val = 0;
lengthOfVector = size(currentCentroid_img2,2); %e.g. 128

for orderOfVectorComponent = 1:lengthOfVector
    acc_val = acc_val + (currentCentroid_img1(1,orderOfVectorComponent)-currentCentroid_img2(1,orderOfVectorComponent))^2;
end

currentDistance = acc_val^0.5;


end


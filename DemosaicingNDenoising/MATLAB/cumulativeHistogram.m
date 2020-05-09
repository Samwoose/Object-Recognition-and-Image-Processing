function cumulativeVector = cumulativeHistogram(imgHeight,ImgWidth,numOfGrayScale)
%CUMULATIVEHISTOGRAM Summary of this function goes here
%   Detailed explanation goes here
numOfBallsInImg = imgHeight*ImgWidth;
numOfBuckets = numOfGrayScale; %256
ballsPerBuckets = numOfBallsInImg / numOfBuckets;

cumulativeVector = zeros(1,numOfBuckets);

for index = 1:numOfBuckets
    if index == 1
        cumulativeVector(index) = ballsPerBuckets;
    else
        cumulativeVector(index) = ballsPerBuckets + cumulativeVector(index - 1);
    end
end



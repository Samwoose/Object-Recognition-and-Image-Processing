function errorRate = errorRateCalculater(trueY,estimatedY)
%KMEANERRORRATECALCULATER Summary of this function goes here
%Calcuate error rate based on true label (trueY) , and estimated
%label(estimatedY) by K-mean clustering algorithm
%   Detailed explanation goes here

numOfAccurateEstimation = 0;
sizeOfLabelMatrix = size(trueY,1); %36

for index = 1:sizeOfLabelMatrix
    if(trueY(index,1) == estimatedY(index,1))
        numOfAccurateEstimation = numOfAccurateEstimation + 1;
    end
end

accuracyRate = numOfAccurateEstimation / sizeOfLabelMatrix;

errorRate = 1 - accuracyRate;



end


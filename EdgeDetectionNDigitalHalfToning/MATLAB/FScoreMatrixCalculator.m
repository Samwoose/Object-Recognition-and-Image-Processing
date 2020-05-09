function FScoreMatrix = FScoreMatrixCalculator(meanPMatrix,meanRMatrix)
%FSCOREMATRIXCALCULATOR Summary of this function goes here
%Calculate mean F score matrix based on mean P matrix and mean R matrix
%   Detailed explanation goes here
numOfIndex = size(meanPMatrix,1);
FScoreMatrix = zeros(numOfIndex,1);

for row = 1: numOfIndex
    FScoreMatrix(row,1) = 2*(meanPMatrix(row,1) * meanRMatrix(row,1))/(meanPMatrix(row,1) + meanRMatrix(row,1));  
end

end

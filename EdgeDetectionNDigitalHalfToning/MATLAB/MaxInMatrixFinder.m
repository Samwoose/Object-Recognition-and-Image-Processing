function [max,location] = MaxInMatrixFinder(inputMatrix)
%MAXINMATRIXFINDER Summary of this function goes here
%Find maximum value and its location in matrix and 
%   Detailed explanation goes here

max = -20000000;
sizeOfIndex = size(inputMatrix,1);
for row = 1: sizeOfIndex
    if(max<=inputMatrix(row,1))
        max = inputMatrix(row,1);
        location = row;
    end
end


end


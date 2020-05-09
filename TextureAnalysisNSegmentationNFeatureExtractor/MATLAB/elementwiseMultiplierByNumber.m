function outputMatrix = elementwiseMultiplierByNumber(inputMatrix1,number)
%ELEMENTWISEMULTIPLIERBYNUMBER Summary of this function goes here
%Perform elementwise multiplication given an input matrix and a number
%   Detailed explanation goes here

height = size(inputMatrix1,1);
width = size(inputMatrix1,2);
outputMatrix = zeros(height,width);

for row = 1:height
    for col = 1:width
        outputMatrix(row,col) = inputMatrix1(row,col)*number;
    end
end


end


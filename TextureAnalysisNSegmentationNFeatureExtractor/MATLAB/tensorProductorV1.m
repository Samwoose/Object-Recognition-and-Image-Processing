function outputMatrix = tensorProductorV1(inputMatrix1,inputMatrix2)
%TENSORPRODUCTOR Summary of this function goes here
%Perform tensor product with two given input matrix
%Each input matrix should be 1xN size.
%   Detailed explanation goes here

sizeOfInputMatrix1 = size(inputMatrix1,2);
sizeOfInputMatrix2 = size(inputMatrix2,2);

outputMatrix = zeros(sizeOfInputMatrix1,sizeOfInputMatrix2);

for row = 1:sizeOfInputMatrix2
    outputMatrix(row,:) = elementwiseMultiplierByNumber(inputMatrix1,inputMatrix2(1,row));
end


end


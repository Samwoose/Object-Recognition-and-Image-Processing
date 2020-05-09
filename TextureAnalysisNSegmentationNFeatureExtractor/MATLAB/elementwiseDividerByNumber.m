function outputMatrix = elementwiseDividerByNumber(inputMatrix1,number)
%ELEMENTWISEDIVIDER Summary of this function goes here
%Perform elementwise devision given two input matrix
%Equivalunt to inputMatrix1 ./ inputMatrix2
%   Detailed explanation goes here
height = size(inputMatrix1,1);
width = size(inputMatrix1,2);
outputMatrix = zeros(height,width);

for row = 1:height
    for col = 1:width
        outputMatrix(row,col) = inputMatrix1(row,col)/number;
    end
end


end


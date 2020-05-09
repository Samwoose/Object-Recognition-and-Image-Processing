function [ matrix_1scale ] = scalerChangerZero2One( matrix_255scale )
%SCALERCHANGERZERO2ONE Summary of this function goes here
%change value in matrix from range 0~255 to range 0~1
%   Detailed explanation goes here

imgHeight = size(matrix_255scale,1);
imgWidth = size(matrix_255scale,2);
matrix_1scale = zeros(imgHeight,imgWidth);
maxIntensity = 255;
for row = 1:imgHeight
    for col = 1:imgWidth
        matrix_1scale(row,col) = matrix_255scale(row,col)/maxIntensity;
    end
end

end


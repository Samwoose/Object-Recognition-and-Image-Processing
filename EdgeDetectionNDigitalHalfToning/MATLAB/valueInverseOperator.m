function [ valueInversedImgMatrix ] = valueInverseOperator( imgMatrix )
%VALUEINVERSEOPERATOR Summary of this function goes here
%inverse value of pixel in binarized image data.
%This is for evaluation processing later in problem 1 (d)
%e.g. 0 -> 255(Background, white), 255 -> 0(Edge, Black)
%   Detailed explanation goes here
imgHeight = size(imgMatrix,1);
imgWidth = size(imgMatrix,2);
Threshold = 127; %This is used to check if current value is 0 or 255 in input image matrix

valueInversedImgMatrix = zeros(imgHeight,imgWidth);

maxIntensity = 255;
minIntensity = 0;

for row = 1:imgHeight
    for col = 1:imgWidth
        if imgMatrix(row,col) > Threshold
            valueInversedImgMatrix(row,col)= minIntensity;
        else
            valueInversedImgMatrix(row,col)= maxIntensity;
        end
    end
end

end


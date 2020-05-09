function [ eightBitScaleImg ] = scalerChanger255( oneBitScaleImg )
%SCALERCHANGER255 Summary of this function goes here
%Change 1 bit scale image(0-1) to 8 bit scale image(0-255)
%   Detailed explanation goes here
imgHeight = size(oneBitScaleImg,1);
imgWidth = size(oneBitScaleImg,2);
eightBitScaleImg = zeros(imgHeight,imgWidth);
maxIntensity = 255;
for row = 1:imgHeight
    for col = 1:imgWidth
        eightBitScaleImg(row,col) = oneBitScaleImg(row,col)*maxIntensity;
    end
end

end


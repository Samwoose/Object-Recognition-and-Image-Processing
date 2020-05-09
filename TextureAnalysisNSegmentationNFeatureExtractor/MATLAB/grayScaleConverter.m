function convertedImg = grayScaleConverter(segmentImg_labelVersion);
%GRAYSCALECONVERTER Summary of this function goes here
%   Detailed explanation goes here
imgHeight = size(segmentImg_labelVersion,1);
imgWidth = size(segmentImg_labelVersion,2);
convertedImg = zeros(imgHeight,imgWidth);

for row = 1:imgHeight
    for col = 1:imgWidth
        if(segmentImg_labelVersion(row,col) == 1)
            convertedImg(row,col) = 0;
        elseif(segmentImg_labelVersion(row,col) == 2)
            convertedImg(row,col) = 51;
        elseif(segmentImg_labelVersion(row,col) == 3)
            convertedImg(row,col) = 102;
        elseif(segmentImg_labelVersion(row,col) == 4)
            convertedImg(row,col) = 153;
        elseif(segmentImg_labelVersion(row,col) == 5)
            convertedImg(row,col) = 204;
        elseif(segmentImg_labelVersion(row,col) == 6)
            convertedImg(row,col) = 255;
        end
    end
end


end


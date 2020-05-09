function segmentImg_labelVersion = cluster2ImgConverter(clusterIndex,convertedImgHeight,convertedImgWidth)
%CLUSTER2IMGCONVERTER Summary of this function goes here
%   Detailed explanation goes here
segmentImg_labelVersion = zeros(convertedImgHeight,convertedImgWidth);
index = 1;
for row = 1:convertedImgHeight
    for col = 1:convertedImgWidth
        segmentImg_labelVersion(row,col) = clusterIndex(index,1);
        index = index+1;
    end
end


end


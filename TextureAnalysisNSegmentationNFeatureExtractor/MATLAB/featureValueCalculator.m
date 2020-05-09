function featureValue = featureValueCalculator(subImg,LawsFilter)
%FEATUREVALUECALCULATOR Summary of this function goes here
%Calculate feature value corresponding center pixel value of sub image with
%current Laws filter
%Sub image is from the extended input image
%   Detailed explanation goes here

subImgHeight = size(subImg,1);
subImgWidth = size(subImg,2);

featureValue = 0;

for row = 1:subImgHeight
    for col = 1:subImgWidth
        featureValue = featureValue + subImg(row,col,1)*LawsFilter(row,col);
    end
end


end


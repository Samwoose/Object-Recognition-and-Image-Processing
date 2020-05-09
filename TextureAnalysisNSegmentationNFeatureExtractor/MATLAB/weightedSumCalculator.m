function weightedSumVal = weightedSumCalculator(currentSmallImg, chosenFilter)
%WEIGHTEDSUMCALCULATOR Summary of this function goes here
%   Detailed explanation goes here

imgHeight = size(currentSmallImg,1);
imgWidth = size(currentSmallImg,2);

weightedSumVal = 0;
for row = 1:imgHeight
    for col = 1:imgWidth
        weightedSumVal = weightedSumVal + currentSmallImg(row,col)*chosenFilter(row,col);
    end
end

end


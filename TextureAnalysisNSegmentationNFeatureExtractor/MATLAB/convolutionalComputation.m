function generatedImg = convolutionalComputation(img_comp,chosenFilter)
%CONVOLUTIONALCOMPUTATION Summary of this function goes here
%   Detailed explanation goes here
amountOfExtension = floor(size(chosenFilter,1)/2);

extendedImg = boundaryExtensioner(img_comp,amountOfExtension);
extendedImg2 = extendedImg;
extendedImgHeight = size(extendedImg,1);
extendedImgWidth = size(extendedImg,2);

for row = (1+amountOfExtension):(extendedImgHeight-amountOfExtension)
    for col = (1+amountOfExtension):(extendedImgWidth-amountOfExtension)
        upperRowLimit = row-amountOfExtension;
        lowerRowLimit = row+amountOfExtension;
        leftColLimit = col - amountOfExtension;
        rightColLimit = col + amountOfExtension;
        currentSmallImg = extendedImg(upperRowLimit:lowerRowLimit,leftColLimit:rightColLimit);
        weightedSumVal = weightedSumCalculator(currentSmallImg, chosenFilter);
        extendedImg2(row,col) = weightedSumVal;
        
    end
end
%crop extendedImg2
upperRowLimit2 = 1+amountOfExtension;
lowerRowLimit2 = extendedImgHeight-amountOfExtension;
leftColLimit2 = 1 + amountOfExtension;
rightColLimit2 = extendedImgWidth - amountOfExtension;
generatedImg = extendedImg2(upperRowLimit2:lowerRowLimit2,leftColLimit2:rightColLimit2);

end


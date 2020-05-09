function pixelWiseEnergyFeatureStorage = energyFeatureGenerator(imagesStroage,windowSize);
%ENERGYFEATUREGENERATOR Summary of this function goes here
%   Detailed explanation goes here

dimensionOfFeature = size(imagesStroage,3);
imgHeight = size(imagesStroage,1);
imgWidth = size(imagesStroage,2);

amountOfExtension = floor(windowSize/2);
extendedImgHeight = imgHeight + amountOfExtension*2 ;
extendedImgWidth = imgWidth + amountOfExtension*2;
extendedImgStorage = zeros(extendedImgHeight,extendedImgWidth,dimensionOfFeature);

for orderOfElement = 1:dimensionOfFeature
    currentExtendedImg = boundaryExtensioner(imagesStroage(:,:,orderOfElement),amountOfExtension);
    for row = (1+amountOfExtension):(extendedImgHeight-amountOfExtension)
        for col = (1+amountOfExtension):(extendedImgWidth-amountOfExtension)
            upperRowLimit = row-amountOfExtension;
            lowerRowLimit = row+amountOfExtension;
            leftColLimit = col - amountOfExtension;
            rightColLimit = col + amountOfExtension;
            currentSmallImg = currentExtendedImg(upperRowLimit:lowerRowLimit,leftColLimit:rightColLimit);
            currentEnergy = energyCalculator(currentSmallImg);
            
            extendedImgStorage(row,col,orderOfElement) = currentEnergy;
            
        
        end
    end

end
upperRowLimit2 = 1+amountOfExtension;
lowerRowLimit2 = extendedImgHeight-amountOfExtension;
leftColLimit2 = 1 + amountOfExtension;
rightColLimit2 = extendedImgWidth - amountOfExtension;
pixelWiseEnergyFeatureStorage = extendedImgStorage(upperRowLimit2:lowerRowLimit2,leftColLimit2:rightColLimit2,:);

end

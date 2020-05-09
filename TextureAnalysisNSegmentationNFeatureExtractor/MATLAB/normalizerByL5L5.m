function finalNormalizedPixelWiseEnergyFeatureStorage = normalizerByL5L5(pixelWiseEnergyFeatureStorage)
%NORMALIZERBYL5L5 Summary of this function goes here
%   Detailed explanation goes here


dimensionOfFeature = size(pixelWiseEnergyFeatureStorage,3);
imgHeight = size(pixelWiseEnergyFeatureStorage,1);
imgWidth = size(pixelWiseEnergyFeatureStorage,2);

normalizedPixelWiseEnergyFeatureStorage = zeros(imgHeight,imgWidth,dimensionOfFeature);
for row = 1:imgHeight
    for col = 1:imgWidth
        currentEnergyByL5L5 = pixelWiseEnergyFeatureStorage(row,col,1);
        
        for orderOfElement = 1:dimensionOfFeature
            normalizedEnergy = pixelWiseEnergyFeatureStorage(row,col,orderOfElement)/currentEnergyByL5L5;
            normalizedPixelWiseEnergyFeatureStorage(row,col,orderOfElement) = normalizedEnergy;
        end
    end

end

%crop normalized storage
finalNormalizedPixelWiseEnergyFeatureStorage = normalizedPixelWiseEnergyFeatureStorage(:,:,2:15);


end


function transformedFeatureVectors = XtrainCreator(normalizedPixelWiseEnergyFeatureStorage);
%XTRAINCREATOR Summary of this function goes here
%   Detailed explanation goes here

dimensionOfFeature = size(normalizedPixelWiseEnergyFeatureStorage,3);
imgHeight = size(normalizedPixelWiseEnergyFeatureStorage,1);
imgWidth = size(normalizedPixelWiseEnergyFeatureStorage,2);

numOfFeatures = imgHeight*imgWidth;
transformedFeatureVectors =zeros(numOfFeatures,dimensionOfFeature); %27000*14

index = 1;
for row = 1:imgHeight
    for col = 1:imgWidth
        transformedFeatureVectors(index,:) = normalizedPixelWiseEnergyFeatureStorage(row,col,:);
        index = index+1;
    end
end




end


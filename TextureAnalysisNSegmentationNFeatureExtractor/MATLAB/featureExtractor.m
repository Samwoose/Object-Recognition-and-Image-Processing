function pixelwiseFeatures = featureExtractor(inputImg,filterBank)
%FEATUREEXTRACTOR Summary of this function goes here
%Extract feature vector from each pixel. Each feature vector has 25-Dim
%because we are using 25 Laws filters.
%   Detailed explanation goes here

amountOfOneSideExtension = 2 ; % each size is extended by 2 because we are using 5x5 Laws filter

%extendedImg = boundaryExtensioner(inputImg,amountOfOneSideExtension);
extendedImg = boundaryExtensionerV2(inputImg);

extendedImgHeight = size(extendedImg,1);
extendedImgWidth = size(extendedImg,2);

originalImgHeight = size(extendedImg,1) - 2*amountOfOneSideExtension;
originalImgWidth = size(extendedImg,2) - 2*amountOfOneSideExtension;

numOfFilters = size(filterBank,3);
numOfPixels = originalImgHeight*originalImgWidth;

pixelwiseFeatures = zeros(1,numOfFilters,numOfPixels); %1x25x16384(=128x128) size
pixelwiseFeaturesIndex = 1;

for row = 1+amountOfOneSideExtension:extendedImgHeight-amountOfOneSideExtension
    for col = 1+amountOfOneSideExtension:extendedImgWidth-amountOfOneSideExtension
        for filterOrder = 1:numOfFilters
            subImg = extendedImg(row-amountOfOneSideExtension:row+amountOfOneSideExtension,col-amountOfOneSideExtension:col+amountOfOneSideExtension,1); %e.g. 5x5x1
            pixelwiseFeatures(1,filterOrder,pixelwiseFeaturesIndex) = featureValueCalculator(subImg,filterBank(:,:,filterOrder));
        end
        pixelwiseFeaturesIndex = pixelwiseFeaturesIndex + 1;
        
    end
end

disp('pixelwise features are extracted')



end


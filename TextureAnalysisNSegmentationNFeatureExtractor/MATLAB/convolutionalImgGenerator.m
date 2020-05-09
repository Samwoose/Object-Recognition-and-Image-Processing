function imagesStorage = convolutionalImgGenerator(img_comp, filterBank)
%CONVOLUTIONALIMGGENERATOR Summary of this function goes here

%   Detailed explanation goes here
imgHeight = size(img_comp,1);
imgWidth = size(img_comp,2);
numOfImg = 15;
numOfChosenFilters = 15;

imagesStorage = zeros(imgHeight, imgWidth,numOfImg);

chosenFilterIndex = [1,2,3,4,5,7,8,9,10,13,14,15,19,20,25];

for filterOrder = 1:numOfChosenFilters
    chosenFilter = filterBank(:,:,chosenFilterIndex(1,filterOrder));
    generatedImg = convolutionalComputation(img_comp,chosenFilter);
    imagesStorage(:,:,filterOrder) = generatedImg ;
end


end


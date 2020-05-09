function extendedImg = boundaryExtensioner(inputImg,amountOfExtension)
%BOUNDARYEXTENSIONER Summary of this function goes here
%extend boundary of image by amount of extension. 
%e.g. amountOfExtension 2 => extendedImg size is (4+N) X (4+N)
%   Detailed explanation goes here

inputImgHeight = size(inputImg,1);
inputImgWidth = size(inputImg,2);

outputImgHeight = inputImgHeight + 2 * amountOfExtension;
outputImgWidth = inputImgWidth + 2 * amountOfExtension;

extendedImg = zeros(outputImgHeight,outputImgWidth,1);

startingRow = amountOfExtension + 1;
endRow = outputImgHeight - amountOfExtension;

startingCol = amountOfExtension + 1;
endCol = outputImgWidth - amountOfExtension;

for row = startingRow:endRow
    for col = startingCol:endCol
        inputImgRow = row - amountOfExtension;
        inputimgCol = col - amountOfExtension;
        extendedImg(row,col,1) = inputImg(inputImgRow,inputimgCol,1);
    end
end


end


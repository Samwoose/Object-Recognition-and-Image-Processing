function currentEnergy = energyCalculator(currentSmallImg);
%ENERGYCALCULATOR Summary of this function goes here
%   Detailed explanation goes here
imgHeight = size(currentSmallImg,1);
imgWidth = size(currentSmallImg,2);

poweredValImg = currentSmallImg.^2;

accVal = 0;
for row = 1:imgHeight
    for col = 1:imgWidth
        accVal  = accVal  + poweredValImg(row,col);
    end
end

currentEnergy = accVal/(imgHeight*imgWidth);

    
end


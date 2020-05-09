function numberOfEachLabelStorage_inputImg = labelCounter(labels_inputImg)
%LABELCOUNTER Summary of this function goes here
%Count number of each label in labels_inputImg
% labels_inputImg's data size is 1090x1 for husky3 image
%   Detailed explanation goes here

numOfLabel1 = 0;
numOfLabel2 = 0;
numOfLabel3 = 0;
numOfLabel4 = 0;
numOfLabel5 = 0;
numOfLabel6 = 0;
numOfLabel7 = 0;
numOfLabel8 = 0;

numOfTotalLabels = size(labels_inputImg,1); %e.g. 1090

for orderOfLabel = 1:numOfTotalLabels
    if(labels_inputImg(orderOfLabel,1)==1)
        numOfLabel1 = numOfLabel1 +1;
    elseif(labels_inputImg(orderOfLabel,1)==2)
        numOfLabel2 = numOfLabel2 +1;
    elseif(labels_inputImg(orderOfLabel,1)==3)
        numOfLabel3 = numOfLabel3 +1;
    elseif(labels_inputImg(orderOfLabel,1)==4)
        numOfLabel4 = numOfLabel4 +1;
    elseif(labels_inputImg(orderOfLabel,1)==5)
        numOfLabel5 = numOfLabel5 +1;
    elseif(labels_inputImg(orderOfLabel,1)==6)
        numOfLabel6 = numOfLabel6 +1;
    elseif(labels_inputImg(orderOfLabel,1)==7)
        numOfLabel7 = numOfLabel7 +1;
    elseif(labels_inputImg(orderOfLabel,1)==8)
        numOfLabel8 = numOfLabel8 +1;
    end
end

numberOfEachLabelStorage_inputImg = [numOfLabel1,numOfLabel2,numOfLabel3,numOfLabel4,numOfLabel5,numOfLabel6,numOfLabel7,numOfLabel8];

end


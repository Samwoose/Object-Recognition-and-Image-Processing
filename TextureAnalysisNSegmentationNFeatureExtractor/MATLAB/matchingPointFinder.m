function pairsOfMatchingPoints = matchingPointFinder(inputImg1,inputImg2);
%MATCHINGPOINTFINDER Summary of this function goes here
%   Detailed explanation goes here

%find descriptors for each input image
[featureFrame_inputImg1,descriptor_inputImg1] = modifiedSIFT_V1(inputImg1);
[featureFrame_inputImg2,descriptor_inputImg2] = modifiedSIFT_V1(inputImg2);

%Use matching points finder funciton
pairsOfMatchingPoints = vl_ubcmatch(descriptor_inputImg1,descriptor_inputImg2);

end


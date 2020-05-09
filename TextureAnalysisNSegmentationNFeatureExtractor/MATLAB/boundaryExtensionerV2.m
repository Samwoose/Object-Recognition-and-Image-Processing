function extendedImg = boundaryExtensionerV2(inputImg)
%BOUNDARYEXTENSIONER Summary of this function goes here
%extend boundary of image by amount of extension. 
%e.g. amountOfExtension 2 => extendedImg size is (4+N) X (4+N)
%   Detailed explanation goes here

extendedImg = zeros(132,132);


%Details on portions is in HW4 sketch paper page 17
%Original portion
extendedImg(3:130,3:130) = inputImg;
%1
extendedImg(3:130,1:2) = inputImg(1:128,1:2);
%2
extendedImg(3:130,131:132) = inputImg(1:128,127:128);
%3
extendedImg(1:2,3:130) = inputImg(1:2,1:128);
%4
extendedImg(131:132,3:130) = inputImg(127:128,1:128);

%5
extendedImg(1:2,1:2) = inputImg(1:2,1:2);
%6
extendedImg(1:2,131:132) = inputImg(1:2,127:128);
%7
extendedImg(131:132,1:2) = inputImg(127:128,1:2);
%8
extendedImg(131:132,131:132) = inputImg(127:128,127:128);


end


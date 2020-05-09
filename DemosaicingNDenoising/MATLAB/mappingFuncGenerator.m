function mappingVector = mappingFuncGenerator(intensityFrequency_OneChannel,imgHeight, imgWidth)
%MAPPINGFUNCGENERATOR Summary of this function goes here
%   Detailed explanation goes here
maxValOfIntensity = 255;

%calculate normalized probability based on intensity frequency for one
%channel
%construct 
probabilityVector = zeros(1, maxValOfIntensity+1) ; %size 1 X 256

totalNumPixels = imgHeight*imgWidth;

for intensity = 1:(maxValOfIntensity+1)
    probabilityVector(intensity) = intensityFrequency_OneChannel(intensity)/totalNumPixels;
end

%Calculate Cummulative Density Function(CDF) for each intensity and return a 1D vector than includes CDF for each intensity
CDFVector = zeros(1,maxValOfIntensity+1); %size 1X256

for index = 1:(maxValOfIntensity+1)
    if index ==1
        CDFVector(index) = probabilityVector(index);
    else
        CDFVector(index) = probabilityVector(index) + CDFVector(index-1);
    end
end

%Calculate transfer function a.k.a mapping function
mappingVector = zeros(1,maxValOfIntensity+1); %size 1X256

for index = 1: (maxValOfIntensity+1)
    mappingVector(index) = floor(CDFVector(index)*maxValOfIntensity); 
end


end


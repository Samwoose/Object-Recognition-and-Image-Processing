function [maxscale,indexOfMaxScale] = indexOfKeypointWithMaxScaleFinder(descriptor)
%INDEXOFKEYPOINTWITHMAXSCALEFINDER Summary of this function goes here
%find index and max scale of keypoint in given descriptor
%   Detailed explanation goes here

sizeOfMagnitudeStorage = size(descriptor,2); %e.g. 1098 for Husky3

magnitudeStorage = zeros(1,sizeOfMagnitudeStorage);

for index = 1:sizeOfMagnitudeStorage
    currentKeypoint = descriptor(:,index);
    magnitudeStorage(1,index) = magnitudeCalculator(currentKeypoint);
end

[maxscale, indexOfMaxScale] = max(magnitudeStorage);


end


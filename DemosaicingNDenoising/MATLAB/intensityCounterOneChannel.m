function oneChannel_Vec = intensityCounterOneChannel(oneChannel,imgHeight, imgWidth)
%INTENSITYCOUNTERONECHANNEL Summary of this function goes here
%   Detailed explanation goes here
maxValOfIntensity = 255;
one = 1;

oneChannel_Vec = zeros(1,maxValOfIntensity+one);

for row = 1:imgHeight
    for col = 1:imgWidth
        intensityVal = int32(oneChannel(row,col,1));
        %Be careful with index
        oneChannel_Vec(intensityVal+1) = oneChannel_Vec(intensityVal+1) + 1;
    end
end


end


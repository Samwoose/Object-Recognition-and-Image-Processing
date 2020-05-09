function blueChannelData = getBlueChannel(threeChannelData, imgHeight, imgWidth)
%GETREDCHANNEL Summary of this function goes here
%   Detailed explanation goes here

%Construct one channel data
blueChannelData = zeros(imgHeight, imgWidth, 1);

for row = 1:imgHeight
   for col = 1:imgWidth
       blueChannelData(row,col,1) = threeChannelData(row,col,3);
   end
end

end
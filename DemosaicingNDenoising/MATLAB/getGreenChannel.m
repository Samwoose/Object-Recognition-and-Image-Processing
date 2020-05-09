function greenChannelData = getGreenChannel(threeChannelData, imgHeight, imgWidth)
%GETREDCHANNEL Summary of this function goes here
%   Detailed explanation goes here

%Construct one channel data
greenChannelData = zeros(imgHeight, imgWidth, 1);

for row = 1:imgHeight
   for col = 1:imgWidth
       greenChannelData(row,col,1) = threeChannelData(row,col,2);
   end
end

end
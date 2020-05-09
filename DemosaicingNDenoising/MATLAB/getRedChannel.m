function redChannelData = getRedChannel(threeChannelData, imgHeight, imgWidth)
%GETREDCHANNEL Summary of this function goes here
%   Detailed explanation goes here

%Construct one channel data
redChannelData = zeros(imgHeight, imgWidth, 1);

for row = 1:imgHeight
   for col = 1:imgWidth
       redChannelData(row,col,1) = threeChannelData(row,col,1);
   end
end

end


function inversedEdgeImg = inverseEdgeNBackground(edgeImg)
%INVERSEEDGENBACKGROUND Summary of this function goes here
%inverse edge and background.
%Edge(255) become 0
%Background(0) becomes 255
%   Detailed explanation goes here
imgHeight = size(edgeImg,1);
imgWidth = size(edgeImg,2);

inversedEdgeImg = zeros(imgHeight,imgWidth);
intensityOfBlack = 0;
intensityOfWhite = 255;

for row = 1:imgHeight
    for col = 1:imgWidth
        if(edgeImg(row,col) == intensityOfBlack)
            inversedEdgeImg(row,col) = intensityOfWhite;
        else
            inversedEdgeImg(row,col) = intensityOfBlack;
        end
    end
end

end


function BinaryEdgeMap = ProbabilityBinarizer( probabilityEdgeMap, probabilityThreshold )
%PBINARIZER Summary of this function goes here
%probability Edge Map: Probability edge map from structured edge detection
%algorith
%probabilityThreshold: Probability threshold within 0~1
%   Detailed explanation goes here

imgHeight = size(probabilityEdgeMap,1);
imgWidth = size(probabilityEdgeMap,2);
BinaryEdgeMap = zeros(imgHeight,imgWidth);
probabilityOfedge1 = 1;%meaning probability of edge is 100%
probabilityOfedge2 = 0;%meaning probability of edge is 0%
for row = 1:imgHeight
    for col = 1:imgWidth
        if probabilityEdgeMap(row, col) > probabilityThreshold
            BinaryEdgeMap(row,col) = probabilityOfedge1; %Edge
        else
            BinaryEdgeMap(row,col) = probabilityOfedge2; %No edge(i.e., background)
        end
    end
end


end


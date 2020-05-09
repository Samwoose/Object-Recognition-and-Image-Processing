function magnitude = magnitudeCalculator(currentKeypointVector)
%MAGNITUDECALCULATOR Summary of this function goes here
%Calculate magnitude of current keypoint vector
%   Detailed explanation goes here

%transpose 128x1 vector to 1x128 vector
currentKeypointVector_tr = double(currentKeypointVector');
lenOfKeypointVector = size(currentKeypointVector_tr,2);

acc_squareVal = 0;

for index = 1:lenOfKeypointVector
    acc_squareVal = acc_squareVal + currentKeypointVector_tr(1,index)^2;
end

magnitude = acc_squareVal^0.5;



end


function [ meanOfInput ] = meanCalculator( input )
%MEANCALCULATOR Summary of this function goes here
%Calculate mean with respect to the given values in input
%   Detailed explanation goes here
sizeOfInput = size(input,1); %99 x 1 => 99 is the size of the input
temp_val = 0; %Need to track sum of values
for index = 1:sizeOfInput
    temp_val = temp_val + input(index,1);
end

%Calculate mean
meanOfInput = temp_val / sizeOfInput;


end


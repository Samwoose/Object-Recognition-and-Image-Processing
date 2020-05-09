function predictedY_double = cell2DoubleConverter(predictedY_cell)
%CELL2DOUBLE Summary of this function goes here
%Convert cell type to double
%   Detailed explanation goes here

sizeOfCell = size(predictedY_cell,1); % e.g. 12
predictedY_double = zeros(sizeOfCell,1);

for index = 1:sizeOfCell
    predictedY_double(index,1) = str2double(predictedY_cell{index,1});
end


end


function filterBank = filterBankGenerator()
%FILTERBANKGENERATOR Summary of this function goes here
%Generate a bank that contains 25 5x5 Laws filters.
%Bank size will be 5x5x25
%   Detailed explanation goes here

sizeOfKernel = 5;
numOfFilters = 25;
filterBank = zeros(sizeOfKernel,sizeOfKernel,numOfFilters); %5x5x25 size

%Five 1D Kernels
L5 = [1 4 6 4 1]; %Level
E5 = [-1 -2 0 2 1] ; %Edge
S5 = [-1 0 2 0 -1] ; %Spot
W5 = [-1 2 0 -2 1] ; %Wave
R5 = [1 -4 6 -4 1] ; %Ripple

Five1DKernels1 = [L5 ; E5 ; S5 ; W5 ; R5];
Five1DKernels2 = [L5 ; E5 ; S5 ; W5 ; R5];

bankIndex = 1;


%filterBank order
%[L5L5 , L5E5, L5S5, L5W5, L5R5, E5L5, E5E5, E5S5, E5W5, E5R5, S5L5, S5E5, S5S5, S5W5, S5R5, W5L5, W5E5, W5S5, W5W5, W5R5, R5L5, R5E5, R5S5, R5W5, R5R5]
for firstKernelIndex = 1:sizeOfKernel
    for secondKernelIndex = 1:sizeOfKernel
        currentKernel1 = Five1DKernels1(firstKernelIndex,:);
        currentKernel2 = Five1DKernels2(secondKernelIndex,:);
        filterBank(:,:,bankIndex) = tensorProductorV1(currentKernel1,currentKernel2);
        
        bankIndex = bankIndex + 1;
    end
end

end


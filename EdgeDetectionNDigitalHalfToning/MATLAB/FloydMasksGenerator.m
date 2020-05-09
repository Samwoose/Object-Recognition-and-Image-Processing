function chosenFloydMask = FloydMasksGenerator(indicatorOfFloyd)
%FLOYDMASKS Summary of this function goes here
%Return a floyd mask based on name of floyd in parameter
%   Detailed explanation goes here
chosenFloydMask = zeros(3,3); %Floyd mask is 3x3 size


if(indicatorOfFloyd == 1)
    %create left to right direction floyd mask
    chosenFloydMask(1,1) =0.0 ;
    chosenFloydMask(1,2) =0.0 ;
    chosenFloydMask(1,3) =0.0 ;
    
    chosenFloydMask(2,1) =0.0 ;
    chosenFloydMask(2,2) =0.0 ;
    chosenFloydMask(2,3) =7.0/16.0 ;
    
    chosenFloydMask(3,1) = 3.0/16.0;
    chosenFloydMask(3,2) = 5.0/16.0;
    chosenFloydMask(3,3) = 1.0/16.0;
    
elseif(indicatorOfFloyd == 2)
    %create right to left direction floyd mask
    chosenFloydMask(1,1) =0.0 ;
    chosenFloydMask(1,2) =0.0 ;
    chosenFloydMask(1,3) =0.0 ;
    
    chosenFloydMask(2,1) =7.0/16.0 ;
    chosenFloydMask(2,2) =0.0 ;
    chosenFloydMask(2,3) =0.0 ;
    
    chosenFloydMask(3,1) = 1.0/16.0;
    chosenFloydMask(3,2) = 5.0/16.0;
    chosenFloydMask(3,3) = 3.0/16.0;
            
end


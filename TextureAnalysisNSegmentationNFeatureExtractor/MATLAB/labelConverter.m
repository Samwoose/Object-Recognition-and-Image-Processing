function convertedLabels_inputImg2 = labelConverter(labels_inputImg2,centroidConversionTable)
%LABELCONVERTER Summary of this function goes here
%Convert labels of input image 2 refering to a centroid conversion table
%   Detailed explanation goes here

correspondingCentroidToLabel1_img2 = centroidConversionTable(1,1);
correspondingCentroidToLabel2_img2 = centroidConversionTable(1,2);
correspondingCentroidToLabel3_img2 = centroidConversionTable(1,3);
correspondingCentroidToLabel4_img2 = centroidConversionTable(1,4);
correspondingCentroidToLabel5_img2 = centroidConversionTable(1,5);
correspondingCentroidToLabel6_img2 = centroidConversionTable(1,6);
correspondingCentroidToLabel7_img2 = centroidConversionTable(1,7);
correspondingCentroidToLabel8_img2 = centroidConversionTable(1,8);

img2_Label1 = 1;
img2_Label2 = 2;
img2_Label3 = 3;
img2_Label4 = 4;
img2_Label5 = 5;
img2_Label6 = 6;
img2_Label7 = 7;
img2_Label8 = 8;

lengthOfLabelVector = size(labels_inputImg2,1); %e.g. 1090 
convertedLabels_inputImg2 = labels_inputImg2;

%Image 2 Label Conversion process 
for orderOfLabel = 1:lengthOfLabelVector
    if(labels_inputImg2(orderOfLabel,1) == img2_Label1)
      convertedLabels_inputImg2(orderOfLabel,1) = correspondingCentroidToLabel1_img2;
    elseif(labels_inputImg2(orderOfLabel,1) == img2_Label2)
      convertedLabels_inputImg2(orderOfLabel,1) = correspondingCentroidToLabel2_img2;
    elseif(labels_inputImg2(orderOfLabel,1) == img2_Label3)
      convertedLabels_inputImg2(orderOfLabel,1) = correspondingCentroidToLabel3_img2;
    elseif(labels_inputImg2(orderOfLabel,1) == img2_Label4)
      convertedLabels_inputImg2(orderOfLabel,1) = correspondingCentroidToLabel4_img2;
    elseif(labels_inputImg2(orderOfLabel,1) == img2_Label5)
      convertedLabels_inputImg2(orderOfLabel,1) = correspondingCentroidToLabel5_img2;
    elseif(labels_inputImg2(orderOfLabel,1) == img2_Label6)
      convertedLabels_inputImg2(orderOfLabel,1) = correspondingCentroidToLabel6_img2;
    elseif(labels_inputImg2(orderOfLabel,1) == img2_Label7)
      convertedLabels_inputImg2(orderOfLabel,1) = correspondingCentroidToLabel7_img2;  
    elseif(labels_inputImg2(orderOfLabel,1) == img2_Label8)
      convertedLabels_inputImg2(orderOfLabel,1) = correspondingCentroidToLabel8_img2;       
    end
end


end


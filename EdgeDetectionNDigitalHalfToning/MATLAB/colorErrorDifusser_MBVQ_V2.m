function output_RGB = colorErrorDifusser_MBVQ_V2(ColorImg,left2RightDirection_Mask,right2LeftDirection_Mask)
%COLORERRORDIFUSSER_MBVQ Summary of this function goes here
%perform MBVQ error diffusion with two given masks 
%   Detailed explanation goes here
filterSize = size(left2RightDirection_Mask,1);
amountOfExtension = floor(filterSize/2);
imgHeight = size(ColorImg,1);
imgWidth = size(ColorImg,2);

bytePerPixel = size(ColorImg,3);
outputImgHeight = imgHeight ;
outputImgWidth = imgWidth ;

processingMatrix = ColorImg ; %Copy the input color img for the processing

output_RGB = zeros(outputImgHeight,outputImgWidth,bytePerPixel); %Initialize output_RGB

for row = 1:imgHeight
    %Forward error diffusion
    if(mod(row,2) == 1)
        for col = 1:imgWidth
            %Get original R,G,B values
            current_R = ColorImg(row,col,1);
            current_G = ColorImg(row,col,2);
            current_B = ColorImg(row,col,3);
            %Get R+error, G+error, B+error values
            current_RNError = processingMatrix(row,col,1);
            current_GNError = processingMatrix(row,col,2);
            current_BNError = processingMatrix(row,col,3);
            
            %Find MBVQ based on original RGB
            current_MBVQ = MBVQIdentifier(current_R, current_G,current_B);
            %Find the closet vertex based on the MBVQ and R,G,B +
            %accumulated error
            current_vertex = getNearestVertexV2(current_MBVQ,current_RNError,current_GNError,current_BNError);
            
            %Save binarized intensity
            %R
            output_RGB(row,col,1) = current_vertex(1);
            %G
            output_RGB(row,col,2) = current_vertex(2);
            %B
            output_RGB(row,col,3) = current_vertex(3);
            
            %Calculate error for each channel
            %R
            current_error_R = current_RNError - output_RGB(row,col,1);
            %G
            current_error_G = current_GNError - output_RGB(row,col,2);
            %B
            current_error_B = current_BNError - output_RGB(row,col,3);
            
            %Diffuse error for each channel
            for filter_row = int64(-amountOfExtension):int64(amountOfExtension)
                for filter_col = int64(-amountOfExtension):int64(amountOfExtension)
                    true_filter_row = filter_row + 2;
                    true_filter_col = filter_col + 2;
                    current_row = row + filter_row;
                    current_col = col + filter_col;
                    
                    if(current_row >= 1 && current_row <= imgHeight && current_col >=1 && current_col <=imgWidth)
                        %R
                        processingMatrix(current_row, current_col,1) = processingMatrix(current_row, current_col,1) + current_error_R*left2RightDirection_Mask(true_filter_row,true_filter_col); 
                        %G
                        processingMatrix(current_row, current_col,2) = processingMatrix(current_row, current_col,2) + current_error_G*left2RightDirection_Mask(true_filter_row,true_filter_col);                       
                        %B
                        processingMatrix(current_row, current_col,3) = processingMatrix(current_row, current_col,3) + current_error_B*left2RightDirection_Mask(true_filter_row,true_filter_col); 
                    end
                end
            end
                  
        end
    
    %backward error diffusion
    elseif(mod(row,2)==0)
        for col = imgWidth:-1:1
            %Get original R,G,B values
            current_R = ColorImg(row,col,1);
            current_G = ColorImg(row,col,2);
            current_B = ColorImg(row,col,3);
            %Get R+error, G+error, B+error values
            current_RNError = processingMatrix(row,col,1);
            current_GNError = processingMatrix(row,col,2);
            current_BNError = processingMatrix(row,col,3);

            %Find MBVQ based on original RGB
            current_MBVQ = MBVQIdentifier(current_R, current_G,current_B);
            %Find the closet vertex based on the MBVQ and R,G,B +
            %accumulated error
            current_vertex = getNearestVertexV2(current_MBVQ,current_RNError,current_GNError,current_BNError);
            
            %Save binarized intensity
            %R
            output_RGB(row,col,1) = current_vertex(1);
            %G
            output_RGB(row,col,2) = current_vertex(2);
            %B
            output_RGB(row,col,3) = current_vertex(3);
            
            %Calculate error for each channel
            %R
            current_error_R = current_RNError - output_RGB(row,col,1);
            %G
            current_error_G = current_GNError - output_RGB(row,col,2);
            %B
            current_error_B = current_BNError - output_RGB(row,col,3);
            
            %Diffuse error for each channel
            for filter_row = int64(-amountOfExtension):int64(amountOfExtension)
                for filter_col = int64(-amountOfExtension):int64(amountOfExtension)
                    true_filter_row = filter_row + 2;
                    true_filter_col = filter_col + 2;
                    current_row = row + filter_row;
                    current_col = col + filter_col;
                    
                    if(current_row >= 1 && current_row <= imgHeight && current_col >=1 && current_col <=imgWidth)
                        %R
                        processingMatrix(current_row, current_col,1) = processingMatrix(current_row, current_col,1) + current_error_R*right2LeftDirection_Mask(true_filter_row,true_filter_col); 
                        %G
                        processingMatrix(current_row, current_col,2) = processingMatrix(current_row, current_col,2) + current_error_G*right2LeftDirection_Mask(true_filter_row,true_filter_col);                       
                        %B
                        processingMatrix(current_row, current_col,3) = processingMatrix(current_row, current_col,3) + current_error_B*right2LeftDirection_Mask(true_filter_row,true_filter_col); 
                    end
                end
            end
                  
        end
    end
end
            
            
        
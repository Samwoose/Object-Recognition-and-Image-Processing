function Gray2DImg = RGB2Gray(RGB2DImag)
%RGB2GRAY Summary of this function goes here
%   Detailed explanation goes here


imgHeight = size(RGB2DImag,1);%row
imgWidth = size(RGB2DImag,2);%col

Gray2DImg = zeros(imgHeight,imgWidth);


for row = 1:imgHeight
    for col = 1:imgWidth
        %coversion from RGB to Gray formula. Refer to HW2 description
        Gray2DImg(row,col) = 0.2989*RGB2DImag(row,col,1) + 0.5870*RGB2DImag(row,col,2) + 0.1140*RGB2DImag(row,col,3);
        
    end
end
            

end


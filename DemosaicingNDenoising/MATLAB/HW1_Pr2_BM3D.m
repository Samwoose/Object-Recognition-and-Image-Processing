%Read .raw file
%Make sure to run this script where it contains raw files below.
filename_corn_noisy = "Corn_noisy.raw";
filename_corn_ori = "Corn_gray.raw";
imgHeight_Corn = 320;
imgWidth_Corn = 320;
imgBytePerPixel_Corn = 1;

noisyData_Corn = readraw_gray(filename_corn_noisy ,imgHeight_Corn, imgWidth_Corn, imgBytePerPixel_Corn);
oriData_Corn = readraw_gray(filename_corn_ori, imgHeight_Corn, imgWidth_Corn, imgBytePerPixel_Corn);


%BM3D algorithm process goes here.
sigma_storage = [10:10:100];
PSNR_storage = zeros(1,size(sigma_storage,2));

%try to find the best sigma that gives the highest PSNR value
for i = 1:size(sigma_storage,2)
    [PSNR, Corn_estimated_BM3D] = BM3D(oriData_Corn, noisyData_Corn, sigma_storage(i));
    PSNR_storage(i) = PSNR;
end

disp([max(PSNR_storage)])
%choose the final sigma that gives the highest PSNR in this problem, it is
%20
Final_sigma = 20;
[PSNR, final_Corn_estimated_BM3D] = BM3D(oriData_Corn, noisyData_Corn, Final_sigma);

%Values of pixels in filter image is within 0~1
%Therefore, We need a post processing to make values of pixels in 0~255
final_Corn_estimated_BM3D = 255.*final_Corn_estimated_BM3D;

%Save filtered image as .raw file
savefilename_Corn = "Corn_BM3D.raw";

temp_Corn = writeraw_gray(final_Corn_estimated_BM3D, savefilename_Corn);



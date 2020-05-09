%Read and load Color image
RoseFile = 'Rose.raw';
imgHeight = 480;
imgWidth = 640;
imgBytePerPixel = 3;

imgRose = readraw_Color(RoseFile,imgHeight,imgWidth,imgBytePerPixel);

%Value test
% imgRoseR = imgRose(:,:,1);
% imgRoseG = imgRose(:,:,2);
% imgRoseB = imgRose(:,:,3);

%Get two floyd masks for each direction
indicator_left2Right = 1;
indicator_right2Left = 2;

left2Right_FloydMask = FloydMasksGenerator(indicator_left2Right);
right2Left_FloydMask = FloydMasksGenerator(indicator_right2Left);



%Extend input color image by one of floyd masks

halfTonedImg_Rose = colorErrorDifusser_MBVQ_V2(imgRose,left2Right_FloydMask,right2Left_FloydMask);

figure(1)
imshow(uint8(halfTonedImg_Rose))

figure(2)
imshow(uint8(imgRose))
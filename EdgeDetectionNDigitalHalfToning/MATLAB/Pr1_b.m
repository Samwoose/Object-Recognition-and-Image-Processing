%Canny Edge Detection

%Dogs.jpg
dogs_Img_Color = imread('Dogs.jpg');
%Need to convert RGB to Gray
dogs_Img_Gray = RGB2Gray(dogs_Img_Color);
%Defind two thresholds 
lowThreshold = 0.1;
highThreshold = 0.3;
thresholdSet = [lowThreshold,highThreshold];

dogs_EdgeMap = edge(dogs_Img_Gray,'Canny',thresholdSet);

%Gallery.jpg
gallery_Img_Color = imread('Gallery.jpg');

%Need to convert RGB to Gray
gallery_Img_Gray = RGB2Gray(gallery_Img_Color);
gallery_EdgeMap = edge(gallery_Img_Gray,'Canny',thresholdSet);

%Save edge maps as raw file
%Scaling Process 0~1 to 0~255
dogs_EdgeMap_scaled = scalerChanger255(dogs_EdgeMap);
gallery_EdgeMap_scaled = scalerChanger255(gallery_EdgeMap);




%inverse value of pixel in binarized image data.
%This is for evaluation processing later in problem 1 (d)
%e.g. 0 -> 255(Background, white), 255 -> 0(Edge, Black)
inversed_dogs_EdgeMap_scaled = valueInverseOperator(dogs_EdgeMap_scaled);
inversed_gallery_EdgeMap_scaled = valueInverseOperator(gallery_EdgeMap_scaled);

%Check the edge image
figure(1)
imshow(inversed_dogs_EdgeMap_scaled)

figure(2)
imshow(inversed_gallery_EdgeMap_scaled)


%Save image data as .raw format
inversed_dogs_EdgeMap_scaled = writeraw_gray(inversed_dogs_EdgeMap_scaled,'Dogs_Edge_Canny.raw');
inversed_gallery_EdgeMap_scaled = writeraw_gray(inversed_gallery_EdgeMap_scaled,'Gallery_Edge_Canny.raw');




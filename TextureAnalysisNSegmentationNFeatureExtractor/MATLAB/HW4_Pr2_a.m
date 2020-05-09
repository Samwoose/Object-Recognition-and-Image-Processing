%% Pr2(a)-1

%Permanently add VLFeat to my MATLAB environment
run('C:\Users\tjdtk\Desktop1\EE 569\HW4\HW4_MATLAB\HW4_Pr2_a\vlfeat-0.9.21-bin.tar\vlfeat-0.9.21-bin\vlfeat-0.9.21\toolbox\vl_setup')
vl_setup demo

%load images
husky3Img = imread('Husky_3.jpg');
husky1Img = imread('Husky_1.jpg');
husky2Img = imread('Husky_2.jpg');
puppy1Img = imread('Puppy_1.jpg');

%Compute features
%Husky3
[featureFrame_Husky3,descriptor_Husky3] = modifiedSIFT_V1(husky3Img);
%Husky1
[featureFrame_Husky1,descriptor_Husky1] = modifiedSIFT_V1(husky1Img);

%Find key-point with the largest scale in Husky3
[maxScale_Husky3,index_Husky3] = indexOfKeypointWithMaxScaleFinder(descriptor_Husky3);


%Find the closest neighbor key point to the key point found above
%Need to transpose keypoint vector(descriptors) 
%e.g. 128x954 -> 954x128 for Husky1 and 128x1090 -> 1090x128 for Husky3
descriptor_Husky3_tr = double(descriptor_Husky3') ;
descriptor_Husky1_tr = double(descriptor_Husky1') ;

%Use nearest neighbor search
indeces = knnsearch(descriptor_Husky1_tr,descriptor_Husky3_tr);

%corresponding index of descriptor_husky1
indexOfClosestKeypoint_Husky1 = indeces(index_Husky3);

%get each key point by index found above for each image.
foundKeypoint_Husky1 = featureFrame_Husky1(:,indexOfClosestKeypoint_Husky1);
foundKeypoint_Husky3 = featureFrame_Husky3(:,index_Husky3);

%Get an orientation of each key point
orientationLocation = 4;
orientation_Husky1 = foundKeypoint_Husky1(orientationLocation,1); %in radian
orientation_Husky3 = foundKeypoint_Husky3(orientationLocation,1); %in radian



%% Pr2(a)-2
%matching points finder
pairsOfMatchingPoints_Husky3NHusky1 = matchingPointFinder(husky3Img,husky1Img);

pairsOfMatchingPoints_Husky3NHusky2 = matchingPointFinder(husky3Img,husky2Img);

pairsOfMatchingPoints_Husky3NPuppy1 = matchingPointFinder(husky3Img,puppy1Img);

pairsOfMatchingPoints_Husky1NPuppy1 = matchingPointFinder(husky1Img,puppy1Img);



%Husky3 and Husky1
dummyOutput1 = matchingSIFTPairsShower(husky3Img,husky1Img,pairsOfMatchingPoints_Husky3NHusky1);

%Husky3 and Husky2
dummyOutput2 = matchingSIFTPairsShower(husky3Img,husky2Img,pairsOfMatchingPoints_Husky3NHusky2);

%Husky3 and Puppy1
dummyOutput3 = matchingSIFTPairsShower(husky3Img,puppy1Img,pairsOfMatchingPoints_Husky3NPuppy1);

%Husky1 and Puppy1
dummyOutput4 = matchingSIFTPairsShower(husky1Img,puppy1Img,pairsOfMatchingPoints_Husky1NPuppy1);


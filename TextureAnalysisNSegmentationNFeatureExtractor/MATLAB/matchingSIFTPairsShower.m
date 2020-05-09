function dummyOutput = matchingSIFTPairsShower(I1,I2,pairsOfMatchingPoints_I1NI2)
%MATCHINGSIFTPAIRSSHOWER Summary of this function goes here
%Show matching SIFT pairs between Image 1(I1), and Image 2(I2)
%e.g. I1: Husky_3 image and I2: Husky_1 image
%   Detailed explanation goes here

dummyOutput = 1;

%Compute feature frame by SIFT for each image
%Image 1
[featureFrame_I1,descriptor_I1] = modifiedSIFT_V1(I1);

%Image 2
[featureFrame_I2,descriptor_I2] = modifiedSIFT_V1(I2);

%Find closest feature frame by descriptor and nearest neighbor search
%Find the closest neighbor key point to the key point found above
%Need to transpose keypoint vector(descriptors) 
%e.g. 128x954 -> 954x128 for Husky1 and 128x1090 -> 1090x128 for Husky3
descriptor_I1_tr = double(descriptor_I1') ;
descriptor_I2_tr = double(descriptor_I2') ;

%Use nearest neighbor search
indeces = knnsearch(descriptor_I2_tr,descriptor_I1_tr);

%Show corresponding feature frames on each image
%Image1
% --------------------------------------------------------------------
%                                                        Load a figure
% --------------------------------------------------------------------

figure(1)
image(I1) ; colormap gray ;
axis equal ; axis off ; axis tight ;
vl_demo_print('sift_basic_0') ;

% --------------------------------------------------------------------
%                                       Convert the to required format
% --------------------------------------------------------------------
I1_gray = single(rgb2gray(I1)) ;

clf ; imagesc(I1_gray)
axis equal ; axis off ; axis tight ;
vl_demo_print('sift_basic_1') ;

hold on ;
h1   = vl_plotframe(featureFrame_I1(:,pairsOfMatchingPoints_I1NI2(1,:))) ; set(h1,'color','k','linewidth',3) ;
h2   = vl_plotframe(featureFrame_I1(:,pairsOfMatchingPoints_I1NI2(1,:))) ; set(h2,'color','y','linewidth',2) ;
hold off ;

%Image2
% --------------------------------------------------------------------
%                                                        Load a figure
% --------------------------------------------------------------------

figure(2)
image(I2) ; colormap gray ;
axis equal ; axis off ; axis tight ;
vl_demo_print('sift_basic_0') ;

% --------------------------------------------------------------------
%                                       Convert the to required format
% --------------------------------------------------------------------
I2_gray = single(rgb2gray(I2)) ;

clf ; imagesc(I2_gray)
axis equal ; axis off ; axis tight ;
vl_demo_print('sift_basic_1') ;

hold on ;
h3   = vl_plotframe(featureFrame_I2(:,pairsOfMatchingPoints_I1NI2(2,:))); set(h3,'color','k','linewidth',3) ;
h4   = vl_plotframe(featureFrame_I2(:,pairsOfMatchingPoints_I1NI2(2,:))) ; set(h4,'color','y','linewidth',2) ;
hold off ;

%%%Partially chosen descriptor in Image 2
perm = randperm(size(featureFrame_I2,2)) ;
numOfFreatureFrames = 50;
sel  = perm(1:numOfFreatureFrames) ;

partialDescriptor_I2 = descriptor_I2(:,sel);
partialDescriptor_I2_tr = double(partialDescriptor_I2');

partialIndeces = knnsearch(partialDescriptor_I2_tr,descriptor_I1_tr);


end


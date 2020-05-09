%Permanently add VLFeat to my MATLAB environment
run('C:\Users\tjdtk\Desktop1\EE 569\HW4\HW4_MATLAB\HW4_Pr2_a\vlfeat-0.9.21-bin.tar\vlfeat-0.9.21-bin\vlfeat-0.9.21\toolbox\vl_setup')
vl_setup demo

%Load Images
%load images
husky3Img = imread('Husky_3.jpg');
husky1Img = imread('Husky_1.jpg');
husky2Img = imread('Husky_2.jpg');
puppy1Img = imread('Puppy_1.jpg');

%plot code words histogram based on Husky_3
orderOfFigure1 = 1;
orderOfFigure2 = 2;
orderOfFigure3 = 3;
orderOfFigure4 = 4;
orderOfFigure5 = 5;
orderOfFigure6 = 6;
orderOfFigure7 = 7;
orderOfFigure8 = 8;

%inputImg1
[featureFrame_husky3Img,descriptor_husky3Img] = modifiedSIFT_V1(husky3Img);

%Compute 8 centroids for each image by K means clustering
%need to transpose descriptors variable matrix to make it a proper
%parameter form
numOfClusters = 8;
%inputImg1
descriptor_husky3Img_tr = double(descriptor_husky3Img');
[labels_inputImg1,Centroids_inputImg1] = kmeans(descriptor_husky3Img_tr,numOfClusters);

%Need to use the same centroids and labels of input image(Husky_3) for all
%other images
 dummyVal1 = codeWordsHistogramGenerator(husky3Img, husky1Img,labels_inputImg1,Centroids_inputImg1,orderOfFigure1,orderOfFigure2);
 
 dummyVal2 = codeWordsHistogramGenerator(husky3Img, husky2Img,labels_inputImg1,Centroids_inputImg1,orderOfFigure3,orderOfFigure4);
 
 dummyVal3 = codeWordsHistogramGenerator(husky3Img, puppy1Img,labels_inputImg1,Centroids_inputImg1,orderOfFigure5,orderOfFigure6);
 
 
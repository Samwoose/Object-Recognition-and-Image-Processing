%add paths for train and test data set
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW4\HW4_MATLAB\HW4_Pr1_a\dataSets\train')
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW4\HW4_MATLAB\HW4_Pr1_a\dataSets\test')

%load images all images have same dimension

imgHeight = 128;
imgWidth = 128;
imgBytePerPixel = 1;

%train data set
%blanket
fileName_blanket1 = "blanket1.raw";
fileName_blanket2 = "blanket2.raw";
fileName_blanket3 = "blanket3.raw";
fileName_blanket4 = "blanket4.raw";
fileName_blanket5 = "blanket5.raw";
fileName_blanket6 = "blanket6.raw";
fileName_blanket7 = "blanket7.raw";
fileName_blanket8 = "blanket8.raw";
fileName_blanket9 = "blanket9.raw";

%brick
fileName_brick1 = "brick1.raw";
fileName_brick2 = "brick2.raw";
fileName_brick3 = "brick3.raw";
fileName_brick4 = "brick4.raw";
fileName_brick5 = "brick5.raw";
fileName_brick6 = "brick6.raw";
fileName_brick7 = "brick7.raw";
fileName_brick8 = "brick8.raw";
fileName_brick9 = "brick9.raw";

%grass
fileName_grass1 = "grass1.raw";
fileName_grass2 = "grass2.raw";
fileName_grass3 = "grass3.raw";
fileName_grass4 = "grass4.raw";
fileName_grass5 = "grass5.raw";
fileName_grass6 = "grass6.raw";
fileName_grass7 = "grass7.raw";
fileName_grass8 = "grass8.raw";
fileName_grass9 = "grass9.raw";


%rice
fileName_rice1 = "rice1.raw";
fileName_rice2 = "rice2.raw";
fileName_rice3 = "rice3.raw";
fileName_rice4 = "rice4.raw";
fileName_rice5 = "rice5.raw";
fileName_rice6 = "rice6.raw";
fileName_rice7 = "rice7.raw";
fileName_rice8 = "rice8.raw";
fileName_rice9 = "rice9.raw";

%test data set
fileName_test1 = "1.raw";
fileName_test2 = "2.raw";
fileName_test3 = "3.raw";
fileName_test4 = "4.raw";
fileName_test5 = "5.raw";
fileName_test6 = "6.raw";
fileName_test7 = "7.raw";
fileName_test8 = "8.raw";
fileName_test9 = "9.raw";
fileName_test10 = "10.raw";
fileName_test11 = "11.raw";
fileName_test12 = "12.raw";

%train data set
img_blanket1 = readraw_gray(fileName_blanket1,imgWidth,imgHeight,imgBytePerPixel);
img_blanket2 = readraw_gray(fileName_blanket2,imgWidth,imgHeight,imgBytePerPixel);
img_blanket3 = readraw_gray(fileName_blanket3,imgWidth,imgHeight,imgBytePerPixel);
img_blanket4 = readraw_gray(fileName_blanket4,imgWidth,imgHeight,imgBytePerPixel);
img_blanket5 = readraw_gray(fileName_blanket5,imgWidth,imgHeight,imgBytePerPixel);
img_blanket6 = readraw_gray(fileName_blanket6,imgWidth,imgHeight,imgBytePerPixel);
img_blanket7 = readraw_gray(fileName_blanket7,imgWidth,imgHeight,imgBytePerPixel);
img_blanket8 = readraw_gray(fileName_blanket8,imgWidth,imgHeight,imgBytePerPixel);
img_blanket9 = readraw_gray(fileName_blanket9,imgWidth,imgHeight,imgBytePerPixel);

img_brick1 = readraw_gray(fileName_blanket1,imgWidth,imgHeight,imgBytePerPixel);
img_brick2 = readraw_gray(fileName_blanket2,imgWidth,imgHeight,imgBytePerPixel);
img_brick3 = readraw_gray(fileName_blanket3,imgWidth,imgHeight,imgBytePerPixel);
img_brick4 = readraw_gray(fileName_blanket4,imgWidth,imgHeight,imgBytePerPixel);
img_brick5 = readraw_gray(fileName_blanket5,imgWidth,imgHeight,imgBytePerPixel);
img_brick6 = readraw_gray(fileName_blanket6,imgWidth,imgHeight,imgBytePerPixel);
img_brick7 = readraw_gray(fileName_blanket7,imgWidth,imgHeight,imgBytePerPixel);
img_brick8 = readraw_gray(fileName_blanket8,imgWidth,imgHeight,imgBytePerPixel);
img_brick9 = readraw_gray(fileName_blanket9,imgWidth,imgHeight,imgBytePerPixel);

img_grass1 = readraw_gray(fileName_grass1,imgWidth,imgHeight,imgBytePerPixel);
img_grass2 = readraw_gray(fileName_grass2,imgWidth,imgHeight,imgBytePerPixel);
img_grass3 = readraw_gray(fileName_grass3,imgWidth,imgHeight,imgBytePerPixel);
img_grass4 = readraw_gray(fileName_grass4,imgWidth,imgHeight,imgBytePerPixel);
img_grass5 = readraw_gray(fileName_grass5,imgWidth,imgHeight,imgBytePerPixel);
img_grass6 = readraw_gray(fileName_grass6,imgWidth,imgHeight,imgBytePerPixel);
img_grass7 = readraw_gray(fileName_grass7,imgWidth,imgHeight,imgBytePerPixel);
img_grass8 = readraw_gray(fileName_grass8,imgWidth,imgHeight,imgBytePerPixel);
img_grass9 = readraw_gray(fileName_grass9,imgWidth,imgHeight,imgBytePerPixel);

img_rice1 = readraw_gray(fileName_rice1,imgWidth,imgHeight,imgBytePerPixel);
img_rice2 = readraw_gray(fileName_rice2,imgWidth,imgHeight,imgBytePerPixel);
img_rice3 = readraw_gray(fileName_rice3,imgWidth,imgHeight,imgBytePerPixel);
img_rice4 = readraw_gray(fileName_rice4,imgWidth,imgHeight,imgBytePerPixel);
img_rice5 = readraw_gray(fileName_rice5,imgWidth,imgHeight,imgBytePerPixel);
img_rice6 = readraw_gray(fileName_rice6,imgWidth,imgHeight,imgBytePerPixel);
img_rice7 = readraw_gray(fileName_rice7,imgWidth,imgHeight,imgBytePerPixel);
img_rice8 = readraw_gray(fileName_rice8,imgWidth,imgHeight,imgBytePerPixel);
img_rice9 = readraw_gray(fileName_rice9,imgWidth,imgHeight,imgBytePerPixel);

%test data set
img_test1 = readraw_gray(fileName_test1,imgWidth,imgHeight,imgBytePerPixel);
img_test2 = readraw_gray(fileName_test2,imgWidth,imgHeight,imgBytePerPixel);
img_test3 = readraw_gray(fileName_test3,imgWidth,imgHeight,imgBytePerPixel);
img_test4 = readraw_gray(fileName_test4,imgWidth,imgHeight,imgBytePerPixel);
img_test5 = readraw_gray(fileName_test5,imgWidth,imgHeight,imgBytePerPixel);
img_test6 = readraw_gray(fileName_test6,imgWidth,imgHeight,imgBytePerPixel);
img_test7 = readraw_gray(fileName_test7,imgWidth,imgHeight,imgBytePerPixel);
img_test8 = readraw_gray(fileName_test8,imgWidth,imgHeight,imgBytePerPixel);
img_test9 = readraw_gray(fileName_test9,imgWidth,imgHeight,imgBytePerPixel);
img_test10 = readraw_gray(fileName_test10,imgWidth,imgHeight,imgBytePerPixel);
img_test11 = readraw_gray(fileName_test11,imgWidth,imgHeight,imgBytePerPixel);
img_test12 = readraw_gray(fileName_test12,imgWidth,imgHeight,imgBytePerPixel);


%Generate Laws filter bank
filterBank = filterBankGenerator();

%% Pr1 (a)-1 Pixelwise Feature extraction
%train data set
pixelwiseFeatures_blanket1 = featureExtractor(img_blanket1,filterBank);
pixelwiseFeatures_blanket2 = featureExtractor(img_blanket2,filterBank);
pixelwiseFeatures_blanket3 = featureExtractor(img_blanket3,filterBank);
pixelwiseFeatures_blanket4 = featureExtractor(img_blanket4,filterBank);
pixelwiseFeatures_blanket5 = featureExtractor(img_blanket5,filterBank);
pixelwiseFeatures_blanket6 = featureExtractor(img_blanket6,filterBank);
pixelwiseFeatures_blanket7 = featureExtractor(img_blanket7,filterBank);
pixelwiseFeatures_blanket8 = featureExtractor(img_blanket8,filterBank);
pixelwiseFeatures_blanket9 = featureExtractor(img_blanket9,filterBank);

pixelwiseFeatures_brick1 = featureExtractor(img_brick1,filterBank);
pixelwiseFeatures_brick2 = featureExtractor(img_brick2,filterBank);
pixelwiseFeatures_brick3 = featureExtractor(img_brick3,filterBank);
pixelwiseFeatures_brick4 = featureExtractor(img_brick4,filterBank);
pixelwiseFeatures_brick5 = featureExtractor(img_brick5,filterBank);
pixelwiseFeatures_brick6 = featureExtractor(img_brick6,filterBank);
pixelwiseFeatures_brick7 = featureExtractor(img_brick7,filterBank);
pixelwiseFeatures_brick8 = featureExtractor(img_brick8,filterBank);
pixelwiseFeatures_brick9 = featureExtractor(img_brick9,filterBank);


pixelwiseFeatures_grass1 = featureExtractor(img_grass1,filterBank);
pixelwiseFeatures_grass2 = featureExtractor(img_grass2,filterBank);
pixelwiseFeatures_grass3 = featureExtractor(img_grass3,filterBank);
pixelwiseFeatures_grass4 = featureExtractor(img_grass4,filterBank);
pixelwiseFeatures_grass5 = featureExtractor(img_grass5,filterBank);
pixelwiseFeatures_grass6 = featureExtractor(img_grass6,filterBank);
pixelwiseFeatures_grass7 = featureExtractor(img_grass7,filterBank);
pixelwiseFeatures_grass8 = featureExtractor(img_grass8,filterBank);
pixelwiseFeatures_grass9 = featureExtractor(img_grass9,filterBank);


pixelwiseFeatures_rice1 = featureExtractor(img_rice1,filterBank);
pixelwiseFeatures_rice2 = featureExtractor(img_rice2,filterBank);
pixelwiseFeatures_rice3 = featureExtractor(img_rice3,filterBank);
pixelwiseFeatures_rice4 = featureExtractor(img_rice4,filterBank);
pixelwiseFeatures_rice5 = featureExtractor(img_rice5,filterBank);
pixelwiseFeatures_rice6 = featureExtractor(img_rice6,filterBank);
pixelwiseFeatures_rice7 = featureExtractor(img_rice7,filterBank);
pixelwiseFeatures_rice8 = featureExtractor(img_rice8,filterBank);
pixelwiseFeatures_rice9 = featureExtractor(img_rice9,filterBank);

%test data set
pixelwiseFeatures_test1 = featureExtractor(img_test1,filterBank);
pixelwiseFeatures_test2 = featureExtractor(img_test2,filterBank);
pixelwiseFeatures_test3 = featureExtractor(img_test3,filterBank);
pixelwiseFeatures_test4 = featureExtractor(img_test4,filterBank);
pixelwiseFeatures_test5 = featureExtractor(img_test5,filterBank);
pixelwiseFeatures_test6 = featureExtractor(img_test6,filterBank);
pixelwiseFeatures_test7 = featureExtractor(img_test7,filterBank);
pixelwiseFeatures_test8 = featureExtractor(img_test8,filterBank);
pixelwiseFeatures_test9 = featureExtractor(img_test9,filterBank);
pixelwiseFeatures_test10 = featureExtractor(img_test10,filterBank);
pixelwiseFeatures_test11 = featureExtractor(img_test11,filterBank);
pixelwiseFeatures_test12 = featureExtractor(img_test12,filterBank);

%% Feature Averaging
%train data set
averagedFeatureVector_blanket1_15D = featureValAverager(pixelwiseFeatures_blanket1);
averagedFeatureVector_blanket2_15D = featureValAverager(pixelwiseFeatures_blanket2);
averagedFeatureVector_blanket3_15D = featureValAverager(pixelwiseFeatures_blanket3);
averagedFeatureVector_blanket4_15D = featureValAverager(pixelwiseFeatures_blanket4);
averagedFeatureVector_blanket5_15D = featureValAverager(pixelwiseFeatures_blanket5);
averagedFeatureVector_blanket6_15D = featureValAverager(pixelwiseFeatures_blanket6);
averagedFeatureVector_blanket7_15D = featureValAverager(pixelwiseFeatures_blanket7);
averagedFeatureVector_blanket8_15D = featureValAverager(pixelwiseFeatures_blanket8);
averagedFeatureVector_blanket9_15D = featureValAverager(pixelwiseFeatures_blanket9);

averagedFeatureVector_brick1_15D = featureValAverager(pixelwiseFeatures_brick1);
averagedFeatureVector_brick2_15D = featureValAverager(pixelwiseFeatures_brick2);
averagedFeatureVector_brick3_15D = featureValAverager(pixelwiseFeatures_brick3);
averagedFeatureVector_brick4_15D = featureValAverager(pixelwiseFeatures_brick4);
averagedFeatureVector_brick5_15D = featureValAverager(pixelwiseFeatures_brick5);
averagedFeatureVector_brick6_15D = featureValAverager(pixelwiseFeatures_brick6);
averagedFeatureVector_brick7_15D = featureValAverager(pixelwiseFeatures_brick7);
averagedFeatureVector_brick8_15D = featureValAverager(pixelwiseFeatures_brick8);
averagedFeatureVector_brick9_15D = featureValAverager(pixelwiseFeatures_brick9);

averagedFeatureVector_grass1_15D = featureValAverager(pixelwiseFeatures_grass1);
averagedFeatureVector_grass2_15D = featureValAverager(pixelwiseFeatures_grass2);
averagedFeatureVector_grass3_15D = featureValAverager(pixelwiseFeatures_grass3);
averagedFeatureVector_grass4_15D = featureValAverager(pixelwiseFeatures_grass4);
averagedFeatureVector_grass5_15D = featureValAverager(pixelwiseFeatures_grass5);
averagedFeatureVector_grass6_15D = featureValAverager(pixelwiseFeatures_grass6);
averagedFeatureVector_grass7_15D = featureValAverager(pixelwiseFeatures_grass7);
averagedFeatureVector_grass8_15D = featureValAverager(pixelwiseFeatures_grass8);
averagedFeatureVector_grass9_15D = featureValAverager(pixelwiseFeatures_grass9);

averagedFeatureVector_rice1_15D = featureValAverager(pixelwiseFeatures_rice1);
averagedFeatureVector_rice2_15D = featureValAverager(pixelwiseFeatures_rice2);
averagedFeatureVector_rice3_15D = featureValAverager(pixelwiseFeatures_rice3);
averagedFeatureVector_rice4_15D = featureValAverager(pixelwiseFeatures_rice4);
averagedFeatureVector_rice5_15D = featureValAverager(pixelwiseFeatures_rice5);
averagedFeatureVector_rice6_15D = featureValAverager(pixelwiseFeatures_rice6);
averagedFeatureVector_rice7_15D = featureValAverager(pixelwiseFeatures_rice7);
averagedFeatureVector_rice8_15D = featureValAverager(pixelwiseFeatures_rice8);
averagedFeatureVector_rice9_15D = featureValAverager(pixelwiseFeatures_rice9);

%test data set
averagedFeatureVector_test1_15D = featureValAverager(pixelwiseFeatures_test1);
averagedFeatureVector_test2_15D = featureValAverager(pixelwiseFeatures_test2);
averagedFeatureVector_test3_15D = featureValAverager(pixelwiseFeatures_test3);
averagedFeatureVector_test4_15D = featureValAverager(pixelwiseFeatures_test4);
averagedFeatureVector_test5_15D = featureValAverager(pixelwiseFeatures_test5);
averagedFeatureVector_test6_15D = featureValAverager(pixelwiseFeatures_test6);
averagedFeatureVector_test7_15D = featureValAverager(pixelwiseFeatures_test7);
averagedFeatureVector_test8_15D = featureValAverager(pixelwiseFeatures_test8);
averagedFeatureVector_test9_15D = featureValAverager(pixelwiseFeatures_test9);
averagedFeatureVector_test10_15D = featureValAverager(pixelwiseFeatures_test10);
averagedFeatureVector_test11_15D = featureValAverager(pixelwiseFeatures_test11);
averagedFeatureVector_test12_15D = featureValAverager(pixelwiseFeatures_test12);


%% Identify feature dimension that has strongest and weakest discriminant power.

%% Feature Reduction
%Construct X_train_15D, and X_test_15D, row: observation(img, but label is
%not included) col:15 feature values

%X_train_15D
X_train_15D =[averagedFeatureVector_blanket1_15D;
    averagedFeatureVector_blanket2_15D;
    averagedFeatureVector_blanket3_15D;
    averagedFeatureVector_blanket4_15D;
    averagedFeatureVector_blanket5_15D;
    averagedFeatureVector_blanket6_15D;
    averagedFeatureVector_blanket7_15D;
    averagedFeatureVector_blanket8_15D;
    averagedFeatureVector_blanket9_15D;
    averagedFeatureVector_brick1_15D;
    averagedFeatureVector_brick2_15D;
    averagedFeatureVector_brick3_15D;
    averagedFeatureVector_brick4_15D;
    averagedFeatureVector_brick5_15D;
    averagedFeatureVector_brick6_15D;
    averagedFeatureVector_brick7_15D;
    averagedFeatureVector_brick8_15D;
    averagedFeatureVector_brick9_15D;
    averagedFeatureVector_grass1_15D;
    averagedFeatureVector_grass2_15D;
    averagedFeatureVector_grass3_15D;
    averagedFeatureVector_grass4_15D;
    averagedFeatureVector_grass5_15D;
    averagedFeatureVector_grass6_15D;
    averagedFeatureVector_grass7_15D;
    averagedFeatureVector_grass8_15D;
    averagedFeatureVector_grass9_15D;
    averagedFeatureVector_rice1_15D;
    averagedFeatureVector_rice2_15D;
    averagedFeatureVector_rice3_15D;
    averagedFeatureVector_rice4_15D;
    averagedFeatureVector_rice5_15D;
    averagedFeatureVector_rice6_15D;
    averagedFeatureVector_rice7_15D;
    averagedFeatureVector_rice8_15D;
    averagedFeatureVector_rice9_15D ]; 

X_test_15D = [averagedFeatureVector_test1_15D;
    averagedFeatureVector_test2_15D;
    averagedFeatureVector_test3_15D;
    averagedFeatureVector_test4_15D;
    averagedFeatureVector_test5_15D;
    averagedFeatureVector_test6_15D;
    averagedFeatureVector_test7_15D;
    averagedFeatureVector_test8_15D;
    averagedFeatureVector_test9_15D;
    averagedFeatureVector_test10_15D;
    averagedFeatureVector_test11_15D;
    averagedFeatureVector_test12_15D] ;
%maybe this one?
%[coeff,score,latent,tsquared,explained] = pca(X(:,3:15));

[coeff_train, newdata_train, latent_train, tsquared_train, explained_train] = pca(X_train_15D);
[coeff_test, newdata_test, latent_test, tsquared_test, explained_test] = pca(X_test_15D);


%plotting
figure(1)
scatter3(newdata_train(:,1),newdata_train(:,2),newdata_train(:,3))
axis equal
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')

figure(2)
scatter3(newdata_test(:,1),newdata_test(:,2),newdata_test(:,3))
axis equal
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')


%Construct reduced X_test_3D and X_train_3D for saving purpose
reducedX_train_3D = [newdata_train(:,1),newdata_train(:,2),newdata_train(:,3)];
reducedX_test_3D = [newdata_test(:,1),newdata_test(:,2),newdata_test(:,3)];
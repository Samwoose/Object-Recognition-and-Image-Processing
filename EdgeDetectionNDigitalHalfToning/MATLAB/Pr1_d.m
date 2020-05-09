%%additionally added part
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW2\HW2_MATLAB\Pr1_c\piotr_toolbox\toolbox\matlab')
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW2\HW2_MATLAB\Pr1_c\piotr_toolbox\toolbox\channels')
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW2\HW2_MATLAB\Pr1_c\piotr_toolbox\toolbox\images')
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW2\HW2_MATLAB\Pr1_c\edges-master')
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW2\HW2_MATLAB\Pr1_c')


%% Pr1_d

%It could take several mins
%Images and evaluation related to Gallery.raw
groundTruthFile_Gallery = 'Gallery_GT.mat';
imgHeight_Gallery = 321;
imgWidth_Gallery = 481;
imgBytePerPixel_Gallery = 1;%gray scale

%Load resulting images and change scale from 0~255 to 0~1
%Sobel
rawFile_Gallery_Sobel = 'Gallery_MagnitudeEdge.raw';
probabilityEdgeMap_Gallery_Sobel_255 = readraw_gray(rawFile_Gallery_Sobel,imgHeight_Gallery,imgWidth_Gallery,imgBytePerPixel_Gallery);
probabilityEdgeMap_Gallery_Sobel_1 = scalerChangerZero2One(probabilityEdgeMap_Gallery_Sobel_255);

%Canny
rawFile_Gallery_Canny = 'Gallery_Edge_Canny.raw';
binaryEdgeMap_Gallery_Canny_255 = readraw_gray(rawFile_Gallery_Canny,imgHeight_Gallery,imgWidth_Gallery,imgBytePerPixel_Gallery);
binaryEdgeMap_Gallery_Canny_1 = scalerChangerZero2One(binaryEdgeMap_Gallery_Canny_255);

%Structured Edge
rawFile_Gallery_SE = 'Gallery_probability_SE.raw' ;
probabilityEdgeMap_Gallery_SE_1 = readraw_gray(rawFile_Gallery_SE,imgHeight_Gallery,imgWidth_Gallery,imgBytePerPixel_Gallery);

%Calculate recall and precision with different thresholds for each ground truth 
%Ground Truth 1
orderOfGroundTruth1_Gallery = 1; 
%Sobel
[thrs1_sobel_Gallery,cntR1_sobel_Gallery,sumR1_sobel_Gallery,cntP1_sobel_Gallery,sumP1_sobel_Gallery] = edgesEvalImg(probabilityEdgeMap_Gallery_Sobel_1,groundTruthFile_Gallery,orderOfGroundTruth1_Gallery);
precision1_sobel_Gallery = elementwiseDivider(cntP1_sobel_Gallery,sumP1_sobel_Gallery);
recall1_sobel_Gallery = elementwiseDivider(cntR1_sobel_Gallery,sumR1_sobel_Gallery);

%Canny
[thrs1_canny_Gallery,cntR1_canny_Gallery,sumR1_canny_Gallery,cntP1_canny_Gallery,sumP1_canny_Gallery] = edgesEvalImg(binaryEdgeMap_Gallery_Canny_1,groundTruthFile_Gallery,orderOfGroundTruth1_Gallery);
precision1_canny_Gallery = elementwiseDivider(cntP1_canny_Gallery,sumP1_canny_Gallery);
recall1_canny_Gallery = elementwiseDivider(cntR1_canny_Gallery,sumR1_canny_Gallery);


%Structured Edge
[thrs1_SE_Gallery,cntR1_SE_Gallery,sumR1_SE_Gallery,cntP1_SE_Gallery,sumP1_SE_Gallery] = edgesEvalImg(probabilityEdgeMap_Gallery_SE_1,groundTruthFile_Gallery,orderOfGroundTruth1_Gallery);
precision1_SE_Gallery = elementwiseDivider(cntP1_SE_Gallery,sumP1_SE_Gallery);
recall1_SE_Gallery = elementwiseDivider(cntR1_SE_Gallery,sumR1_SE_Gallery);

%Ground Truth 2
orderOfGroundTruth2_Gallery = 2; 
%Sobel
[thrs2_sobel_Gallery,cntR2_sobel_Gallery,sumR2_sobel_Gallery,cntP2_sobel_Gallery,sumP2_sobel_Gallery] = edgesEvalImg(probabilityEdgeMap_Gallery_Sobel_1,groundTruthFile_Gallery,orderOfGroundTruth2_Gallery);
precision2_sobel_Gallery = elementwiseDivider(cntP2_sobel_Gallery,sumP2_sobel_Gallery);
recall2_sobel_Gallery = elementwiseDivider(cntR2_sobel_Gallery,sumR2_sobel_Gallery);

%Canny
[thrs2_canny_Gallery,cntR2_canny_Gallery,sumR2_canny_Gallery,cntP2_canny_Gallery,sumP2_canny_Gallery] = edgesEvalImg(binaryEdgeMap_Gallery_Canny_1,groundTruthFile_Gallery,orderOfGroundTruth2_Gallery);
precision2_canny_Gallery = elementwiseDivider(cntP2_canny_Gallery,sumP2_canny_Gallery);
recall2_canny_Gallery = elementwiseDivider(cntR2_canny_Gallery,sumR2_canny_Gallery);

%Structured Edge
[thrs2_SE_Gallery,cntR2_SE_Gallery,sumR2_SE_Gallery,cntP2_SE_Gallery,sumP2_SE_Gallery] = edgesEvalImg(probabilityEdgeMap_Gallery_SE_1,groundTruthFile_Gallery,orderOfGroundTruth2_Gallery);
precision2_SE_Gallery = elementwiseDivider(cntP2_SE_Gallery,sumP2_SE_Gallery);
recall2_SE_Gallery = elementwiseDivider(cntR2_SE_Gallery,sumR2_SE_Gallery);

%Ground Truth 3
orderOfGroundTruth3_Gallery = 3; 
%Sobel
[thrs3_sobel_Gallery,cntR3_sobel_Gallery,sumR3_sobel_Gallery,cntP3_sobel_Gallery,sumP3_sobel_Gallery] = edgesEvalImg(probabilityEdgeMap_Gallery_Sobel_1,groundTruthFile_Gallery,orderOfGroundTruth3_Gallery);
precision3_sobel_Gallery = elementwiseDivider(cntP3_sobel_Gallery,sumP3_sobel_Gallery);
recall3_sobel_Gallery = elementwiseDivider(cntR3_sobel_Gallery,sumR3_sobel_Gallery);
 
%Canny
[thrs3_canny_Gallery,cntR3_canny_Gallery,sumR3_canny_Gallery,cntP3_canny_Gallery,sumP3_canny_Gallery] = edgesEvalImg(binaryEdgeMap_Gallery_Canny_1,groundTruthFile_Gallery,orderOfGroundTruth3_Gallery);
precision3_canny_Gallery = elementwiseDivider(cntP3_canny_Gallery,sumP3_canny_Gallery);
recall3_canny_Gallery = elementwiseDivider(cntR3_canny_Gallery,sumR3_canny_Gallery);
 
%Structured Edge
[thrs3_SE_Gallery,cntR3_SE_Gallery,sumR3_SE_Gallery,cntP3_SE_Gallery,sumP3_SE_Gallery] = edgesEvalImg(probabilityEdgeMap_Gallery_SE_1,groundTruthFile_Gallery,orderOfGroundTruth3_Gallery);
precision3_SE_Gallery = elementwiseDivider(cntP3_SE_Gallery,sumP3_SE_Gallery);
recall3_SE_Gallery = elementwiseDivider(cntR3_SE_Gallery,sumR3_SE_Gallery);

%Ground Truth 4
orderOfGroundTruth4_Gallery = 4; 
%Sobel
[thrs4_sobel_Gallery,cntR4_sobel_Gallery,sumR4_sobel_Gallery,cntP4_sobel_Gallery,sumP4_sobel_Gallery] = edgesEvalImg(probabilityEdgeMap_Gallery_Sobel_1,groundTruthFile_Gallery,orderOfGroundTruth4_Gallery);
precision4_sobel_Gallery = elementwiseDivider(cntP4_sobel_Gallery,sumP4_sobel_Gallery);
recall4_sobel_Gallery = elementwiseDivider(cntR4_sobel_Gallery,sumR4_sobel_Gallery);
%Canny
[thrs4_canny_Gallery,cntR4_canny_Gallery,sumR4_canny_Gallery,cntP4_canny_Gallery,sumP4_canny_Gallery] = edgesEvalImg(binaryEdgeMap_Gallery_Canny_1,groundTruthFile_Gallery,orderOfGroundTruth4_Gallery);
precision4_canny_Gallery = elementwiseDivider(cntP4_canny_Gallery,sumP4_canny_Gallery);
recall4_canny_Gallery = elementwiseDivider(cntR4_canny_Gallery,sumR4_canny_Gallery);
 
%Structured Edge
[thrs4_SE_Gallery,cntR4_SE_Gallery,sumR4_SE_Gallery,cntP4_SE_Gallery,sumP4_SE_Gallery] = edgesEvalImg(probabilityEdgeMap_Gallery_SE_1,groundTruthFile_Gallery,orderOfGroundTruth4_Gallery);
precision4_SE_Gallery = elementwiseDivider(cntP4_SE_Gallery,sumP4_SE_Gallery);
recall4_SE_Gallery = elementwiseDivider(cntR4_SE_Gallery,sumR4_SE_Gallery);

%Ground Truth 5
orderOfGroundTruth5_Gallery = 5; 
%Sobel
[thrs5_sobel_Gallery,cntR5_sobel_Gallery,sumR5_sobel_Gallery,cntP5_sobel_Gallery,sumP5_sobel_Gallery] = edgesEvalImg(probabilityEdgeMap_Gallery_Sobel_1,groundTruthFile_Gallery,orderOfGroundTruth5_Gallery);
precision5_sobel_Gallery = elementwiseDivider(cntP5_sobel_Gallery,sumP5_sobel_Gallery);
recall5_sobel_Gallery = elementwiseDivider(cntR5_sobel_Gallery,sumR5_sobel_Gallery);
%Canny
[thrs5_canny_Gallery,cntR5_canny_Gallery,sumR5_canny_Gallery,cntP5_canny_Gallery,sumP5_canny_Gallery] = edgesEvalImg(binaryEdgeMap_Gallery_Canny_1,groundTruthFile_Gallery,orderOfGroundTruth5_Gallery);
precision5_canny_Gallery = elementwiseDivider(cntP5_canny_Gallery,sumP5_canny_Gallery);
recall5_canny_Gallery = elementwiseDivider(cntR5_canny_Gallery,sumR5_canny_Gallery);
 
%Structured Edge
[thrs5_SE_Gallery,cntR5_SE_Gallery,sumR5_SE_Gallery,cntP5_SE_Gallery,sumP5_SE_Gallery] = edgesEvalImg(probabilityEdgeMap_Gallery_SE_1,groundTruthFile_Gallery,orderOfGroundTruth5_Gallery);
precision5_SE_Gallery = elementwiseDivider(cntP5_SE_Gallery,sumP5_SE_Gallery);
recall5_SE_Gallery = elementwiseDivider(cntR5_SE_Gallery,sumR5_SE_Gallery);


%Calculater Mean Precision and Mean Recall over 5 ground truths
numOfGroundtruth_Gallery = 5;
 
%Sobel
meanPrecision_sobel_Gallery =elementwiseDividerByInteger( elementwiseAdditioner(precision5_sobel_Gallery,elementwiseAdditioner(precision4_sobel_Gallery, elementwiseAdditioner(precision3_sobel_Gallery, elementwiseAdditioner(precision1_sobel_Gallery,precision2_sobel_Gallery)))) ,numOfGroundtruth_Gallery) ;
meanRecall_sobel_Gallery =elementwiseDividerByInteger( elementwiseAdditioner(recall5_sobel_Gallery,elementwiseAdditioner(recall4_sobel_Gallery, elementwiseAdditioner(recall3_sobel_Gallery, elementwiseAdditioner(recall1_sobel_Gallery,recall2_sobel_Gallery)))) ,numOfGroundtruth_Gallery) ;

%Canny
meanPrecision_canny_Gallery =elementwiseDividerByInteger( elementwiseAdditioner(precision5_canny_Gallery,elementwiseAdditioner(precision4_canny_Gallery, elementwiseAdditioner(precision3_canny_Gallery, elementwiseAdditioner(precision1_canny_Gallery,precision2_canny_Gallery)))) ,numOfGroundtruth_Gallery) ;
meanRecall_canny_Gallery =elementwiseDividerByInteger( elementwiseAdditioner(recall5_canny_Gallery,elementwiseAdditioner(recall4_canny_Gallery, elementwiseAdditioner(recall3_canny_Gallery, elementwiseAdditioner(recall1_canny_Gallery,recall2_canny_Gallery)))) ,numOfGroundtruth_Gallery) ;
 
%Structured Edge
meanPrecision_SE_Gallery =elementwiseDividerByInteger( elementwiseAdditioner(precision5_SE_Gallery,elementwiseAdditioner(precision4_SE_Gallery, elementwiseAdditioner(precision3_SE_Gallery, elementwiseAdditioner(precision1_SE_Gallery,precision2_SE_Gallery)))) ,numOfGroundtruth_Gallery) ;
meanRecall_SE_Gallery =elementwiseDividerByInteger( elementwiseAdditioner(recall5_SE_Gallery,elementwiseAdditioner(recall4_SE_Gallery, elementwiseAdditioner(recall3_SE_Gallery, elementwiseAdditioner(recall1_SE_Gallery,recall2_SE_Gallery)))) ,numOfGroundtruth_Gallery) ;
 


%Calculate F measure
%Sobel
meanFScore_sobel_Gallery = FScoreMatrixCalculator(meanPrecision_sobel_Gallery,meanRecall_sobel_Gallery);
%Canny
meanFScore_canny_Gallery = FScoreMatrixCalculator(meanPrecision_canny_Gallery,meanRecall_canny_Gallery);
 
%Structured Edge
meanFScore_SE_Gallery = FScoreMatrixCalculator(meanPrecision_SE_Gallery,meanRecall_SE_Gallery);
 
 
%Find max F score with respect to threshold
%Sobel
[max_FScore_sobel_Gallery, location_sobel_Gallery] = MaxInMatrixFinder(meanFScore_sobel_Gallery);
threshold_maxFscore_sobel_Gallery = thrs1_sobel_Gallery(location_sobel_Gallery);
%Canny
[max_FScore_canny_Gallery, location_canny_Gallery] = MaxInMatrixFinder(meanFScore_canny_Gallery);
threshold_maxFscore_canny_Gallery = thrs1_canny_Gallery(location_canny_Gallery);
 
%Structured Edge
[max_FScore_SE_Gallery, location_SE_Gallery] = MaxInMatrixFinder(meanFScore_SE_Gallery);
threshold_maxFscore_SE_Gallery = thrs1_SE_Gallery(location_SE_Gallery);





%Images and evaluation related to Dogs.raw
groundTruthFile = 'Dogs_GT.mat';
imgHeight_Dogs = 321;
imgWidth_Dogs = 481;
imgBytePerPixel_Dogs = 1;%gray scale

%Load resulting images and change scale from 0~255 to 0~1
%Sobel
rawFile_Dogs_Sobel = 'Dogs_MagnitudeEdge.raw';
probabilityEdgeMap_Dog_Sobel_255 = readraw_gray(rawFile_Dogs_Sobel,imgHeight_Dogs,imgWidth_Dogs,imgBytePerPixel_Dogs);
probabilityEdgeMap_Dog_Sobel_1 = scalerChangerZero2One(probabilityEdgeMap_Dog_Sobel_255);

%Canny
rawFile_Dogs_Canny = 'Dogs_Edge_Canny.raw';
binaryEdgeMap_Dogs_Canny_255 = readraw_gray(rawFile_Dogs_Canny,imgHeight_Dogs,imgWidth_Dogs,imgBytePerPixel_Dogs);
binaryEdgeMap_Dogs_Canny_1 = scalerChangerZero2One(binaryEdgeMap_Dogs_Canny_255);


%Structured Edge
rawFile_Dogs_SE = 'Dogs_probability_SE.raw' ;
probabilityEdgeMap_Dog_SE_1 = readraw_gray(rawFile_Dogs_SE,imgHeight_Dogs,imgWidth_Dogs,imgBytePerPixel_Dogs);


%Calculate recall and precision with different thresholds for each ground truth 
%Ground Truth 1
orderOfGroundTruth1 = 1; 
%Sobel
[thrs1_sobel,cntR1_sobel,sumR1_sobel,cntP1_sobel,sumP1_sobel] = edgesEvalImg(probabilityEdgeMap_Dog_Sobel_1,groundTruthFile,orderOfGroundTruth1);
precision1_sobel_dogs = elementwiseDivider(cntP1_sobel,sumP1_sobel);
recall1_sobel_dogs = elementwiseDivider(cntR1_sobel,sumR1_sobel);
%Canny
[thrs1_canny,cntR1_canny,sumR1_canny,cntP1_canny,sumP1_canny] = edgesEvalImg(binaryEdgeMap_Dogs_Canny_1,groundTruthFile,orderOfGroundTruth1);
precision1_canny_dogs = elementwiseDivider(cntP1_canny,sumP1_canny);
recall1_canny_dogs = elementwiseDivider(cntR1_canny,sumR1_canny);

%Structured Edge
[thrs1_SE,cntR1_SE,sumR1_SE,cntP1_SE,sumP1_SE] = edgesEvalImg(probabilityEdgeMap_Dog_SE_1,groundTruthFile,orderOfGroundTruth1);
precision1_SE_dogs = elementwiseDivider(cntP1_SE,sumP1_SE);
recall1_SE_dogs = elementwiseDivider(cntR1_SE,sumR1_SE);

%Ground Truth 2
orderOfGroundTruth2 = 2; 
%Sobel
[thrs2_sobel,cntR2_sobel,sumR2_sobel,cntP2_sobel,sumP2_sobel] = edgesEvalImg(probabilityEdgeMap_Dog_Sobel_1,groundTruthFile,orderOfGroundTruth2);
precision2_sobel_dogs = elementwiseDivider(cntP2_sobel,sumP2_sobel);
recall2_sobel_dogs = elementwiseDivider(cntR2_sobel,sumR2_sobel);

%Canny
[thrs2_canny,cntR2_canny,sumR2_canny,cntP2_canny,sumP2_canny] = edgesEvalImg(binaryEdgeMap_Dogs_Canny_1,groundTruthFile,orderOfGroundTruth2);
precision2_canny_dogs = elementwiseDivider(cntP2_canny,sumP2_canny);
recall2_canny_dogs = elementwiseDivider(cntR2_canny,sumR2_canny);

%Structured Edge
[thrs2_SE,cntR2_SE,sumR2_SE,cntP2_SE,sumP2_SE] = edgesEvalImg(probabilityEdgeMap_Dog_SE_1,groundTruthFile,orderOfGroundTruth2);
precision2_SE_dogs = elementwiseDivider(cntP2_SE,sumP2_SE);
recall2_SE_dogs = elementwiseDivider(cntR2_SE,sumR2_SE);

%Ground Truth 3
orderOfGroundTruth3 = 3; 
%Sobel
[thrs3_sobel,cntR3_sobel,sumR3_sobel,cntP3_sobel,sumP3_sobel] = edgesEvalImg(probabilityEdgeMap_Dog_Sobel_1,groundTruthFile,orderOfGroundTruth3);
precision3_sobel_dogs = elementwiseDivider(cntP3_sobel,sumP3_sobel);
recall3_sobel_dogs = elementwiseDivider(cntR3_sobel,sumR3_sobel);

%Canny
[thrs3_canny,cntR3_canny,sumR3_canny,cntP3_canny,sumP3_canny] = edgesEvalImg(binaryEdgeMap_Dogs_Canny_1,groundTruthFile,orderOfGroundTruth3);
precision3_canny_dogs = elementwiseDivider(cntP3_canny,sumP3_canny);
recall3_canny_dogs = elementwiseDivider(cntR3_canny,sumR3_canny);

%Structured Edge
[thrs3_SE,cntR3_SE,sumR3_SE,cntP3_SE,sumP3_SE] = edgesEvalImg(probabilityEdgeMap_Dog_SE_1,groundTruthFile,orderOfGroundTruth3);
precision3_SE_dogs = elementwiseDivider(cntP3_SE,sumP3_SE);
recall3_SE_dogs = elementwiseDivider(cntR3_SE,sumR3_SE);

%Ground Truth 4
orderOfGroundTruth4 = 4; 
%Sobel
[thrs4_sobel,cntR4_sobel,sumR4_sobel,cntP4_sobel,sumP4_sobel] = edgesEvalImg(probabilityEdgeMap_Dog_Sobel_1,groundTruthFile,orderOfGroundTruth4);
precision4_sobel_dogs = elementwiseDivider(cntP4_sobel,sumP4_sobel);
recall4_sobel_dogs = elementwiseDivider(cntR4_sobel,sumR4_sobel);
%Canny
[thrs4_canny,cntR4_canny,sumR4_canny,cntP4_canny,sumP4_canny] = edgesEvalImg(binaryEdgeMap_Dogs_Canny_1,groundTruthFile,orderOfGroundTruth4);
precision4_canny_dogs = elementwiseDivider(cntP4_canny,sumP4_canny);
recall4_canny_dogs = elementwiseDivider(cntR4_canny,sumR4_canny);

%Structured Edge
[thrs4_SE,cntR4_SE,sumR4_SE,cntP4_SE,sumP4_SE] = edgesEvalImg(probabilityEdgeMap_Dog_SE_1,groundTruthFile,orderOfGroundTruth4);
precision4_SE_dogs = elementwiseDivider(cntP4_SE,sumP4_SE);
recall4_SE_dogs = elementwiseDivider(cntR4_SE,sumR4_SE);

%Ground Truth 5
orderOfGroundTruth5 = 5; 
%Sobel
[thrs5_sobel,cntR5_sobel,sumR5_sobel,cntP5_sobel,sumP5_sobel] = edgesEvalImg(probabilityEdgeMap_Dog_Sobel_1,groundTruthFile,orderOfGroundTruth5);
precision5_sobel_dogs = elementwiseDivider(cntP5_sobel,sumP5_sobel);
recall5_sobel_dogs = elementwiseDivider(cntR5_sobel,sumR5_sobel);
%Canny
[thrs5_canny,cntR5_canny,sumR5_canny,cntP5_canny,sumP5_canny] = edgesEvalImg(binaryEdgeMap_Dogs_Canny_1,groundTruthFile,orderOfGroundTruth5);
precision5_canny_dogs = elementwiseDivider(cntP5_canny,sumP5_canny);
recall5_canny_dogs = elementwiseDivider(cntR5_canny,sumR5_canny);

%Structured Edge
[thrs5_SE,cntR5_SE,sumR5_SE,cntP5_SE,sumP5_SE] = edgesEvalImg(probabilityEdgeMap_Dog_SE_1,groundTruthFile,orderOfGroundTruth5);
precision5_SE_dogs = elementwiseDivider(cntP5_SE,sumP5_SE);
recall5_SE_dogs = elementwiseDivider(cntR5_SE,sumR5_SE);


%Calculater Mean Precision and Mean Recall over 5 ground truths
numOfGroundtruth = 5;

%Sobel
meanPrecision_sobel_dogs =elementwiseDividerByInteger( elementwiseAdditioner(precision5_sobel_dogs,elementwiseAdditioner(precision4_sobel_dogs, elementwiseAdditioner(precision3_sobel_dogs, elementwiseAdditioner(precision1_sobel_dogs,precision2_sobel_dogs)))) ,numOfGroundtruth) ;
meanRecall_sobel_dogs =elementwiseDividerByInteger( elementwiseAdditioner(recall5_sobel_dogs,elementwiseAdditioner(recall4_sobel_dogs, elementwiseAdditioner(recall3_sobel_dogs, elementwiseAdditioner(recall1_sobel_dogs,recall2_sobel_dogs)))) ,numOfGroundtruth) ;

%Canny
meanPrecision_canny_dogs =elementwiseDividerByInteger( elementwiseAdditioner(precision5_canny_dogs,elementwiseAdditioner(precision4_canny_dogs, elementwiseAdditioner(precision3_canny_dogs, elementwiseAdditioner(precision1_canny_dogs,precision2_canny_dogs)))) ,numOfGroundtruth) ;
meanRecall_canny_dogs =elementwiseDividerByInteger( elementwiseAdditioner(recall5_canny_dogs,elementwiseAdditioner(recall4_canny_dogs, elementwiseAdditioner(recall3_canny_dogs, elementwiseAdditioner(recall1_canny_dogs,recall2_canny_dogs)))) ,numOfGroundtruth) ;

%Structured Edge
meanPrecision_SE_dogs =elementwiseDividerByInteger( elementwiseAdditioner(precision5_SE_dogs,elementwiseAdditioner(precision4_SE_dogs, elementwiseAdditioner(precision3_SE_dogs, elementwiseAdditioner(precision1_SE_dogs,precision2_SE_dogs)))) ,numOfGroundtruth) ;
meanRecall_SE_dogs =elementwiseDividerByInteger( elementwiseAdditioner(recall5_SE_dogs,elementwiseAdditioner(recall4_SE_dogs, elementwiseAdditioner(recall3_SE_dogs, elementwiseAdditioner(recall1_SE_dogs,recall2_SE_dogs)))) ,numOfGroundtruth) ;


%Calculate F measure
%Sobel
meanFScore_sobel_dogs = FScoreMatrixCalculator(meanPrecision_sobel_dogs,meanRecall_sobel_dogs);
%Canny
meanFScore_canny_dogs = FScoreMatrixCalculator(meanPrecision_canny_dogs,meanRecall_canny_dogs);

%Structured Edge
meanFScore_SE_dogs = FScoreMatrixCalculator(meanPrecision_SE_dogs,meanRecall_SE_dogs);


%Find max F score with respect to threshold
%Sobel
[max_FScore_sobel_dog, location_sobel_dog] = MaxInMatrixFinder(meanFScore_sobel_dogs);
threshold_maxFscore_sobel_dog = thrs1_sobel(location_sobel_dog);
%Canny
[max_FScore_canny_dog, location_canny_dog] = MaxInMatrixFinder(meanFScore_canny_dogs);
threshold_maxFscore_canny_dog = thrs1_canny(location_canny_dog);

%Structured Edge
[max_FScore_SE_dog, location_SE_dog] = MaxInMatrixFinder(meanFScore_SE_dogs);
threshold_maxFscore_SE_dog = thrs1_SE(location_SE_dog);

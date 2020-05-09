%load images all images have same dimension

imgHeight = 600;
imgWidth = 450;
imgBytePerPixel = 1;

fileName_compt = "comp.raw";

img_comp = readraw_gray(fileName_compt,imgWidth,imgHeight,imgBytePerPixel);

%Generate Laws filter bank
filterBank = filterBankGenerator();

%Generate 15 gray scale images by 15 chosen Laws filter
imagesStroage = convolutionalImgGenerator(img_comp, filterBank);

%Computation energy feature
windowSize = 25; %window size can be various
pixelWiseEnergyFeatureStorage = energyFeatureGenerator(imagesStroage,windowSize);

%Normalize energy feature by L5L5 component
normalizedPixelWiseEnergyFeatureStorage = normalizerByL5L5(pixelWiseEnergyFeatureStorage);
%Perform Segmentation by K means algorithm
transformedFeatureVectors = XtrainCreator(normalizedPixelWiseEnergyFeatureStorage);

numofClusters =6 ;
[clusterIndex, centroids] = kmeans(transformedFeatureVectors,numofClusters);

%Converte clusterIndex to image
convertedImgHeight = 450;
convertedImgWidth = 600;
segmentImg_labelVersion = cluster2ImgConverter(clusterIndex,convertedImgHeight,convertedImgWidth);
%gray scale conversion
convertedImg = grayScaleConverter(segmentImg_labelVersion);

imshow(uint8(convertedImg))
// HW2_EE569.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include "ImageLoadRead.h"
#include <cmath>
#include <math.h>
#include "ImageProcessorHW2.h"
#include "FiltersHW2.h"
#include "ImageWriteSave.h"
#include<tuple>//for tuple

using namespace cv;

using namespace std;


int main(int argc, char** argv)
{
    //paths for Dogs.raw
    string readPathDogs = argv[1];
    string savePathXDirectionEdgeMap_Dogs = argv[2];
    string savePathYDirectionEdgeMap_Dogs = argv[3];
    string savePathMagnitudeEdgeMap_Dogs = argv[4];
    string readPathDogsMagnitudeEdgeMap = argv[4];//read the same raw file as a file in save path magnitude edge map
    string savePathTunnedEdgeMap_Dogs = argv[10];

    //paths for Gallery.raw
    string readPathGallery = argv[5];
    string savePathXDirectionEdgeMap_Gallery = argv[6];
    string savePathYDirectionEdgeMap_Gallery = argv[7];
    string savePathMagnitudeEdgeMap_Gallery = argv[8];
    string readPathMagnitudeEdgeMap_Gallery = argv[8];//read the same raw file as a file in save path magnitude edge map
    string savePathTunnedEdgeMap_Gallery = argv[9];


    //paths for LightHouse.raw
    string readPathLightHouse = argv[11];
    string savePathFixedThresholding_LightHouse = argv[12];
    string savePathRandomThresholding_LightHouse = argv[13];
    string savePath_thresMatrix_2X2_LightHouse = argv[14];
    string savePath_thresMatrix_8X8_LightHouse = argv[15];
    string savePath_thresMatrix_32X32_LightHouse = argv[16];
    string savePathLightHouse_floyd = argv[17];
    string savePathLightHouse_jjn = argv[18];
    string savePathLightHouse_stucki = argv[19];

    //paths for Rose.raw
    string readPathRose = argv[20];
    string savePathRose_Seperable = argv[21];
    







    //******************************Digital Halftoning**************************************
    //*******LightHouse.raw

    //Dimensions of LightHouse.raw from HW2 description
    int BytesPerPixelLightHouse = 1;
    int widthLightHouse = 750;
    int heightLightHouse = 500;

    //Create an object to load and read raw file
    ImageLoadRead readObjLightHouse(heightLightHouse, widthLightHouse, BytesPerPixelLightHouse, readPathLightHouse);
    //Load Image
    readObjLightHouse.rawImgLoad();

    //get maxtrix form image data
    std::vector<std::vector<std::vector<double>>> lightHouseImg = readObjLightHouse.getImageData();
    //Make a processor for LightHouse image
    ImageProcessorHW2 imageProcessorLightHouse(heightLightHouse, widthLightHouse, BytesPerPixelLightHouse, lightHouseImg);
    //Create filter object for LightHouse image processing
    int temp_lightHouse = -1; //Need some random integer to create object. It has no meaning.
    FiltersHW2 filterObjLightHouse(temp_lightHouse);



    //*************Error Diffusion
    //(1)Floyd-Steinberg's error diffusion.
    //Generate 2 Floyd-Steinberg's error diffusion masks
    std::vector<std::vector<double>> left2Right_floydErrorDiffusionMask = filterObjLightHouse.left2Right_floydErrorDiffusionMaskGenerator();
    std::vector<std::vector<double>> right2Left_floydErrorDiffusionMask = filterObjLightHouse.right2Left_floydErrorDiffusionMaskGenerator();
    
    //Perform Error diffusion. This Implement include cropping image process already in it. No need to perform cropping image
    std::vector<std::vector<std::vector<double>>>final_ErrorDiffusedBinaryImg_floyd = imageProcessorLightHouse.halftonedImgGenerator_ErrorDiffusionV2(lightHouseImg, left2Right_floydErrorDiffusionMask, right2Left_floydErrorDiffusionMask);

    //(2)JJN Error Diffusion.
    //Generate 2 JJN Error Diffuion masks
    std::vector<std::vector<double>> left2Right_jjnErrorDiffusionMask = filterObjLightHouse.left2Right_jjnErrorDiffusionMaskGenerator();
    std::vector<std::vector<double>> right2Left_jjnErrorDiffusionMask = filterObjLightHouse.right2Left_jjnErrorDiffusionMaskGenerator();

    //Perform Error diffusion. This Implement include cropping image process already in it. No need to perform cropping image
    std::vector<std::vector<std::vector<double>>>final_ErrorDiffusedBinaryImg_jjn = imageProcessorLightHouse.halftonedImgGenerator_ErrorDiffusionV2(lightHouseImg, left2Right_jjnErrorDiffusionMask, right2Left_jjnErrorDiffusionMask);


    //(3)Stucki Error Diffusion
    //Generate 2 Stucki Error Diffuion masks
    std::vector<std::vector<double>> left2Right_stuckiErrorDiffusionMask = filterObjLightHouse.left2Right_stuckiErrorDiffusionMaskGenerator();
    std::vector<std::vector<double>> right2Left_stuckiErrorDiffusionMask = filterObjLightHouse.right2Left_stuckiErrorDiffusionMaskGenerator();

    //Perform Error diffusion. This Implement include cropping image process already in it. No need to perform cropping image
    std::vector<std::vector<std::vector<double>>>final_ErrorDiffusedBinaryImg_stucki = imageProcessorLightHouse.halftonedImgGenerator_ErrorDiffusionV2(lightHouseImg, left2Right_stuckiErrorDiffusionMask, right2Left_stuckiErrorDiffusionMask);



    //Save the processed image as raw file
    ImageWriteSave saveObjLightHouse_floyd(heightLightHouse, widthLightHouse, savePathLightHouse_floyd);
    saveObjLightHouse_floyd.saveAsRawfile(final_ErrorDiffusedBinaryImg_floyd);

    ImageWriteSave saveObjLightHouse_jjn(heightLightHouse, widthLightHouse, savePathLightHouse_jjn);
    saveObjLightHouse_jjn.saveAsRawfile(final_ErrorDiffusedBinaryImg_jjn);

    ImageWriteSave saveObjLightHouse_stucki(heightLightHouse, widthLightHouse, savePathLightHouse_stucki);
    saveObjLightHouse_stucki.saveAsRawfile(final_ErrorDiffusedBinaryImg_stucki);




    //*************Dithering
    //(3)Dithering Matrix

    //Create 2x2 Index Matrix and corresponding threshold matrix
    std::vector<std::vector<double>> basicIndexMatrix = filterObjLightHouse.basicIndexMatrixGenerator();
    std::vector<std::vector<double>> basicThresholdingMatrix = filterObjLightHouse.thresholdMatrixGenerator(basicIndexMatrix);
    //Create 8X8 Index Matrix and corresponding threshold matrix
    std::vector<std::vector<double>> indexMatrix_4X4 = filterObjLightHouse.indexMatrixGenerator(basicIndexMatrix); //4X4 index matrix
    std::vector<std::vector<double>> indexMatrix_8X8 = filterObjLightHouse.indexMatrixGenerator(indexMatrix_4X4); //8X8 index matrix
    std::vector<std::vector<double>> ThresholdingMatrix_8X8 = filterObjLightHouse.thresholdMatrixGenerator(indexMatrix_8X8); //8X8 threshold matrix
    //Create 32X32 Index Matrix and corresponding threshold matrix
    std::vector<std::vector<double>> indexMatrix_16X16 = filterObjLightHouse.indexMatrixGenerator(indexMatrix_8X8); //16X16 index matrix
    std::vector<std::vector<double>> indexMatrix_32X32 = filterObjLightHouse.indexMatrixGenerator(indexMatrix_16X16); //32X32 index matrix
    std::vector<std::vector<double>> ThresholdingMatrix_32X32 = filterObjLightHouse.thresholdMatrixGenerator(indexMatrix_32X32); //32X32 threshold matrix

    //Perfom dithering with each threshold matrix
    //2X2 threshold matrix
    std::vector<std::vector<std::vector<double>>> TonedImg_thresholdMarix_2X2_LightHouse = imageProcessorLightHouse.ditheringMatrixImgGenerator(lightHouseImg, basicThresholdingMatrix);
    //8X8 threshold matrix
    std::vector<std::vector<std::vector<double>>> TonedImg_thresholdMarix_8X8_LightHouse = imageProcessorLightHouse.ditheringMatrixImgGenerator(lightHouseImg, ThresholdingMatrix_8X8);
    //32X32 threshold matrix
    std::vector<std::vector<std::vector<double>>> TonedImg_thresholdMarix_32X32_LightHouse = imageProcessorLightHouse.ditheringMatrixImgGenerator(lightHouseImg, ThresholdingMatrix_32X32);


    //Save Images
    ImageWriteSave saveObjLightHouse_2X2(heightLightHouse, widthLightHouse, savePath_thresMatrix_2X2_LightHouse);
    saveObjLightHouse_2X2.saveAsRawfile(TonedImg_thresholdMarix_2X2_LightHouse);

    ImageWriteSave saveObjLightHouse_8X8(heightLightHouse, widthLightHouse, savePath_thresMatrix_8X8_LightHouse);
    saveObjLightHouse_8X8.saveAsRawfile(TonedImg_thresholdMarix_8X8_LightHouse);

    ImageWriteSave saveObjLightHouse_32X32(heightLightHouse, widthLightHouse, savePath_thresMatrix_32X32_LightHouse);
    saveObjLightHouse_32X32.saveAsRawfile(TonedImg_thresholdMarix_32X32_LightHouse);




    //(2) Random Thresholding
    //Perform Random Thresholding half toning to the LightHouse Image
    //Note that resulting image will be different every time you run the program because of randomness of Dithering algorithm.
    std::vector<std::vector<std::vector<double>>> RandomThresholdingTonedImg_LightHouse = imageProcessorLightHouse.randomThresholdedImgGenerator(lightHouseImg);

    //Save a result Image
    ImageWriteSave saveObjLightHouseRandomThreshold(heightLightHouse, widthLightHouse, savePathRandomThresholding_LightHouse);
    saveObjLightHouseRandomThreshold.saveAsRawfile(RandomThresholdingTonedImg_LightHouse);



    //(1) Fixed Thresholding
    //Perform Fixed Thresholding half toning to the LightHouse image
    double fixedThreshold = 128; //This threshold is from the HW2 description
    std::vector<std::vector<std::vector<double>>> fixedThresholdingTonedImg_LightHouse = imageProcessorLightHouse.fixedThresholdedImgGenerator(lightHouseImg, fixedThreshold);

    //Save a result Image
    ImageWriteSave saveObjLightHouseFixedThreshold(heightLightHouse, widthLightHouse, savePathFixedThresholding_LightHouse);

    saveObjLightHouseFixedThreshold.saveAsRawfile(fixedThresholdingTonedImg_LightHouse);





    
    
    
    
    
    
    
    
    
    //*************Rose.raw
    //Dimension of Rose.raw from HW2 description
    int BytePerPixelRose = 3;
    int widthRose = 640;
    int heightRose = 480;

    //Create an object to load and read raw file
    ImageLoadRead readObjRose(heightRose, widthRose, BytePerPixelRose, readPathRose);
    //Load Image
    readObjRose.rawImgLoad();

    //get maxtrix form image data
    std::vector<std::vector<std::vector<double>>> roseImg = readObjRose.getImageData();
    //get each channel
    std::vector<std::vector<std::vector<double>>> redChannel_Rose = readObjRose.getRedChannel();
    std::vector<std::vector<std::vector<double>>> greenChannel_Rose = readObjRose.getGreenChannel();
    std::vector<std::vector<std::vector<double>>> blueChannel_Rose = readObjRose.getBlueChannel();

    //Make a processor for Rose image
    ImageProcessorHW2 imageProcessorRose(heightRose, widthRose, BytePerPixelRose, roseImg);
    //Create filter object for Rose image processing
    int temp_Rose = -1; //Need some random integer to create object. It has no meaning.
    FiltersHW2 filterObjRose(temp_Rose);
    //Generate 2 Floyd-Steinberg's error diffusion masks
    std::vector<std::vector<double>> left2Right_floydErrorDiffusionMask_Rose = filterObjRose.left2Right_floydErrorDiffusionMaskGenerator();
    std::vector<std::vector<double>> right2Left_floydErrorDiffusionMask_Rose = filterObjRose.right2Left_floydErrorDiffusionMaskGenerator();

    //Perform Error Diffusion for each channel
    std::vector<std::vector<std::vector<double>>> tonnedRed;
    std::vector<std::vector<std::vector<double>>> tonnedGreen;
    std::vector<std::vector<std::vector<double>>> tonnedBlue;

    tie(tonnedRed,tonnedGreen,tonnedBlue) = imageProcessorRose.halftonedImgGenerator_ErrorDiffusion_Color(redChannel_Rose, greenChannel_Rose, blueChannel_Rose, left2Right_floydErrorDiffusionMask_Rose, right2Left_floydErrorDiffusionMask_Rose);

    
    //save 3 channels as one color image raw file
    ImageWriteSave saveObjRose_Seperable(heightRose, widthRose, savePathRose_Seperable);
    saveObjRose_Seperable.saveAsRawfileColor(tonnedRed, tonnedGreen, tonnedBlue);






    






    //**********************Image Edge Detection*************************************************
    //******************************************Sobel Edge Detector******************************
     //Create Filter class that has several filters neeed in this project
    int temp = -1; // to create FiltersHW2 class, I need some arbitrary integer
    FiltersHW2 filterObj(temp);
    vector<vector<double>> xDirectionSobelFilter = filterObj.xDirectionSobelFilterGenerator();
    vector<vector<double>> yDirectionSobelFilter = filterObj.yDirectionSobelFilterGenerator();


    //*****Gallery.raw
    //Load and read image data from raw files first 
    int BytesPerPixelGallery = 3;
    int widthGallery = 481;
    int heightGallery = 321;

    ImageLoadRead readObjGallery(heightGallery, widthGallery, BytesPerPixelGallery, readPathGallery);
    //Load Image
    readObjGallery.rawImgLoad();

    //get matrix data of raw image
    std::vector<std::vector<std::vector<double>>> imageData_red_Gallery = readObjGallery.getRedChannel();
    std::vector<std::vector<std::vector<double>>> imageData_green_Gallery = readObjGallery.getGreenChannel();
    std::vector<std::vector<std::vector<double>>> imageData_blue_Gallery = readObjGallery.getBlueChannel();

    //make processors for images
    int BytePerPixelGalleryGray = 1;
    ImageProcessorHW2 imagePreProcessorGallery(heightGallery, widthGallery, BytePerPixelGalleryGray, readObjGallery.getImageData());

    //Convert RGB to gray scale using a given formula in hw2 description.
    vector<vector<vector<double>>> converted_Gallery_gray = imagePreProcessorGallery.RGB2Gray(imageData_red_Gallery, imageData_green_Gallery, imageData_blue_Gallery);

    ImageProcessorHW2 imgProcessorGallery(heightGallery, widthGallery, BytePerPixelGalleryGray, converted_Gallery_gray);

    vector<vector<vector<double>>> xDirectionGradientEdgeMap_Gallery = imgProcessorGallery.xDirectionGradientEdgeMapGenerator(xDirectionSobelFilter, converted_Gallery_gray);
    vector<vector<vector<double>>> yDirectionGradientEdgeMap_Gallery = imgProcessorGallery.yDirectionGradientEdgeMapGenerator(yDirectionSobelFilter, converted_Gallery_gray);


    //Save x direction Gradient Edge Map as raw file
    ImageWriteSave saveObjGalleryXEdge(heightGallery, widthGallery, savePathXDirectionEdgeMap_Gallery);
    ImageWriteSave saveObjGalleryYEdge(heightGallery, widthGallery, savePathYDirectionEdgeMap_Gallery);

    saveObjGalleryXEdge.saveAsRawfile(xDirectionGradientEdgeMap_Gallery);
    saveObjGalleryYEdge.saveAsRawfile(yDirectionGradientEdgeMap_Gallery);

    //Find gradient Magnitude Edge Map
    vector<vector<vector<double>>> gradientMagnitudeEdgeMap_Gallery = imgProcessorGallery.gradientMagnitudeEdgeMapGenerator(xDirectionSobelFilter, yDirectionSobelFilter, converted_Gallery_gray);

    //Save gradient magnitude edge map
    ImageWriteSave saveObjGalleryMagnitude(heightGallery, widthGallery, savePathMagnitudeEdgeMap_Gallery);

    saveObjGalleryMagnitude.saveAsRawfile(gradientMagnitudeEdgeMap_Gallery);

    //Tunning gradient magnitude edge map
    //Load and read Gallery gradient magnitude edge map
    int BytesPerPixelGallery_Magnitue = 1;
    ImageLoadRead readObjGalleryMagnitude(heightGallery, widthGallery, BytesPerPixelGallery_Magnitue, readPathMagnitudeEdgeMap_Gallery);
    readObjGalleryMagnitude.rawImgLoad();

    //Get a magnitude Gallery edge map data
    vector<vector<vector<double>>> magnitudeEdgeMap_Gallery = readObjGalleryMagnitude.getImageData();

    //Tunning(i.e. thresholding edges)
    double targetThresholdPercentage_Gallery = 90;//90 means 90 percent in cumulative histogram.
    vector<vector<vector<double>>> tunnedEdgeMap_Gallery = imgProcessorGallery.tunnedAndBinaryEdgeMapGeneratorByMaxGradient(magnitudeEdgeMap_Gallery,targetThresholdPercentage_Gallery);

    //Save a result image
    ImageWriteSave saveObjGalleryTunned(heightGallery, widthGallery, savePathTunnedEdgeMap_Gallery);

    saveObjGalleryTunned.saveAsRawfile(tunnedEdgeMap_Gallery);


    //****Dogs.raw
    //Load and read image data from raw files first 
    int BytesPerPixelDogs = 3;
    int widthDogs = 481;
    int heightDogs = 321;
    


    ImageLoadRead readObjDogs(heightDogs, widthDogs, BytesPerPixelDogs, readPathDogs);
    //Load Image
    readObjDogs.rawImgLoad();

    //get matrix data of raw image
    std::vector<std::vector<std::vector<double>>> imageData_red_dogs = readObjDogs.getRedChannel();
    std::vector<std::vector<std::vector<double>>> imageData_green_dogs = readObjDogs.getGreenChannel();
    std::vector<std::vector<std::vector<double>>> imageData_blue_dogs = readObjDogs.getBlueChannel();

    //make processors for images
    int BytePerPixelDogsGray = 1;
    ImageProcessorHW2 imagePreProcessorDogs(heightDogs, widthDogs, BytePerPixelDogsGray, readObjDogs.getImageData());

    //Convert RGB to gray scale using a given formula in hw2 description.
    vector<vector<vector<double>>> converted_dogs_gray = imagePreProcessorDogs.RGB2Gray(imageData_red_dogs, imageData_green_dogs, imageData_blue_dogs);

   
    ImageProcessorHW2 imgProcessorDogs(heightDogs, widthDogs, BytePerPixelDogsGray, converted_dogs_gray);
    
   
    vector<vector<vector<double>>> xDirectionGradientEdgeMap = imgProcessorDogs.xDirectionGradientEdgeMapGenerator(xDirectionSobelFilter, converted_dogs_gray);
    vector<vector<vector<double>>> yDirectionGradientEdgeMap = imgProcessorDogs.yDirectionGradientEdgeMapGenerator(yDirectionSobelFilter, converted_dogs_gray);

    //Save x direction Gradient Edge Map as raw file
    ImageWriteSave saveObjDogsXEdge(heightDogs, widthDogs, savePathXDirectionEdgeMap_Dogs);
    ImageWriteSave saveObjDogsYEdge(heightDogs, widthDogs, savePathYDirectionEdgeMap_Dogs);

    saveObjDogsXEdge.saveAsRawfile(xDirectionGradientEdgeMap);
    saveObjDogsYEdge.saveAsRawfile(yDirectionGradientEdgeMap);


    //Find gradient Magnitude Edge Map
    vector<vector<vector<double>>> gradientMagnitudeEdgeMap = imgProcessorDogs.gradientMagnitudeEdgeMapGenerator(xDirectionSobelFilter, yDirectionSobelFilter, converted_dogs_gray);

    //Save gradient magnitude edge map
    ImageWriteSave saveObjDogsMagnitude(heightDogs, widthDogs, savePathMagnitudeEdgeMap_Dogs);

    saveObjDogsMagnitude.saveAsRawfile(gradientMagnitudeEdgeMap);


    //Tunning gradient magnitude edgemap
    //Load and read Dogs gradient magnitude edge map
    int BytesPerPixelDogs_Magnitue = 1;
    ImageLoadRead readObjDogsMagnitude(heightDogs, widthDogs, BytesPerPixelDogs_Magnitue, readPathDogsMagnitudeEdgeMap);
    //load Dogs.raw file to buffer and read it to matrix data
    readObjDogsMagnitude.rawImgLoad();

    //Get a magnitude Gallery edge map data
    vector<vector<vector<double>>> magnitudeEdgeMap_Dogs = readObjDogsMagnitude.getImageData();

    //Tunning(i.e. thresholding edges)
    double targetThresholdPercentage_Dogs = 90; //90 means 90 percent in cumulative histogram.
    vector<vector<vector<double>>> tunnedEdgeMap_Dogs = imgProcessorDogs.tunnedAndBinaryEdgeMapGeneratorByMaxGradient(magnitudeEdgeMap_Dogs, targetThresholdPercentage_Dogs);

    //Save a result image
    ImageWriteSave saveObjDogsTunned(heightDogs, widthDogs, savePathTunnedEdgeMap_Dogs);

    saveObjDogsTunned.saveAsRawfile(tunnedEdgeMap_Dogs);


    return 0;
}


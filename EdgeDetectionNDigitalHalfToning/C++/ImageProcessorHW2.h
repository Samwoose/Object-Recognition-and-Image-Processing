#pragma once

#ifndef IMAGEPROCESSORHW2_H
#define IMAGEPROCESSORHW2_H
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <math.h>
#include <ctime> //this is for random number
#include<tuple>//for tuple

class ImageProcessorHW2
{


private:
	int imgHeight;
	int imgWidth;
	int imgBytesPerPixel;


	std::vector<std::vector<std::vector<double>>> inputImg;


public:
	//constructor 
	ImageProcessorHW2(int height, int width, int BytePerPixel, std::vector<std::vector<std::vector<double>>> inputImage);

	//destructor
	~ImageProcessorHW2();

	


	//************Class Methods for general Image Processing Purposes
	//return a extended boundary matrix by zero padding method 
	std::vector<std::vector<std::vector<double>>> boundaryExtension(std::vector<std::vector<double>> filter);
	//return extended 3D matrix Version2
	//please perform this before apply any filter
	std::vector<std::vector<std::vector<double>>> boundaryExtensionV2(std::vector<std::vector<std::vector<double>>> inputImg, std::vector<std::vector<double>> filter);


	//return a filtered extended matrix by a given filter
	std::vector<std::vector<std::vector<double>>> convolution(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> extendedMatrix);

	//Convolution function with different normalization
	std::vector<std::vector<std::vector<double>>> convolutionV2(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> extendedMatrix);


	//return a cropped matrix by size of a given filter
	std::vector<std::vector<std::vector<double>>> matrixCropper(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> extendedMatrix);

	//Post processing. Handle Negative values and Large values greated during any processing manual mapping e.g. negative -> 0, Bigger than 255 -> 255 
	std::vector<std::vector<std::vector<double>>> pixelValuePostProcessor(std::vector<std::vector<std::vector<double>>> processedMatrix);

	//max value finder in a given 3D matrix
	double maxFinder(std::vector<std::vector<std::vector<double>>> targetMatrix);

	//min value finder in a given 3D matrix
	double minFinder(std::vector<std::vector<std::vector<double>>> targetMatrix);

	//Normalize a given matrix using minimum and maximum values in the matrix
	std::vector<std::vector<std::vector<double>>> minMaxNormalizer(std::vector<std::vector<std::vector<double>>> targetMatrix);




	//*************Class Methods for HW2 Problem 1 Edge Detection
	//return a converted gray scale image from color image
	std::vector<std::vector<std::vector<double>>> RGB2Gray(std::vector<std::vector<std::vector<double>>> redChannel, std::vector<std::vector<std::vector<double>>> greenChannel, std::vector<std::vector<std::vector<double>>>blueChannel);

	//Generate x-direction gradient edge map
	std::vector<std::vector<std::vector<double>>> xDirectionGradientEdgeMapGenerator(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> oriImgData);
	//Generate y-direction gradient edge map
	std::vector<std::vector<std::vector<double>>> yDirectionGradientEdgeMapGenerator(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> oriImgData);

	//Generate normalized gradient magnitude map
	std::vector<std::vector<std::vector<double>>> gradientMagnitudeEdgeMapGenerator(std::vector<std::vector<double>> xDirectionFilter, std::vector<std::vector<double>> yDirectionFilter, std::vector<std::vector<std::vector<double>>> oriImgData);

	//Count number of each intensity in one channel and return a corresponding 1D vector that includes the number of appearance of each intensity
	std::vector<int> intensityCounterOneChannel(std::vector<std::vector<std::vector<double>>> oneChannel);

	
	//Calculate Normalized Probability and return a 1D vector that includes normalized probability for each intensity
	std::vector<double> normalizedProbCalculator(std::vector<int> intensitiesOneChannel);

	
	//Calculate Cummulative Density Function(CDF) for each intensity and return a 1D vector than includes CDF for each intensity
	std::vector<double> CDFCalculator(std::vector<double> normalizedProbVector);

	//Generate a tunned edge map by threshold value
	std::vector<std::vector<std::vector<double>>> tunnedAndBinaryEdgeMapGenerator(std::vector<std::vector<std::vector<double>>> gradientMagnitudeEdgeMap , double targetThreshold);
	//Generate a tunned edge map by threshold value of gradient magnitude
	std::vector<std::vector<std::vector<double>>> tunnedAndBinaryEdgeMapGeneratorByMaxGradient(std::vector<std::vector<std::vector<double>>> gradientMagnitudeEdgeMap, double targetThreshold);


	//***************************Class Methods for Problem 2 Digital Halftoning

	//Generate a halftoned image by fixed thresholding from a gray scale image
	std::vector<std::vector<std::vector<double>>> fixedThresholdedImgGenerator(std::vector<std::vector<std::vector<double>>> inputImg, double targetThreshold);

	//Generate a halftoned image by random thresholding from a gray scale image
	std::vector<std::vector<std::vector<double>>> randomThresholdedImgGenerator(std::vector<std::vector<std::vector<double>>> inputImg);

	//Generate a halftoned image by a given dithering matrix(i.e., thresholding matrix)
	std::vector<std::vector<std::vector<double>>> ditheringMatrixImgGenerator(std::vector<std::vector<std::vector<double>>> inputImg, std::vector<std::vector<double>> thresholdMatrix);

	//***Error Diffusion
	//Generate a halftoned image by a provided error diffusion mask with the serpentine scanning
	std::vector<std::vector<std::vector<double>>> halftonedImgGenerator_ErrorDiffusion(std::vector<std::vector<std::vector<double>>> extendedInputImg, std::vector<std::vector<double>> left2Right_errorDiffusionMatrix, std::vector<std::vector<double>> right2Left_errorDiffusionMatrix);

	std::vector<std::vector<std::vector<double>>> halftonedImgGenerator_ErrorDiffusionV2(std::vector<std::vector<std::vector<double>>> inputImg, std::vector<std::vector<double>> left2Right_errorDiffusionMatrix, std::vector<std::vector<double>> right2Left_errorDiffusionMatrix);

	//universal converter for a given one channel 
	//e.g. Red->Cyan or Green->Magenta or Blue->Yellow or vice versa
	std::vector<std::vector<std::vector<double>>> oneChannelRGB2CMYConverter(std::vector<std::vector<std::vector<double>>> oneChannelImg );

	//***Color Image Error Diffusion
	//Generate a halftoned 3 channels by a provided error diffusion mask with the serpentine scanning
	std::tuple< std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>> , std::vector<std::vector<std::vector<double>>> > halftonedImgGenerator_ErrorDiffusion_Color(std::vector<std::vector<std::vector<double>>> redChannel, std::vector<std::vector<std::vector<double>>> greenChannel, std::vector<std::vector<std::vector<double>>> blueChannel, std::vector<std::vector<double>> left2Right_errorDiffusionMatrix, std::vector<std::vector<double>> right2Left_errorDiffusionMatrix);

};







#endif // ! IMAGEPROCESSORHW2_H


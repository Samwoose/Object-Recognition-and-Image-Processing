#pragma once

#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <math.h>

class ImageProcessor
{


private:
	int imgHeight ;
    int imgWidth ;
	int imgBytesPerPixel ;
	

	std::vector<std::vector<std::vector<double>>> inputImg; 
	

public:
	//constructor 
	ImageProcessor(int height, int width, int BytePerPixel, std::vector<std::vector<std::vector<double>>> inputImage);

	//destructor
	~ImageProcessor();

	//Class method for Method A Step1
	//Count number of each intensity in one channel and return a corresponding 1D vector that includes the number of appearance of each intensity
	std::vector<int> intensityCounterOneChannel(std::vector<std::vector<std::vector<double>>> oneChannel);

	//Class method for Method A Step2
	//Calculate Normalized Probability and return a 1D vector that includes normalized probability for each intensity
	std::vector<double> normalizedProbCalculator(std::vector<int> intensitiesOneChannel);

	//Class method for Method A Step3
	//Calculate Cummulative Density Function(CDF) for each intensity and return a 1D vector than includes CDF for each intensity
	std::vector<double> CDFCalculator(std::vector<double> normalizedProbVector);

	//Class method for Method A Step4
	//Generate and return transfer function(i.e., mapping function) that maps original intensity for each pixel to manipulated intensity 
	std::vector<double> mappingFuncGenerator(std::vector<double> CDFVector);

	//Class method for Method A Step5
	//Perform histogram manipulation method A(Transfer Function Method(i.e.,Mapping Function)) to input one channel image and image matrix that its histgram is modified.
	std::vector<std::vector<std::vector<double>>> hisManipulatorMethodA(std::vector<double> mappingFunction, std::vector<std::vector<std::vector<double>>> oneChannelImg);

	//Class method for Method B
	//Perform histogram manipulation methodB(Filling Buckets Method) to one channel input image and return image matrix that its histgram is modified
	std::vector<std::vector<std::vector<double>>> hisManipulatorMethodB(std::vector<std::vector<std::vector<double>>> oneChannelImg);




	//************Class Methods for general Image Processing Purposes
	//return a extended boundary matrix by zero padding method 
	std::vector<std::vector<std::vector<double>>> boundaryExtension(std::vector<std::vector<double>> filter);

	//return a filtered extended matrix by a given filter
	std::vector<std::vector<std::vector<double>>> convolution(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> extendedMatrix);

	//return a cropped matrix by size of a given filter
	std::vector<std::vector<std::vector<double>>> matrixCropper(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> extendedMatrix);

	//Calculate peak signal to noise ratio and return the value
	double PSNRcalculator(std::vector<std::vector<std::vector<double>>> originImage, std::vector<std::vector<std::vector<double>>> filteredImage);



	//****************Class Method for Denoising
	//Perform Bilateral Denoising Algorithm to an extended image and return extended filtered image
	std::vector<std::vector<std::vector<double>>> BilateralDenoising(std::vector<std::vector<std::vector<double>>> extendedImg, int filterSizeOfOneSide, double sigma_c, double sigma_s);

	//Perform Non Local Mean Denoising Algorithm to an non extended image(i.e. input image has the same dimension of image as noisy image) and return non extended filtered image
	std::vector<std::vector<std::vector<double>>> NonLocalMeanDenoising(std::vector<std::vector<std::vector<double>>> noisyImg, int largeWindowSizeOfOneSide, int smallWindowSizeOfSide, double hyperParameter_h, double hyperParameter_a);

};



#endif // ! IMAGEPROCESSOR_H


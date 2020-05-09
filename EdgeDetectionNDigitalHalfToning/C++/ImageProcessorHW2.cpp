#include "ImageProcessorHW2.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <math.h>
#include <ctime> //this is for random number
#include<tuple>//for tuple

ImageProcessorHW2::ImageProcessorHW2(int height, int width, int BytesPerPixel, std::vector<std::vector<std::vector<double>>> inputImage) {
	imgHeight = height;
	imgWidth = width;
	imgBytesPerPixel = BytesPerPixel;

	inputImg = inputImage;

}
ImageProcessorHW2::~ImageProcessorHW2() {}

//max value finder in a given 3D matrix
double ImageProcessorHW2::maxFinder(std::vector<std::vector<std::vector<double>>> targetMatrix) {
	double maxInGivenMatrix = 0.0;
	double currentMax = -100000000.0; //low bound of maximum. Assume that all values in matrix are greater than this initial value 

	int imgHeight = targetMatrix.size();
	int imgWidth = targetMatrix[0].size();

	//Search maxmum value in the matrix linearly
	for (int row = 0; row < imgHeight; row++) {
		for (int col = 0; col < imgWidth; col++) {
			if (targetMatrix[row][col][0] > currentMax) {
				currentMax = targetMatrix[row][col][0];
			}
		}
	}
	maxInGivenMatrix = currentMax;

	return maxInGivenMatrix;

}

//min value finder in a given 3D matrix
double ImageProcessorHW2::minFinder(std::vector<std::vector<std::vector<double>>> targetMatrix) {
	double minInGivenMatrix = 0.0;
	double currentMin = +100000000.0; //upper bound of maximum. Assume that all values in matrix are lower than this initial value 
	int imgHeight = targetMatrix.size();
	int imgWidth = targetMatrix[0].size();

	//Search mininum value in the matrix linearly
	for (int row = 0; row < imgHeight; row++) {
		for (int col = 0; col < imgWidth; col++) {
			if (targetMatrix[row][col][0] < currentMin) {
				currentMin = targetMatrix[row][col][0];
			}
		}
	}
	minInGivenMatrix = currentMin;
	return minInGivenMatrix;
}

//Normalize a given matrix using minimum and maximum values in the matrix
std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::minMaxNormalizer(std::vector<std::vector<std::vector<double>>> targetMatrix) {
	//find minimum and maximum values in the matrix
	double minInMatrix = this->minFinder(targetMatrix);
	double maxInMatrix = this->maxFinder(targetMatrix);
	
	int imgHeight = targetMatrix.size();
	int imgWidth = targetMatrix[0].size();

	double maxIntensity = 255;

	std::vector<std::vector<std::vector<double>>> normalizedMatrix = targetMatrix;

	for (int row = 0; row < imgHeight; row++) {
		for (int col = 0; col < imgWidth; col++) {
			//normalize value of pixel using max-min normalization technique. Refer to disscussion in D2L website
			normalizedMatrix[row][col][0] = maxIntensity * (targetMatrix[row][col][0] - minInMatrix) / (maxInMatrix - minInMatrix);
		}
	}

	return normalizedMatrix;


	
}



//return extended 3D matrix
//please perform this before apply any filter
std::vector<std::vector<std::vector<double>>>  ImageProcessorHW2::boundaryExtension(std::vector<std::vector<double>> filter) {
	int filterSize = filter.size(); //size of column or row of the filter, ex) 5*5 size filter => return value would be 5
	int amountOfExtension = filterSize / 2;  //ex) filter size 5*5 => 5/2 = 2 => one side of matrix size will be extended by 2
	int extendedHeight = 2 * amountOfExtension + imgHeight; //ex) demension of original matrix = 30*30 => demension of expended matrix by 5*5 size filter(4+30)*(4+30) 
	int extendedWidth = 2 * amountOfExtension + imgWidth;

	std::vector<double> extended_1D(imgBytesPerPixel);
	std::vector<std::vector<double>> extended_2D(extendedWidth, extended_1D);
	std::vector<std::vector<std::vector<double>>> extended_3D(extendedHeight, extended_2D);

	for (int col = 0; col < extendedWidth; col++) {
		for (int row = 0; row < extendedHeight; row++) {
			if (col < amountOfExtension) {
				extended_3D[row][col][0] = 0;
			}
			else if (col >= extendedWidth - amountOfExtension) {
				extended_3D[row][col][0] = 0;
			}
			else if (row < amountOfExtension) {
				extended_3D[row][col][0] = 0;
			}
			else if (row >= extendedHeight - amountOfExtension) {
				extended_3D[row][col][0] = 0;
			}

			else {
				extended_3D[row][col][0] = inputImg[row - amountOfExtension][col - amountOfExtension][0];
			}
		}

	}

	return extended_3D;
}


//return extended 3D matrix Version2
//please perform this before apply any filter
std::vector<std::vector<std::vector<double>>>  ImageProcessorHW2::boundaryExtensionV2(std::vector<std::vector<std::vector<double>>> inputImg ,std::vector<std::vector<double>> filter) {
	int imgHeight = inputImg.size();
	int imgWidth = inputImg[0].size();
	int imgBytesPerPixel = 1;//gray scale
	int filterSize = filter.size(); //size of column or row of the filter, ex) 5*5 size filter => return value would be 5
	int amountOfExtension = filterSize / 2;  //ex) filter size 5*5 => 5/2 = 2 => one side of matrix size will be extended by 2
	int extendedHeight = 2 * amountOfExtension + imgHeight; //ex) demension of original matrix = 30*30 => demension of expended matrix by 5*5 size filter(4+30)*(4+30) 
	int extendedWidth = 2 * amountOfExtension + imgWidth;

	std::vector<double> extended_1D(imgBytesPerPixel);
	std::vector<std::vector<double>> extended_2D(extendedWidth, extended_1D);
	std::vector<std::vector<std::vector<double>>> extended_3D(extendedHeight, extended_2D);

	for (int col = 0; col < extendedWidth; col++) {
		for (int row = 0; row < extendedHeight; row++) {
			if (col < amountOfExtension) {
				extended_3D[row][col][0] = 0;
			}
			else if (col >= extendedWidth - amountOfExtension) {
				extended_3D[row][col][0] = 0;
			}
			else if (row < amountOfExtension) {
				extended_3D[row][col][0] = 0;
			}
			else if (row >= extendedHeight - amountOfExtension) {
				extended_3D[row][col][0] = 0;
			}

			else {
				extended_3D[row][col][0] = inputImg[row - amountOfExtension][col - amountOfExtension][0];
			}
		}

	}

	return extended_3D;
}






//convolution. Apply a given filter to extended matrix and return a filtered matrix
std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::convolution(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> extendedMatrix) {
	int filterSize = filter.size();
	int amountOfExtension = filterSize / 2;
	int extendedImgHeight = extendedMatrix.size();
	int extendedImgWidth = extendedMatrix[0].size();

	std::vector<std::vector<std::vector<double>>> filteredMatrix = extendedMatrix;


	int row_counter = 0;

	for (int row = amountOfExtension; row < extendedImgHeight - amountOfExtension; row++) {
		int col_counter = 0;

		for (int col = amountOfExtension; col < extendedImgWidth - amountOfExtension; col++) {
			//elementwise multiplication between filter and extended matrix
			double tem_val = 0;
			double tem_normalization = 0;
			for (int filter_row = 0; filter_row < filterSize; filter_row++) {
				for (int filter_col = 0; filter_col < filterSize; filter_col++) {
					tem_val = tem_val + filter[filter_row][filter_col] * extendedMatrix[filter_row + row_counter][filter_col + col_counter][imgBytesPerPixel - 1];
					tem_normalization = tem_normalization + filter[filter_row][filter_col];
				}
			}

			filteredMatrix[row][col][imgBytesPerPixel - 1] = tem_val / tem_normalization;

			col_counter++;

		}

		row_counter++;

	}

	return filteredMatrix;

}


//convolution Version2. with no normalization. Apply a given filter to extended matrix and return a filtered matrix
std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::convolutionV2(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> extendedMatrix) {
	int filterSize = filter.size();
	int amountOfExtension = filterSize / 2;
	int extendedImgHeight = extendedMatrix.size();
	int extendedImgWidth = extendedMatrix[0].size();

	std::vector<std::vector<std::vector<double>>> filteredMatrix = extendedMatrix;


	int row_counter = 0;

	for (int row = amountOfExtension; row < extendedImgHeight - amountOfExtension; row++) {
		int col_counter = 0;

		for (int col = amountOfExtension; col < extendedImgWidth - amountOfExtension; col++) {
			//elementwise multiplication between filter and extended matrix
			double tem_val = 0;
			
			for (int filter_row = 0; filter_row < filterSize; filter_row++) {
				for (int filter_col = 0; filter_col < filterSize; filter_col++) {
					tem_val = tem_val + filter[filter_row][filter_col] * extendedMatrix[filter_row + row_counter][filter_col + col_counter][imgBytesPerPixel - 1];
					
				}
			}

			filteredMatrix[row][col][imgBytesPerPixel - 1] = tem_val;

			col_counter++;

		}

		row_counter++;

	}

	return filteredMatrix;

}



//crop a given expended matrix depending on a given filter and return the cropped matrix
std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::matrixCropper(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> extendedMatrix) {
	int filterSize = filter.size();
	int amountOfExtension = filterSize / 2;
	int extendedImgHeight = extendedMatrix.size();
	int extendedImgWidth = extendedMatrix[0].size();

	int croppedHeight = extendedImgHeight - 2 * amountOfExtension;
	int croopedWidth = extendedImgWidth - 2 * amountOfExtension;

	std::vector<double> cropped_1D(imgBytesPerPixel);
	std::vector<std::vector<double>> cropped_2D(croopedWidth, cropped_1D);
	std::vector<std::vector<std::vector<double>>> cropped_3D(croppedHeight, cropped_2D);

	for (int row = 0; row < croppedHeight; row++) {
		for (int col = 0; col < croopedWidth; col++) {
			cropped_3D[row][col][imgBytesPerPixel - 1] = extendedMatrix[row + amountOfExtension][col + amountOfExtension][imgBytesPerPixel - 1];
		}
	}

	return cropped_3D;
}


//Post processing. Handle Negative values and Large values greated during any processing manual mapping e.g. negative -> 0, Bigger than 255 -> 255 
std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::pixelValuePostProcessor(std::vector<std::vector<std::vector<double>>> processedMatrix) {
	int imgHeight = processedMatrix.size();
	int imgWidth = processedMatrix[0].size();

	std::vector<std::vector<std::vector<double>>> postProcessedImg = processedMatrix;//initialize with 3D data image variable.

	//**Pre Processing Handle overflow issue between unsigned char and double type mannually

	for (int row = 0; row < imgHeight; row++) {
		for (int col = 0; col < imgWidth; col++) {
			if (processedMatrix[row][col][0] < 0) {
				postProcessedImg[row][col][0] = 0.0;
			}
			else if (processedMatrix[row][col][0] > 255.0) {
				postProcessedImg[row][col][0] = 255.0;
			}

		}
	}

	return postProcessedImg;
}


std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::RGB2Gray(std::vector<std::vector<std::vector<double>>> redChannel, std::vector<std::vector<std::vector<double>>> greenChannel, std::vector<std::vector<std::vector<double>>>blueChannel) {
	int imgHeight = redChannel.size();
	int imgWidth = redChannel[0].size();
	int imgBytePerPixel = 1; // 1 for gray scale image

	
	std::vector<std::vector<std::vector<double>>> grayScale_3D = redChannel; //initialize gray scale 3 D data variable

	for (int row = 0; row < imgHeight; row++) {
		for (int col = 0; col < imgWidth; col++) {
			grayScale_3D[row][col][imgBytePerPixel - 1] = 0.2989 * redChannel[row][col][imgBytePerPixel - 1]
				+ 0.5870 * greenChannel[row][col][imgBytePerPixel - 1]
				+ 0.1140 * blueChannel[row][col][imgBytePerPixel - 1];
		}
	}

	return grayScale_3D;



}


std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::xDirectionGradientEdgeMapGenerator(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> oriImgData) {
	//Extend original input image based on the given filter
	std::vector<std::vector<std::vector<double>>> extendedImg = this->boundaryExtension(filter);

	//perform convolution with the given filter
	std::vector<std::vector<std::vector<double>>> filteredImg = this->convolutionV2(filter, extendedImg);

	//Crop the filtered image above to original size
	std::vector<std::vector<std::vector<double>>> croppedFilteredImg  = this->matrixCropper(filter, filteredImg);


	//normalize Gx values
	std::vector<std::vector<std::vector<double>>> normalizedFilteredImg = this->minMaxNormalizer(croppedFilteredImg);

	


	/*
	//test values
	std::cout << "test values" << std::endl;
	
	for (int row = 0; row < final_xDirectionGradientEdgeMap.size(); row++) {
		for (int col = 0; col < final_xDirectionGradientEdgeMap[0].size(); col++) {
			std::cout << "current value: " << final_xDirectionGradientEdgeMap[row][col][0] << std::endl;

			if (final_xDirectionGradientEdgeMap[row][col][0] < -255.0) {
				std::cout << "Big negative value: " << final_xDirectionGradientEdgeMap[row][col][0] << std::endl;
			}
			else if (final_xDirectionGradientEdgeMap[row][col][0] > 255.0) {
				std::cout <<"Big positive value: " << final_xDirectionGradientEdgeMap[row][col][0] << std::endl;
			}

		}
	}
	*/


	return normalizedFilteredImg;
}

std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::yDirectionGradientEdgeMapGenerator(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> oriImgData) {
	//Extend original input image based on the given filter
	std::vector<std::vector<std::vector<double>>> extendedImg = this->boundaryExtension(filter);

	//perform convolution with the given filter
	std::vector<std::vector<std::vector<double>>> filteredImg = this->convolutionV2(filter, extendedImg);

	//Crop the filtered image above to original size
	std::vector<std::vector<std::vector<double>>> croppedFilteredImg = this->matrixCropper(filter, filteredImg);


	//normalize Gx values
	std::vector<std::vector<std::vector<double>>> normalizedFilteredImg = this->minMaxNormalizer(croppedFilteredImg);

	
	
	

	return normalizedFilteredImg;
}


//Generate normalized gradient magnitude map
std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::gradientMagnitudeEdgeMapGenerator(std::vector<std::vector<double>> xDirectionFilter, std::vector<std::vector<double>> yDirectionFilter, std::vector<std::vector<std::vector<double>>> oriImgData) {
	//perform x direction gradient
	//Extend original input image based on the given filter
	std::vector<std::vector<std::vector<double>>> extendedImg_X = this->boundaryExtension(xDirectionFilter);

	//perform convolution with the given filter
	std::vector<std::vector<std::vector<double>>> filteredImg_X = this->convolutionV2(xDirectionFilter, extendedImg_X);

	//Crop the filtered image above to original size
	std::vector<std::vector<std::vector<double>>> final_xDirectionGradientEdgeMap = this->matrixCropper(xDirectionFilter, filteredImg_X);


	//perform y direction gradient
	//Extend original input image based on the given filter
	std::vector<std::vector<std::vector<double>>> extendedImg_Y = this->boundaryExtension(yDirectionFilter);

	//perform convolution with the given filter
	std::vector<std::vector<std::vector<double>>> filteredImg_Y = this->convolutionV2(yDirectionFilter, extendedImg_Y);

	//Crop the filtered image above to original size
	std::vector<std::vector<std::vector<double>>> final_yDirectionGradientEdgeMap = this->matrixCropper(yDirectionFilter, filteredImg_Y);


	//Generate gradient magnitude edge map
	std::vector<std::vector<std::vector<double>>> gradientMagnitudeEdgeMap = final_xDirectionGradientEdgeMap;//initialize 3D img data with one of filtered gradient edge map. Can use either direction edge map

	for (int row = 0; row < gradientMagnitudeEdgeMap.size(); row++) {
		for (int col = 0; col < gradientMagnitudeEdgeMap[0].size(); col++) {
			//Calculate gradient magnitude and save the value at corresponding pixel
			gradientMagnitudeEdgeMap[row][col][0] = sqrt(pow(final_xDirectionGradientEdgeMap[row][col][0],2.0)+pow(final_yDirectionGradientEdgeMap[row][col][0], 2.0));
		}
	}

	//Post processing
	std::vector<std::vector<std::vector<double>>> preprocessed_gradientMagnitudeEdgeMap = this->pixelValuePostProcessor(gradientMagnitudeEdgeMap);

	/*
	//test values
	std::cout << "test values" << std::endl;

	for (int row = 0; row < gradientMagnitudeEdgeMap.size(); row++) {
		for (int col = 0; col < gradientMagnitudeEdgeMap[0].size(); col++) {
			std::cout << "current value: " << gradientMagnitudeEdgeMap[row][col][0] << std::endl;

			

		}
	}
	*/
	return preprocessed_gradientMagnitudeEdgeMap;
}

std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::tunnedAndBinaryEdgeMapGenerator(std::vector<std::vector<std::vector<double>>> gradientMagnitudeEdgeMap , double targetThreshold) {
	//find cumulative histogram in terms of percentage (0~1) scale e.g. 0=>0%, 1=>100%
	std::vector<int> countedIntensitiesStorage = this->intensityCounterOneChannel(gradientMagnitudeEdgeMap);
	std::vector<double> probabilityVector = this->normalizedProbCalculator(countedIntensitiesStorage);
	std::vector<double> CDFVector = this->CDFCalculator(probabilityVector);

	


	double scaledTargetThreshold = 0.01 * targetThreshold; //scale target threshold within (0,1) range
	double targetIntensity = 0.0; //This will be used as threshold for tunning
	//find target intensity that will be used as threshold for tunning 
	for (int index = 0; index < CDFVector.size(); index++) {
		if (CDFVector[index] > scaledTargetThreshold) {
			targetIntensity = static_cast<double>(index); //index indicates intensity
			std::cout << "Target intensity is found" << std::endl;
			break;
		}
	}

	//tunning and binarizing edge map
	std::vector<std::vector<std::vector<double>>> tunnedAndBinarizedEdgeMap = gradientMagnitudeEdgeMap;//initialization
	//Note that in this problem, edge corresponds to value 0, and background corresponds to value 255
	double edgeIntensity = 0.0;
	double backGroundIntensity = 255.0;
	for (int row = 0; row < tunnedAndBinarizedEdgeMap.size(); row++) {
		for (int col = 0; col < tunnedAndBinarizedEdgeMap[0].size(); col++) {
			//consider as edge. Note that here edge corresponds to value 0
			if (tunnedAndBinarizedEdgeMap[row][col][0]>targetIntensity) {
				tunnedAndBinarizedEdgeMap[row][col][0] = edgeIntensity;
			}
			//consider as background. Note that here background corresponds to value 255
			else if(tunnedAndBinarizedEdgeMap[row][col][0] < targetIntensity){
				tunnedAndBinarizedEdgeMap[row][col][0] = backGroundIntensity;
			}
		}
	}

	


	return tunnedAndBinarizedEdgeMap;

}

//Generate a tunned edge map by threshold value of gradient magnitude
std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::tunnedAndBinaryEdgeMapGeneratorByMaxGradient(std::vector<std::vector<std::vector<double>>> gradientMagnitudeEdgeMap, double targetThreshold) {
	double maxGradient = this->maxFinder(gradientMagnitudeEdgeMap);//find maximum gradient magnitude in gradient magnitude edge map



	double scaledTargetThreshold = 0.01 * targetThreshold; //scale target threshold within (0,1) range
	double targetIntensity = maxGradient*scaledTargetThreshold; //This will be used as threshold for tunning
	

	//tunning and binarizing edge map
	std::vector<std::vector<std::vector<double>>> tunnedAndBinarizedEdgeMap = gradientMagnitudeEdgeMap;//initialization
	//Note that, in this problem, edge corresponds to value 0, and background corresponds to value 255
	double edgeIntensity = 0.0;
	double backGroundIntensity = 255.0;
	for (int row = 0; row < tunnedAndBinarizedEdgeMap.size(); row++) {
		for (int col = 0; col < tunnedAndBinarizedEdgeMap[0].size(); col++) {
			//consider as edge. Note that here edge corresponds to value 0
			if (tunnedAndBinarizedEdgeMap[row][col][0] > targetIntensity) {
				tunnedAndBinarizedEdgeMap[row][col][0] = edgeIntensity;
			}
			//consider as background. Note that here background corresponds to value 255
			else if (tunnedAndBinarizedEdgeMap[row][col][0] < targetIntensity) {
				tunnedAndBinarizedEdgeMap[row][col][0] = backGroundIntensity;
			}
		}
	}




	return tunnedAndBinarizedEdgeMap;

}






//*********************Class Methods from Histogram Manipulation. They will be used for tunning and binarizing edge map

//Count number of each intensity in one channel and return a corresponding 1D vector that includes the number of appearance of each intensity
std::vector<int> ImageProcessorHW2::intensityCounterOneChannel(std::vector<std::vector<std::vector<double>>> oneChannel) {
	int maxValOfIntensity = 255;
	int zero = 0;
	int bytePerPixel = 1;
	int one = 1;

	//Initialize size of 256 1D Vector with zeros 
	std::vector<int> one_channel_vec(maxValOfIntensity + one, zero); // 1D vector that will have number of appearance of each intensity for one channel 2D image matrix data




	//counting frequency of each intensity
	for (int row = 0; row < imgHeight; row++) {
		for (int col = 0; col < imgWidth; col++) {
			int intensityVal = static_cast<int>(oneChannel[row][col][bytePerPixel - 1]);//be careful with index

			one_channel_vec[intensityVal] = one_channel_vec[intensityVal] + 1; //1 means it increases appearance of the intensity by 1.

		}
	}

	//return 1D vector that has frequency of intensities(0~255) for each channel
	return one_channel_vec;

}

//Calculate Normalized Probability and return a 1D vector that includes normalized probability for each intensity
std::vector<double> ImageProcessorHW2::normalizedProbCalculator(std::vector<int> intensitiesOneChannel) {
	int maxValOfIntensity = 255;
	double zero = 0.0;
	int one = 1;

	//Initialize size of 256 1D Vector with zeros 
	std::vector<double> probabilityVector(maxValOfIntensity + one, zero);

	double totalNumPixels = static_cast<double>(imgHeight)* static_cast<double>(imgWidth); //Need to make sure this object is created by right image, so then it will be run with correct imgHeight and imgWidth


	for (int intensity = 0; intensity <= maxValOfIntensity; intensity++) {
		probabilityVector[intensity] = static_cast<double>(intensitiesOneChannel[intensity]) / (totalNumPixels);
	}

	return probabilityVector;
}


//Calculate Cummulative Density Function(CDF) for each intensity and return a 1D vector than includes CDF for each intensity
std::vector<double> ImageProcessorHW2::CDFCalculator(std::vector<double> normalizedProbVector) {
	int maxValOfIntensity = 255;
	int one = 1;
	double zero = 0.0;

	//Initialize size of 256 1D Vector with zero
	std::vector<double> CDFVector(maxValOfIntensity + one, zero);


	for (int index = 0; index <= maxValOfIntensity; index++) {

		if (index == 0) {
			CDFVector[index] = normalizedProbVector[index];
		}

		else {
			CDFVector[index] = normalizedProbVector[index] + CDFVector[index - 1]; //add previous value to current value, so it becomes CDF for current index(intensity)
		}





	}

	return CDFVector;
}


//***************************Class Methods for Problem 2 Digital Halftoning

//Generate a halftoned image by fixed thresholding from a gray scale image
std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::fixedThresholdedImgGenerator(std::vector<std::vector<std::vector<double>>> inputImg, double targetThreshold) {
	int imgHeight = inputImg.size();
	int imgWidth = inputImg[0].size();
	double maxIntensity = 255;
	double minIntensity = 0;
	std::vector<std::vector<std::vector<double>>> halfTonedByFixedThresholdImg = inputImg; //initialize output image

	for (int row = 0; row < imgHeight; row++) {
		for (int col = 0; col < imgWidth; col++) {
			if (inputImg[row][col][0] >= targetThreshold) {
				halfTonedByFixedThresholdImg[row][col][0] = maxIntensity;
			}
			else if(inputImg[row][col][0] < targetThreshold) {
				halfTonedByFixedThresholdImg[row][col][0] = minIntensity;
			}
		}
	}

	return halfTonedByFixedThresholdImg;

	
}


//Generate a halftoned image by random thresholding from a gray scale image
std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::randomThresholdedImgGenerator(std::vector<std::vector<std::vector<double>>> inputImg) {
	int imgHeight = inputImg.size();
	int imgWidth = inputImg[0].size();
	double maxIntensity = 255;
	double minIntensity = 0;
	int one = 1;
	int maxIntensityInteger = 255;
	std::vector<std::vector<std::vector<double>>> halfTonedByRandomThresholdImg = inputImg; //initialize output image
	
	

	for (int row = 0; row < imgHeight; row++) {
		for (int col = 0; col < imgWidth; col++) {
			double randomThreshold = static_cast<double>( (rand()% maxIntensityInteger)+one ); //Generate random number between 1 ~ 255 and cast it to double type
			if (inputImg[row][col][0] >= randomThreshold) {
				halfTonedByRandomThresholdImg[row][col][0] = maxIntensity;
			}
			else if (inputImg[row][col][0] < randomThreshold) {
				halfTonedByRandomThresholdImg[row][col][0] = minIntensity;
			}
		}
	}

	return halfTonedByRandomThresholdImg;

}

//Generate a halftoned image by a given dithering matrix(i.e., thresholding matrix)
std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::ditheringMatrixImgGenerator(std::vector<std::vector<std::vector<double>>> inputImg, std::vector<std::vector<double>> thresholdMatrix) {
	int imgHeight = inputImg.size();
	int imgWidth = inputImg[0].size();
	
	int sizeOfThresholdMatrix = thresholdMatrix.size();
	double minIntensity = 0.0;
	double maxIntensity = 255.0;
	std::vector<std::vector<std::vector<double>>> thresholdedImg = inputImg; //initialize output image

	for (int row = 0; row < imgHeight; row++) {
		for (int col = 0; col < imgWidth; col++) {
			int row_thresholdMatrix = row % sizeOfThresholdMatrix;
			int col_thresholdMatrix = col % sizeOfThresholdMatrix;

			if (inputImg[row][col][0] <= thresholdMatrix[row_thresholdMatrix][col_thresholdMatrix]) {
				thresholdedImg[row][col][0] = minIntensity;
			}
			else if(inputImg[row][col][0] > thresholdMatrix[row_thresholdMatrix][col_thresholdMatrix]){
				thresholdedImg[row][col][0] = maxIntensity;
			}
		}
	}
	
	



	return thresholdedImg;

}



//***Error Diffusion
	//Generate a halftoned image by a provided error diffusion mask with the serpentine scanning
std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::halftonedImgGenerator_ErrorDiffusion(std::vector<std::vector<std::vector<double>>> extendedInputImg, std::vector<std::vector<double>> left2Right_errorDiffusionMatrix, std::vector<std::vector<double>> right2Left_errorDiffusionMatrix) {
	int filterSize = left2Right_errorDiffusionMatrix.size();
	int amountOfExtension = filterSize / 2;
	int extendedImgHeight = extendedInputImg.size();
	int extendedImgWidth = extendedInputImg[0].size();

	int bytePerPixel = 1; //gray scale
	int outputImgHeight = extendedImgHeight - 2 * amountOfExtension;
	int outputImgWidth = extendedImgWidth - 2 * amountOfExtension;
	
	double threshold = 127;//this is a typical threshold.
	double maxIntensity = 255;
	double minIntensity = 0;

	std::vector<std::vector<std::vector<double>>> processingMatrix = extendedInputImg; //copy matrix for processing
	
	std::vector<double> binaryHalfTonedImg_1D(bytePerPixel);
	std::vector<std::vector<double>> binaryHalfTonedImg_2D(outputImgWidth, binaryHalfTonedImg_1D);
	std::vector<std::vector<std::vector<double>>> binaryHalfTonedImg_3D(outputImgHeight, binaryHalfTonedImg_2D);

	//need row_counter and col_counter to index processingMatrix correctly
	int row_counter = 0;

	for (int row = amountOfExtension; row < extendedImgHeight - amountOfExtension; row++) {
		
		int col_counter = 0;
		//row even case. processing direction is forward	
		if (row % 2 == 1) {
			//Column starts from the beginning to the end
			for (int col = amountOfExtension; col < extendedImgWidth - amountOfExtension; col++) {
				//initialization estimated f_ij
				double current_estimated_f_ij = processingMatrix[row][col][0];
				
				//Binarize estimated f_ij
				if (current_estimated_f_ij > threshold) {
					binaryHalfTonedImg_3D[row - amountOfExtension][col - amountOfExtension][0] = maxIntensity;
				}
				else {
					binaryHalfTonedImg_3D[row - amountOfExtension][col - amountOfExtension][0] = minIntensity;
				}
				//Calculate current error
				double current_Error = current_estimated_f_ij - binaryHalfTonedImg_3D[row - amountOfExtension][col - amountOfExtension][0];

			
				for (int filter_row = 0; filter_row < filterSize; filter_row++) {
					for (int filter_col = 0; filter_col < filterSize; filter_col++) {
						//Check the formula in Discussion note
						processingMatrix[filter_row + row_counter][filter_col + col_counter][0] = processingMatrix[filter_row + row_counter][filter_col + col_counter][0] + current_Error * left2Right_errorDiffusionMatrix[filter_row][filter_col];
					}
				}
			
				col_counter++;
				
			}
			
			

		}

		//row odd case. processing direction is backward	
		else if (row % 2 == 0) {
			
			//Column starts from the end to the beginning
			for (int col = extendedImgWidth - amountOfExtension -1 ; col >= amountOfExtension; col--) {
				//initialization estimated f_ij
				double current_estimated_f_ij = processingMatrix[row][col][0];
				

				//Binarize estimated f_ij
				if (current_estimated_f_ij > threshold) {
					binaryHalfTonedImg_3D[row - amountOfExtension][col - amountOfExtension][0] = maxIntensity;
				}
				else {
					binaryHalfTonedImg_3D[row - amountOfExtension][col - amountOfExtension][0] = minIntensity;
				}
				//Calculate current error
				double current_Error = current_estimated_f_ij - binaryHalfTonedImg_3D[row - amountOfExtension][col - amountOfExtension][0];


				for (int filter_row = 0; filter_row < filterSize; filter_row++) {
					for (int filter_col = 0; filter_col < filterSize; filter_col++) {
						//Check the formula in Discussion note
						processingMatrix[filter_row + row_counter][filter_col + col_counter][0] = processingMatrix[filter_row + row_counter][filter_col + col_counter][0] + current_Error * right2Left_errorDiffusionMatrix[filter_row][filter_col];
					}
				}
				col_counter++;
			}

			
		
		}
		row_counter++;

	}
	
	return binaryHalfTonedImg_3D;
}


//***Error Diffusion
	//Generate a halftoned image by a provided error diffusion mask with the serpentine scanning
std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::halftonedImgGenerator_ErrorDiffusionV2(std::vector<std::vector<std::vector<double>>> inputImg, std::vector<std::vector<double>> left2Right_errorDiffusionMatrix, std::vector<std::vector<double>> right2Left_errorDiffusionMatrix) {
	int filterSize = left2Right_errorDiffusionMatrix.size();
	int imgHeight = inputImg.size();
	int imgWidth = inputImg[0].size();
	int bytePerPixel = 1; //gray scale
	int amountOfExtension = filterSize / 2;

	double threshold = 127;//this is a typical threshold.
	double maxIntensity = 255;
	double minIntensity = 0;

	std::vector<std::vector<std::vector<double>>> processingMatrix = inputImg; //copy matrix for processing

	std::vector<double> binaryHalfTonedImg_1D(bytePerPixel);
	std::vector<std::vector<double>> binaryHalfTonedImg_2D(imgWidth, binaryHalfTonedImg_1D);
	std::vector<std::vector<std::vector<double>>> binaryHalfTonedImg_3D(imgHeight, binaryHalfTonedImg_2D);

	for (int row = 0; row < imgHeight; row++) {
		//Forward error diffusion
		if ((row % 2) == 0) {
			for (int col = 0; col < imgWidth; col++) {
				//binarization
				if(processingMatrix[row][col][0]>=threshold){
					binaryHalfTonedImg_3D[row][col][0] = maxIntensity;
				}
				else {
					binaryHalfTonedImg_3D[row][col][0] = minIntensity;
				}

				//calculate error
				double current_error = processingMatrix[row][col][0] - binaryHalfTonedImg_3D[row][col][0];
				
				//error diffusion
				//filter row range -1 to 1 (indexing purpose) true index range is (0~2) for 3x3 filter
				for (int filter_row = -amountOfExtension; filter_row <= amountOfExtension; filter_row++) {
					for (int filter_col = -amountOfExtension; filter_col <= amountOfExtension; filter_col++) {
						int current_row = row + filter_row;
						int current_col = col + filter_col;
						int true_filter_row = filter_row + amountOfExtension;
						int true_filter_col = filter_col + amountOfExtension;

						//perform error diffusion only for the pixel in proper index range
						if (current_row >= 0 && current_row < imgHeight && current_col >= 0 && current_col < imgWidth) {
							
							processingMatrix[current_row][current_col][0] = processingMatrix[current_row][current_col][0] + current_error * left2Right_errorDiffusionMatrix[true_filter_row][true_filter_col];
							
						}
					}
				}
			}
		
		}
		//Backward error diffusion
		else if ((row % 2) == 1) {
			for (int col = imgWidth - 1; col >= 0; col--) {
				//binarization
				if (processingMatrix[row][col][0] >= threshold) {
					binaryHalfTonedImg_3D[row][col][0] = maxIntensity;
				}
				else {
					binaryHalfTonedImg_3D[row][col][0] = minIntensity;
				}

				//calculate error
				double current_error = processingMatrix[row][col][0] - binaryHalfTonedImg_3D[row][col][0];
				
				//error diffusion
				//filter row range -1 to 1 (indexing purpose) true index range is (0~2) for 3x3 filter
				for (int filter_row = -amountOfExtension; filter_row <= amountOfExtension; filter_row++) {
					for (int filter_col = -amountOfExtension; filter_col <= amountOfExtension; filter_col++) {
						int current_row = row + filter_row;
						int current_col = col + filter_col;
						int true_filter_row = filter_row + amountOfExtension;
						int true_filter_col = filter_col + amountOfExtension;

						//perform error diffusion only for the pixel in proper index range
						if (current_row >= 0 && current_row < imgHeight && current_col >= 0 && current_col < imgWidth) {
							processingMatrix[current_row][current_col][0] = processingMatrix[current_row][current_col][0] + current_error * right2Left_errorDiffusionMatrix[true_filter_row][true_filter_col];
							
							
						}


					}
				}
			}
		}
	
	
	}

	

	return binaryHalfTonedImg_3D;
}

//universal converter for a given one channel 
//e.g. Red->Cyan or Green->Magenta or Blue->Yellow or vice versa
std::vector<std::vector<std::vector<double>>> ImageProcessorHW2::oneChannelRGB2CMYConverter(std::vector<std::vector<std::vector<double>>> oneChannelImg) {
	int channelHeight = oneChannelImg.size();
	int channelWidth = oneChannelImg[0].size();
	double maxIntensity = 255;
	//Initialization converted one channel img
	std::vector<std::vector<std::vector<double>>> convertedOneChannel = oneChannelImg;

	//Conversion processing
	for (int row = 0; row < channelHeight; row++) {
		for (int col = 0; col < channelWidth; col++) {
			//This relationship is from discussion week4 note
			convertedOneChannel[row][col][0] = maxIntensity - oneChannelImg[row][col][0];
		}
	}

	return convertedOneChannel;



}


//***Color Image Error Diffusion
//Generate a halftoned 3 channels by a provided error diffusion mask with the serpentine scanning
std::tuple< std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>> > ImageProcessorHW2::halftonedImgGenerator_ErrorDiffusion_Color(std::vector<std::vector<std::vector<double>>> redChannel, std::vector<std::vector<std::vector<double>>> greenChannel, std::vector<std::vector<std::vector<double>>> blueChannel, std::vector<std::vector<double>> left2Right_errorDiffusionMatrix, std::vector<std::vector<double>> right2Left_errorDiffusionMatrix) {

	//Convert each channels
	//Red - > Cyan
	std::vector<std::vector<std::vector<double>>> cyanChannel = this->oneChannelRGB2CMYConverter(redChannel);
	//Green -> Magenta
	std::vector<std::vector<std::vector<double>>> magentaChannel = this->oneChannelRGB2CMYConverter(greenChannel);
	//Blue -> Yellow
	std::vector<std::vector<std::vector<double>>> yellowChannel = this->oneChannelRGB2CMYConverter(blueChannel);


	
	//Perfom error diffusion 
	//**Note in error diffusion method, it has the same performance as cropping matrix while make an binary matrix
	//Cyan
	std::vector<std::vector<std::vector<double>>> tonned_extended_Cyan = this->halftonedImgGenerator_ErrorDiffusionV2(cyanChannel, left2Right_errorDiffusionMatrix, right2Left_errorDiffusionMatrix);

	//Magenta
	std::vector<std::vector<std::vector<double>>> tonned_extended_Magenta = this->halftonedImgGenerator_ErrorDiffusionV2(magentaChannel, left2Right_errorDiffusionMatrix, right2Left_errorDiffusionMatrix);

	//Yellow
	std::vector<std::vector<std::vector<double>>> tonned_extended_Yellow = this->halftonedImgGenerator_ErrorDiffusionV2(yellowChannel, left2Right_errorDiffusionMatrix, right2Left_errorDiffusionMatrix);

	
	//Convert Back to R, G, B
	//Cyan -> Red
	std::vector<std::vector<std::vector<double>>> final_red = this->oneChannelRGB2CMYConverter(tonned_extended_Cyan);
	//Magenta -> Green
	std::vector<std::vector<std::vector<double>>> final_green = this->oneChannelRGB2CMYConverter(tonned_extended_Magenta);
	//Yellow -> Blue
	std::vector<std::vector<std::vector<double>>> final_blue = this->oneChannelRGB2CMYConverter(tonned_extended_Yellow);

	return std::make_tuple(final_red, final_green, final_blue);
}

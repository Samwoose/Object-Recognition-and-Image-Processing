#include "ImageProcessor.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <math.h>


ImageProcessor::ImageProcessor(int height,int width, int BytesPerPixel, std::vector<std::vector<std::vector<double>>> inputImage) {
    imgHeight = height;
	imgWidth = width;
	imgBytesPerPixel = BytesPerPixel;
	
	inputImg = inputImage;
	
}
ImageProcessor::~ImageProcessor() {}

//return extended 3D matrix
//please perform this before apply any filter
std::vector<std::vector<std::vector<double>>>  ImageProcessor::boundaryExtension(std::vector<std::vector<double>> filter) {
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
				extended_3D[row][col][imgBytesPerPixel - 1] = 0;
			}
			else if (col >= extendedWidth - amountOfExtension) {
				extended_3D[row][col][imgBytesPerPixel - 1] = 0;
			}
			else if (row < amountOfExtension) {
				extended_3D[row][col][imgBytesPerPixel - 1] = 0;
			}
			else if (row >= extendedHeight - amountOfExtension) {
				extended_3D[row][col][imgBytesPerPixel - 1] = 0;
			}
			
			else {
				extended_3D[row][col][imgBytesPerPixel - 1] = inputImg[row - amountOfExtension][col - amountOfExtension][imgBytesPerPixel - 1];
			}
		}
	
	}

	return extended_3D;
}


//convolution. Apply a given filter to extended matrix and return a filtered matrix
std::vector<std::vector<std::vector<double>>> ImageProcessor::convolution(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> extendedMatrix) {
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
					tem_val = tem_val + filter[filter_row][filter_col] * extendedMatrix[filter_row+row_counter][filter_col+col_counter][imgBytesPerPixel - 1];
					tem_normalization = tem_normalization + filter[filter_row][filter_col];
				}
			}

			filteredMatrix[row][col][imgBytesPerPixel-1] = tem_val/ tem_normalization;

			col_counter++;
			
		}

		row_counter++;
		
	}

	return filteredMatrix;

}


//crop a given expended matrix depending on a given filter and return the cropped matrix
std::vector<std::vector<std::vector<double>>> ImageProcessor::matrixCropper(std::vector<std::vector<double>> filter, std::vector<std::vector<std::vector<double>>> extendedMatrix) {
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


double ImageProcessor::PSNRcalculator(std::vector<std::vector<std::vector<double>>> originImage, std::vector<std::vector<std::vector<double>>> filteredImage) {
	int imgWidth = originImage[0].size();
	int imgHeight = originImage.size();

	double MSE = 0.0;
	double PSNR = 0.0; //peak-signal-to-noise (dB) 
	double MAX = 255.0;

	//Check formula on HW1 page 5.
	for (int row = 0; row < imgHeight  ; row++) {
		for (int col = 0; col < imgWidth; col++) {
			MSE = MSE + (filteredImage[row][col][imgBytesPerPixel - 1] - originImage[row][col][imgBytesPerPixel - 1])* (filteredImage[row][col][imgBytesPerPixel - 1] - originImage[row][col][imgBytesPerPixel - 1]);
		}
	}
	
	MSE = MSE * (1.0 / imgHeight) * (1.0 / imgWidth);

	PSNR = 10 * log10(MAX * MAX / MSE);

	return PSNR;


}

std::vector<std::vector<std::vector<double>>> ImageProcessor::BilateralDenoising(std::vector<std::vector<std::vector<double>>> extendedImg, int filterSizeOfOneSide, double sigma_c, double sigma_s) {
	std::vector<std::vector<std::vector<double>>> extendedFilteredImg = extendedImg;
	int amountOfIncrease = filterSizeOfOneSide / 2; //e.g. filter's one side size =3 => 3/2 = 1, A side of image is increased by 1 based on the given filter size.
	int extImgWidth = extendedImg[0].size();
	int extimgHeight = extendedImg.size();
	for (int row = (0 + amountOfIncrease); row < (extimgHeight - amountOfIncrease); row++) {
		for (int col = (0 + amountOfIncrease); col < (extImgWidth - amountOfIncrease); col++) {
			double numerator = 0.0;
			double denominator = 0.0;

			//k and l are notations for the index from the homework 1 description. 
			for (int k = (row - amountOfIncrease); k <= (row + amountOfIncrease); k++) {
				for (int l = (col - amountOfIncrease); l <= (col + amountOfIncrease); l++) {
					double term1 = -((row - k) * (row - k) + (col - l) * (col - l)) / (2.0 * sigma_c * sigma_c);
					double term2 = -((extendedImg[row][col][0] - extendedImg[k][l][0]) * (extendedImg[row][col][0] - extendedImg[k][l][0])) / (2 * sigma_s * sigma_s);
					double w_ijkl = exp(term1 + term2); //w_ijkl represents w(i,j,k,l) from the homework description.

					denominator = denominator + w_ijkl;
					numerator = numerator + (extendedImg[k][l][0] * w_ijkl);

					
					
				}
			}
			


			extendedFilteredImg[row][col][0] = numerator / denominator;

		}
	}
	
	return extendedFilteredImg;

}

//Perform Non Local Mean Denoising Algorithm to an non extended image(i.e. input image has the same dimension of image as noisy image) and return non extended filtered image
std::vector<std::vector<std::vector<double>>> ImageProcessor::NonLocalMeanDenoising(std::vector<std::vector<std::vector<double>>> noisyImg, int largeWindowSizeOfOneSide, int smallWindowSizeOfOneSide, double hyperParameter_h, double hyperParameter_a) {
	std::vector<std::vector<std::vector<double>>> filteredImgByNLM = noisyImg;
	int amountOfIncreaseByLargeWindow = largeWindowSizeOfOneSide / 2; //e.g. amount of increases by large window = 5/2 = 2 (integer)
	int amountOfIncreaseBySmallWindow = smallWindowSizeOfOneSide / 2; //e.g. amount of increases by small window = 3/2 = 1 (integer)
	int imgWidth = noisyImg[0].size();
	int imgHeight = noisyImg.size();
	double pi = 3.14159265;
	//In this implementation variables notations are closed to homework1 description
	//i,k,n1 : represent row
	//j,l,n2 : represent column
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			double denominator = 0.0;
			double numerator = 0.0;

			for (int k = (i - amountOfIncreaseByLargeWindow); k <= (i + amountOfIncreaseByLargeWindow); k++) {
				for (int l = (j - amountOfIncreaseByLargeWindow); l <= (j + amountOfIncreaseByLargeWindow); l++) {
					double expo_numerator = 0.0;
					double n1_prime = 0.0;//it will be used for G(n1',n2') term. n1'and n2' range depends on small window size. e.g. small window size 3 => 0<=n1<=2
					double n2_prime = 0.0;
					
					for (int n1 = (k - amountOfIncreaseBySmallWindow); n1 <= (k + amountOfIncreaseBySmallWindow); n1++) {
						for (int n2 = (l - amountOfIncreaseBySmallWindow); n2 <= (l - amountOfIncreaseBySmallWindow); n2++) {
							//only sum up terms when index is in index of noisy image
							if (((i - n1) >= 0) && ((i - n1) < imgHeight) && ((j - n2) >= 0) && ((j - n2) < imgWidth) && ((k - n1) >= 0) && ((k - n1) < imgHeight) && ((l - n2) >= 0) && ((l - n2) < imgWidth)) {
								double G_term = (1.0 / (sqrt(2 * pi) * hyperParameter_a)) * exp(-((n1_prime * n1_prime + n2_prime * n2_prime) / (2 * hyperParameter_a * hyperParameter_a)));
								expo_numerator = expo_numerator + G_term * pow((noisyImg[i - n1][j - n2][0] - noisyImg[k - n1][l - n2][0]), 2.0);

							}
						}
					}
					double w_ijkl = exp(-(expo_numerator / pow(hyperParameter_h, 2))); //w_ijkl = w(i,j,k,l)
					denominator = denominator + w_ijkl;
					if ((k >= 0) && (k < imgHeight) && (l >= 0) && (l < imgWidth)) {
						numerator = numerator + (noisyImg[k][l][0]) * w_ijkl;
					}


				}
			}

			filteredImgByNLM[i][j][0] = numerator / denominator;
		}
	}

	return filteredImgByNLM;
}





//*********************Class Methods for Histogram Manipulation 

//Method A Step1
//Count number of each intensity in one channel and return a corresponding 1D vector that includes the number of appearance of each intensity
std::vector<int> ImageProcessor::intensityCounterOneChannel(std::vector<std::vector<std::vector<double>>> oneChannel) {
	int maxValOfIntensity = 255;
	int zero = 0;
	int bytePerPixel = 1;
	int one = 1;

	//Initialize size of 256 1D Vector with zeros 
	std::vector<int> one_channel_vec(maxValOfIntensity+ one, zero); // 1D vector that will have number of appearance of each intensity for one channel 2D image matrix data
	
	

	
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
//Method A Step2
//Calculate Normalized Probability and return a 1D vector that includes normalized probability for each intensity
std::vector<double> ImageProcessor::normalizedProbCalculator(std::vector<int> intensitiesOneChannel) {
	int maxValOfIntensity = 255;
	double zero = 0.0;
	int one = 1;

	//Initialize size of 256 1D Vector with zeros 
	std::vector<double> probabilityVector(maxValOfIntensity+ one, zero);
	
	double totalNumPixels = static_cast<double>(imgHeight) * static_cast<double>(imgWidth); //Need to make sure this object is created by right image, so then it will be run with correct imgHeight and imgWidth


	for (int intensity = 0; intensity <= maxValOfIntensity; intensity++) {
		probabilityVector[intensity] = static_cast<double>(intensitiesOneChannel[intensity]) / (totalNumPixels);
	}

	return probabilityVector;
}

//Method A Step3
//Calculate Cummulative Density Function(CDF) for each intensity and return a 1D vector than includes CDF for each intensity
std::vector<double> ImageProcessor::CDFCalculator(std::vector<double> normalizedProbVector) {
	int maxValOfIntensity = 255;
	int one = 1;
	double zero = 0.0;
	
	//Initialize size of 256 1D Vector with zero
	std::vector<double> CDFVector(maxValOfIntensity+one,zero);

	
	for (int index = 0; index <= maxValOfIntensity; index++) {
		
		if (index == 0) {
			CDFVector[index] = normalizedProbVector[index];
		}

		else {
			CDFVector[index] = normalizedProbVector[index] + CDFVector[index-1] ; //add previous value to current value, so it becomes CDF for current index(intensity)
		}
		
		
		


	}
	
	return CDFVector;
}

//Class method for Method A Step4
//Generate and return transfer function(i.e., mapping function) that maps original intensity for each pixel to manipulated intensity 
std::vector<double> ImageProcessor::mappingFuncGenerator(std::vector<double> CDFVector) {
	std::vector<double> mappingVector = CDFVector;

	int maxValOfIntensity = 255;

	for (int index = 0; index <= maxValOfIntensity; index++) {
		mappingVector[index] = floor( CDFVector[index] * static_cast<double>(maxValOfIntensity) ); //round down to closest integer but type of value is still double
	}

	return mappingVector;
}


//Class method for Method A Step5
//Perform histogram manipulation method A to input one channel image and return it.
std::vector<std::vector<std::vector<double>>> ImageProcessor::hisManipulatorMethodA(std::vector<double> mappingFunction, std::vector<std::vector<std::vector<double>>> oneChannelImg) {
	int maxValOfIntensity = 255;
	int inputImgHeight = oneChannelImg.size();
	int inputImgWidth = oneChannelImg[0].size();

	std::vector<std::vector<std::vector<double>>> manipulatedOneChannelImg = oneChannelImg;


	for (int intensity = 0; intensity <= 255; intensity++) {
		for (int col = 0; col < inputImgWidth; col++) {
			for (int row = 0; row < inputImgHeight; row++) {

				//whenever current intensity appears in the input one channel image it maps(i.e., changes) original intensity to manipulated intensity value
				if (static_cast<int>(oneChannelImg[row][col][0])==intensity) {
					manipulatedOneChannelImg[row][col][0] = mappingFunction[intensity];
				}
			}
		}
	
	}

	return manipulatedOneChannelImg;
}



//Class method for Method B
//Perform histogram manipulation methodB(Filling Buckets Method) to one channel input image and return image matrix that its histgram is modified
std::vector<std::vector<std::vector<double>>> ImageProcessor::hisManipulatorMethodB(std::vector<std::vector<std::vector<double>>> oneChannelImg) {
	std::vector<std::vector<std::vector<double>>> manipulatedOneChannelImg = oneChannelImg;
	int inputImgHeight = oneChannelImg.size();
	int inputImgWidth = oneChannelImg[0].size();
	int numOfBallsInImg = inputImgHeight * inputImgWidth; //e.g. number of balls(a.k.a pixels) : Height of Image * Width of Image
	int numOfBuckets = 256; //e.g. gray scale: 0~255=> total 256 values. This number corresponds to number of buckets
	int ballsPerBuckets = numOfBallsInImg / numOfBuckets;//e.g. for toy image 560*400/256 = 875
	int maxValOfIntensity = 255;

	int numberOfElementInPair = 3; //e.g.(Value of intensity, row value of its location, column value of its location)
	int zero = 0; //zero represents integer zero
	 
	std::vector<std::vector<int>> allPairsStorage;

	//Step1. Sort intensity values from image and save it with its location, row and column
	for (int intensity = 0; intensity <= maxValOfIntensity; intensity++) {
		for (int row = 0; row < inputImgHeight; row++) {
			for (int col = 0; col < inputImgWidth; col++) {
				std::vector<int> intensityLocationPair(numberOfElementInPair, zero); //initialize data variable to save pairs
				if (intensity == static_cast<int>(oneChannelImg[row][col][0])) {
					intensityLocationPair[0] = static_cast<int>(oneChannelImg[row][col][0]);//save intensity value
					intensityLocationPair[1] = row;//save row value of the intensity's location
					intensityLocationPair[2] = col;//save column value of the intensity's location
					allPairsStorage.push_back(intensityLocationPair);//push back to all pairs storage
				}
			}
		}	
	}

	//Step2. Manipulate intensity values in all pair storage as many as balls per buckets
	int index = 0; // it will be used to indicate an index of all pair storage
	for (int intensity = 0; intensity <= maxValOfIntensity; intensity++) {
		for (int counter = 0; counter < ballsPerBuckets; counter++) {
			allPairsStorage[index][0] = intensity;
			index++;
		}
	}

	//Step3. Map the manipulated intensities to location of original image, row and column
	for (int index = 0; index < allPairsStorage.size(); index++) {
		int row = allPairsStorage[index][1];//assign row value from the storage
		int col = allPairsStorage[index][2];//assign column value from the storage

		double manipulatedIntensity = static_cast<double>(allPairsStorage[index][0]);//assign manipulated intensity from the storage

		manipulatedOneChannelImg[row][col][0] = manipulatedIntensity;
	}

	return manipulatedOneChannelImg;
}
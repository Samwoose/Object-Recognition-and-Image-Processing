#include "FiltersHW2.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <math.h>



FiltersHW2::FiltersHW2(int tep) {
	temp = tep;
}
FiltersHW2::~FiltersHW2() {}

std::vector<std::vector<double>> FiltersHW2::xDirectionSobelFilterGenerator() {

	int size_xDirectionSobelFilter = 3; //We will use 3x3 size filter

	std::vector<double> xDirectionSobelFilter_1D(size_xDirectionSobelFilter);
	std::vector<std::vector<double>> xDirectionSobelFilter_2D(size_xDirectionSobelFilter, xDirectionSobelFilter_1D);

	//filter coefficients are from discussion week 3
	xDirectionSobelFilter_2D[0][0] = -1;
	xDirectionSobelFilter_2D[0][1] = 0;
	xDirectionSobelFilter_2D[0][2] = +1;

	xDirectionSobelFilter_2D[1][0] = -2;
	xDirectionSobelFilter_2D[1][1] = 0;
	xDirectionSobelFilter_2D[1][2] = +2;

	xDirectionSobelFilter_2D[2][0] = -1;
	xDirectionSobelFilter_2D[2][1] = 0;
	xDirectionSobelFilter_2D[2][2] = 1;

	return xDirectionSobelFilter_2D;
}


std::vector<std::vector<double>> FiltersHW2::yDirectionSobelFilterGenerator() {
	int size_yDirectionSobelFilter = 3; //We will use 3x3 size filter

	std::vector<double> yDirectionSobelFilter_1D(size_yDirectionSobelFilter);
	std::vector<std::vector<double>> yDirectionSobelFilter_2D(size_yDirectionSobelFilter, yDirectionSobelFilter_1D);

	yDirectionSobelFilter_2D[0][0] = 1;
	yDirectionSobelFilter_2D[0][1] = 2;
	yDirectionSobelFilter_2D[0][2] = 1;

	yDirectionSobelFilter_2D[1][0] = 0;
	yDirectionSobelFilter_2D[1][1] = 0;
	yDirectionSobelFilter_2D[1][2] = 0;

	yDirectionSobelFilter_2D[2][0] = -1;
	yDirectionSobelFilter_2D[2][1] = -2;
	yDirectionSobelFilter_2D[2][2] = -1;

	return yDirectionSobelFilter_2D;

}


//*********************Filters for Digital Halftoning
//*******************Dithering
//Generate a basic Index Matrix. Its size is 2 x 2.
std::vector<std::vector<double>> FiltersHW2::basicIndexMatrixGenerator() {
	int sizeOfIndexMatrix = 2; //We will use 2x2 size filter

	std::vector<double> basicIndexMatrix_1D(sizeOfIndexMatrix);
	std::vector<std::vector<double>> basicIndexMatrix_2D(sizeOfIndexMatrix, basicIndexMatrix_1D);

	basicIndexMatrix_2D[0][0] = 1;
	basicIndexMatrix_2D[0][1] = 2;

	basicIndexMatrix_2D[1][0] = 3;
	basicIndexMatrix_2D[1][1] = 0;

	return basicIndexMatrix_2D;
}


//Generate 2N x 2N size Index Matrix. n is a parameter of this function
std::vector<std::vector<double>> FiltersHW2::indexMatrixGenerator(std::vector<std::vector<double>> previousIndexMatrix) {
	int inputMatrixHeight = previousIndexMatrix.size();
	int inputMatrixWidth = previousIndexMatrix[0].size();

	int sizeOfOnesideOfInput = inputMatrixHeight; //input matrix is always even x even size

	int outputMatrixHeight = 2 * inputMatrixHeight;
	int outputMatrixWidth = 2 * inputMatrixWidth;

	//initialize output matrix
	std::vector<double> outputMatrix_1D(outputMatrixWidth);
	std::vector<std::vector<double>> outputMatrix_2D(outputMatrixHeight, outputMatrix_1D);

	

	for (int row = 0; row < inputMatrixHeight; row++) {
		for (int col = 0; col < inputMatrixWidth; col++) {
			//Check detail of this part on practice note p.5(Personally own, Not available on public)
			//Also check the formula on HW2 Description.
			//case1: First Block Part in output matrix
			outputMatrix_2D[row][col] = previousIndexMatrix[row][col] * 4 + 1;
			//case2: Second Block Part in output matrix
			outputMatrix_2D[row][col + sizeOfOnesideOfInput] = previousIndexMatrix[row][col] * 4 + 2;
			//case3: Third Block Part in output matrix
			outputMatrix_2D[row + sizeOfOnesideOfInput][col] = previousIndexMatrix[row][col] * 4 + 3;
			//case4: Forth Block Part in output matrix
			outputMatrix_2D[row + sizeOfOnesideOfInput][col + sizeOfOnesideOfInput] = previousIndexMatrix[row][col] * 4 ;

		}
	}

	
	return outputMatrix_2D;
}

//Convert 2N x 2N index matrix to a Threshold Matrix and return it.
std::vector<std::vector<double>> FiltersHW2::thresholdMatrixGenerator(std::vector<std::vector<double>> indexMatrix) {
	int inputMatrixHeight = indexMatrix.size();
	int inputMatrixWidth = indexMatrix[0].size();

	int sizeOfOnesideOfInput = inputMatrixHeight; //input matrix is always even x even size
	double maxIntensity = 255.0;
	//initialize thresholding matrix with the same size as input index matrix
	std::vector<double> thresholdingMatrix_1D(inputMatrixWidth);
	std::vector<std::vector<double>>thresholdingMatrix_2D(inputMatrixHeight, thresholdingMatrix_1D);

	for (int row = 0; row < inputMatrixHeight; row++) {
		for (int col = 0; col < inputMatrixWidth; col++) {
			thresholdingMatrix_2D[row][col] = (indexMatrix[row][col] + 0.5) / (pow(static_cast<double>(sizeOfOnesideOfInput), 2.0))* maxIntensity;

		}
	}

	return thresholdingMatrix_2D;

}



//**********************Masks for Error Diffusion
//(1) Floyd-Steinberg's Error Diffusion Mask
std::vector<std::vector<double>> FiltersHW2::left2Right_floydErrorDiffusionMaskGenerator() {
	int maskHeight = 3;
	int maskWidth = 3;
	
	//initialize thresholding matrix with the same size as input index matrix
	std::vector<double> floydErrorDiffusionMask_1D(maskWidth);
	std::vector<std::vector<double>>floydErrorDiffusionMask_2D(maskHeight, floydErrorDiffusionMask_1D);

	//Assign weights of the mask properly
	//Weights are from the HW2 description
	floydErrorDiffusionMask_2D[0][0] = 0;
	floydErrorDiffusionMask_2D[0][1] = 0;
	floydErrorDiffusionMask_2D[0][2] = 0;

	floydErrorDiffusionMask_2D[1][0] = 0;
	floydErrorDiffusionMask_2D[1][1] = 0;
	floydErrorDiffusionMask_2D[1][2] = 7/16.0;

	floydErrorDiffusionMask_2D[2][0] = 3/ 16.0;
	floydErrorDiffusionMask_2D[2][1] = 5/ 16.0;
	floydErrorDiffusionMask_2D[2][2] = 1/ 16.0;

	return floydErrorDiffusionMask_2D;

}


//**********************Masks for Error Diffusion
//(1) Floyd-Steinberg's Error Diffusion Mask
std::vector<std::vector<double>> FiltersHW2::right2Left_floydErrorDiffusionMaskGenerator() {
	int maskHeight = 3;
	int maskWidth = 3;

	//initialize thresholding matrix with the same size as input index matrix
	std::vector<double> floydErrorDiffusionMask_1D(maskWidth);
	std::vector<std::vector<double>>floydErrorDiffusionMask_2D(maskHeight, floydErrorDiffusionMask_1D);

	//Assign weights of the mask properly
	//Weights are from the HW2 description
	floydErrorDiffusionMask_2D[0][0] = 0;
	floydErrorDiffusionMask_2D[0][1] = 0;
	floydErrorDiffusionMask_2D[0][2] = 0;

	floydErrorDiffusionMask_2D[1][0] = 7 / 16.0;
	floydErrorDiffusionMask_2D[1][1] = 0;
	floydErrorDiffusionMask_2D[1][2] = 0;

	floydErrorDiffusionMask_2D[2][0] = 1 / 16.0;
	floydErrorDiffusionMask_2D[2][1] = 5 / 16.0;
	floydErrorDiffusionMask_2D[2][2] = 3 / 16.0;

	return floydErrorDiffusionMask_2D;
}



//(2) Error Diffusion Mask by Jarvis, Judice, and Ninke(JJN)
std::vector<std::vector<double>> FiltersHW2::left2Right_jjnErrorDiffusionMaskGenerator() {
	int maskHeight = 5;
	int maskWidth = 5;

	//initialize thresholding matrix with the same size as input index matrix
	std::vector<double> jjnErrorDiffusionMask_1D(maskWidth);
	std::vector<std::vector<double>>jjnErrorDiffusionMask_2D(maskHeight, jjnErrorDiffusionMask_1D);

	//Assign weights of the mask properly
	//Weights are from the HW2 description
	jjnErrorDiffusionMask_2D[0][0] = 0;
	jjnErrorDiffusionMask_2D[0][1] = 0;
	jjnErrorDiffusionMask_2D[0][2] = 0;
	jjnErrorDiffusionMask_2D[0][3] = 0;
	jjnErrorDiffusionMask_2D[0][4] = 0;

	jjnErrorDiffusionMask_2D[1][0] = 0;
	jjnErrorDiffusionMask_2D[1][1] = 0;
	jjnErrorDiffusionMask_2D[1][2] = 0;
	jjnErrorDiffusionMask_2D[1][3] = 0;
	jjnErrorDiffusionMask_2D[1][4] = 0;

	jjnErrorDiffusionMask_2D[2][0] = 0;
	jjnErrorDiffusionMask_2D[2][1] = 0;
	jjnErrorDiffusionMask_2D[2][2] = 0;
	jjnErrorDiffusionMask_2D[2][3] = 7.0 / 48.0;
	jjnErrorDiffusionMask_2D[2][4] = 5.0 / 48.0;

	jjnErrorDiffusionMask_2D[3][0] = 3 / 48.0;
	jjnErrorDiffusionMask_2D[3][1] = 5 / 48.0;
	jjnErrorDiffusionMask_2D[3][2] = 7 / 48.0;
	jjnErrorDiffusionMask_2D[3][3] = 5 / 48.0;
	jjnErrorDiffusionMask_2D[3][4] = 3 / 48.0;

	jjnErrorDiffusionMask_2D[4][0] = 1 / 48.0;
	jjnErrorDiffusionMask_2D[4][1] = 3 / 48.0;
	jjnErrorDiffusionMask_2D[4][2] = 5 / 48.0;
	jjnErrorDiffusionMask_2D[4][3] = 3 / 48.0;
	jjnErrorDiffusionMask_2D[4][4] = 1 / 48.0;

	return jjnErrorDiffusionMask_2D;
}

//(2) Error Diffusion Mask by Jarvis, Judice, and Ninke(JJN)
std::vector<std::vector<double>> FiltersHW2::right2Left_jjnErrorDiffusionMaskGenerator() {
	int maskHeight = 5;
	int maskWidth = 5;

	//initialize thresholding matrix with the same size as input index matrix
	std::vector<double> jjnErrorDiffusionMask_1D(maskWidth);
	std::vector<std::vector<double>>jjnErrorDiffusionMask_2D(maskHeight, jjnErrorDiffusionMask_1D);

	//Assign weights of the mask properly
	//Weights are from the HW2 description
	jjnErrorDiffusionMask_2D[0][0] = 0;
	jjnErrorDiffusionMask_2D[0][1] = 0;
	jjnErrorDiffusionMask_2D[0][2] = 0;
	jjnErrorDiffusionMask_2D[0][3] = 0;
	jjnErrorDiffusionMask_2D[0][4] = 0;

	jjnErrorDiffusionMask_2D[1][0] = 0;
	jjnErrorDiffusionMask_2D[1][1] = 0;
	jjnErrorDiffusionMask_2D[1][2] = 0;
	jjnErrorDiffusionMask_2D[1][3] = 0;
	jjnErrorDiffusionMask_2D[1][4] = 0;

	jjnErrorDiffusionMask_2D[2][0] = 5.0 / 48.0;
	jjnErrorDiffusionMask_2D[2][1] = 7.0 / 48.0;
	jjnErrorDiffusionMask_2D[2][2] = 0;
	jjnErrorDiffusionMask_2D[2][3] = 0;
	jjnErrorDiffusionMask_2D[2][4] = 0;

	jjnErrorDiffusionMask_2D[3][0] = 3 / 48.0;
	jjnErrorDiffusionMask_2D[3][1] = 5 / 48.0;
	jjnErrorDiffusionMask_2D[3][2] = 7 / 48.0;
	jjnErrorDiffusionMask_2D[3][3] = 5 / 48.0;
	jjnErrorDiffusionMask_2D[3][4] = 3 / 48.0;

	jjnErrorDiffusionMask_2D[4][0] = 1 / 48.0;
	jjnErrorDiffusionMask_2D[4][1] = 3 / 48.0;
	jjnErrorDiffusionMask_2D[4][2] = 5 / 48.0;
	jjnErrorDiffusionMask_2D[4][3] = 3 / 48.0;
	jjnErrorDiffusionMask_2D[4][4] = 1 / 48.0;

	return jjnErrorDiffusionMask_2D;
}



//(3) Error Diffusion Mask by Stucki
std::vector<std::vector<double>> FiltersHW2::left2Right_stuckiErrorDiffusionMaskGenerator() {
	int maskHeight = 5;
	int maskWidth = 5;

	//initialize thresholding matrix with the same size as input index matrix
	std::vector<double> stuckiErrorDiffusionMask_1D(maskWidth);
	std::vector<std::vector<double>>stuckiErrorDiffusionMask_2D(maskHeight, stuckiErrorDiffusionMask_1D);

	//Assign weights of the mask properly
	//Weights are from the HW2 description
	stuckiErrorDiffusionMask_2D[0][0] = 0;
	stuckiErrorDiffusionMask_2D[0][1] = 0;
	stuckiErrorDiffusionMask_2D[0][2] = 0;
	stuckiErrorDiffusionMask_2D[0][3] = 0;
	stuckiErrorDiffusionMask_2D[0][4] = 0;

	stuckiErrorDiffusionMask_2D[1][0] = 0;
	stuckiErrorDiffusionMask_2D[1][1] = 0;
	stuckiErrorDiffusionMask_2D[1][2] = 0;
	stuckiErrorDiffusionMask_2D[1][3] = 0;
	stuckiErrorDiffusionMask_2D[1][4] = 0;

	stuckiErrorDiffusionMask_2D[2][0] = 0;
	stuckiErrorDiffusionMask_2D[2][1] = 0;
	stuckiErrorDiffusionMask_2D[2][2] = 0;
	stuckiErrorDiffusionMask_2D[2][3] = 8 / 42.0;
	stuckiErrorDiffusionMask_2D[2][4] = 4 / 42.0;

	stuckiErrorDiffusionMask_2D[3][0] = 2 / 42.0;
	stuckiErrorDiffusionMask_2D[3][1] = 4 / 42.0;
	stuckiErrorDiffusionMask_2D[3][2] = 8 / 42.0;
	stuckiErrorDiffusionMask_2D[3][3] = 4 / 42.0;
	stuckiErrorDiffusionMask_2D[3][4] = 2 / 42.0;

	stuckiErrorDiffusionMask_2D[4][0] = 1 / 42.0;
	stuckiErrorDiffusionMask_2D[4][1] = 2 / 42.0;
	stuckiErrorDiffusionMask_2D[4][2] = 4 / 42.0;
	stuckiErrorDiffusionMask_2D[4][3] = 2 / 42.0;
	stuckiErrorDiffusionMask_2D[4][4] = 1 / 42.0;

	return stuckiErrorDiffusionMask_2D;


}

//(3) Error Diffusion Mask by Stucki
std::vector<std::vector<double>> FiltersHW2::right2Left_stuckiErrorDiffusionMaskGenerator() {
	int maskHeight = 5;
	int maskWidth = 5;

	//initialize thresholding matrix with the same size as input index matrix
	std::vector<double> stuckiErrorDiffusionMask_1D(maskWidth);
	std::vector<std::vector<double>>stuckiErrorDiffusionMask_2D(maskHeight, stuckiErrorDiffusionMask_1D);

	//Assign weights of the mask properly
	//Weights are from the HW2 description
	stuckiErrorDiffusionMask_2D[0][0] = 0;
	stuckiErrorDiffusionMask_2D[0][1] = 0;
	stuckiErrorDiffusionMask_2D[0][2] = 0;
	stuckiErrorDiffusionMask_2D[0][3] = 0;
	stuckiErrorDiffusionMask_2D[0][4] = 0;

	stuckiErrorDiffusionMask_2D[1][0] = 0;
	stuckiErrorDiffusionMask_2D[1][1] = 0;
	stuckiErrorDiffusionMask_2D[1][2] = 0;
	stuckiErrorDiffusionMask_2D[1][3] = 0;
	stuckiErrorDiffusionMask_2D[1][4] = 0;

	stuckiErrorDiffusionMask_2D[2][0] = 4 / 42.0;
	stuckiErrorDiffusionMask_2D[2][1] = 8 / 42.0;
	stuckiErrorDiffusionMask_2D[2][2] = 0;
	stuckiErrorDiffusionMask_2D[2][3] = 0;
	stuckiErrorDiffusionMask_2D[2][4] = 0;

	stuckiErrorDiffusionMask_2D[3][0] = 2 / 42.0;
	stuckiErrorDiffusionMask_2D[3][1] = 4 / 42.0;
	stuckiErrorDiffusionMask_2D[3][2] = 8 / 42.0;
	stuckiErrorDiffusionMask_2D[3][3] = 4 / 42.0;
	stuckiErrorDiffusionMask_2D[3][4] = 2 / 42.0;

	stuckiErrorDiffusionMask_2D[4][0] = 1 / 42.0;
	stuckiErrorDiffusionMask_2D[4][1] = 2 / 42.0;
	stuckiErrorDiffusionMask_2D[4][2] = 4 / 42.0;
	stuckiErrorDiffusionMask_2D[4][3] = 2 / 42.0;
	stuckiErrorDiffusionMask_2D[4][4] = 1 / 42.0;

	return stuckiErrorDiffusionMask_2D;


}
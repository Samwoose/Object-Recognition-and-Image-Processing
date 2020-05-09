#pragma once

#ifndef FILTERSHW2_H
#define FILTERSHW2_H
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <math.h>


class FiltersHW2
{


private:
	int temp;//Some random int variable to construct class


public:
	//constructor 
	FiltersHW2(int tep);

	//destructor
	~FiltersHW2();
	//*******************Filters for Edge Detection
	//Generate X direction sobel filter 
	std::vector<std::vector<double>> xDirectionSobelFilterGenerator();

	//Generate Y direction sobel filter 
	std::vector<std::vector<double>> yDirectionSobelFilterGenerator();


	//*********************Filters for Digital Halftoning
	//*******************Dithering
	//Generate a basic Index Matrix. Its size is 2 x 2.
	std::vector<std::vector<double>> basicIndexMatrixGenerator();

	//Generate 2N x 2N size Index Matrix
	std::vector<std::vector<double>> indexMatrixGenerator(std::vector<std::vector<double>> previousIndexMatrix);

	//Convert a given index matrix to a Threshold Matrix and return it
	std::vector<std::vector<double>> thresholdMatrixGenerator(std::vector<std::vector<double>> indexMatrix);

	//**********************Masks for Error Diffusion
	//(1) Floyd-Steinberg's Error Diffusion Mask
	std::vector<std::vector<double>> left2Right_floydErrorDiffusionMaskGenerator();
	
	//(2) Error Diffusion Mask by Jarvis, Judice, and Ninke(JJN)
	std::vector<std::vector<double>> left2Right_jjnErrorDiffusionMaskGenerator();

	//(3) Error Diffusion Mask by Stucki
	std::vector<std::vector<double>> left2Right_stuckiErrorDiffusionMaskGenerator();

	//**********************Masks for Error Diffusion
	//(1) Floyd-Steinberg's Error Diffusion Mask
	std::vector<std::vector<double>> right2Left_floydErrorDiffusionMaskGenerator();

	//(2) Error Diffusion Mask by Jarvis, Judice, and Ninke(JJN)
	std::vector<std::vector<double>> right2Left_jjnErrorDiffusionMaskGenerator();

	//(3) Error Diffusion Mask by Stucki
	std::vector<std::vector<double>> right2Left_stuckiErrorDiffusionMaskGenerator();

};



#endif // ! FILTERSHW2_H


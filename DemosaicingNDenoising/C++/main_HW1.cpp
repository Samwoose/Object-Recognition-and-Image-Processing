// This sample code reads in image data from a RAW image file and 
// writes it into another file

// NOTE:	The code assumes that the image is of size 256 x 256 and is in the
//			RAW format. You will need to make corresponding changes to
//			accommodate images of different sizes and/or types

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include "ImageLoadRead.h"
#include "ImageWriteSave.h"
#include "ImageProcessor.h"
#include "Filters.h"
#include <cmath>
#include <math.h>

using namespace std;


int main(int argc, char* argv[])

{
	//**Need to modify command line properly to get correct outputs
	//Necessary Paths for demosaicing
	string readPathDog = argv[1];
	string savePathDogBi = argv[2];
	string savePathDogMHC = argv[3];

	//Necessary Paths for denosing
	string readPathCorn1 = argv[4];
	string savePathCorn1 = argv[5];
	string savePathCorn2 = argv[6];
	
	string readPathCornNoNoise = argv[7];
	string savePathCornBiDenoise = argv[8];
	string savePathCornNLMDenoise = argv[9];

	//Necessary Path for histogram manipulation
	string readPathToy = argv[10];
	string savePathToyMethodA = argv[11];
	string savePathToyMethodB = argv[12];

	




	//******************Image Demosaicing*******************************
	// Define variables. This variables depend on an image for each problem 
	int BytesPerPixel = 1;
	int width = 600;
	int height = 532;

	ImageLoadRead readObj(height, width, BytesPerPixel, readPathDog);
	//load image
	readObj.rawImgLoad();

	//get matrix data of raw image
	std::vector<std::vector<std::vector<double>>> imageData_green_bl = readObj.getImageData();
	std::vector<std::vector<std::vector<double>>> imageData_red_bl = readObj.getImageData();
	std::vector<std::vector<std::vector<double>>> imageData_blue_bl = readObj.getImageData();
	std::vector<std::vector<std::vector<double>>> inputImage = readObj.getImageData();



	//Problem1. (a)Bilinear Demosaicing
	//Green Channel Estimation
	//practice page(18)
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			//Case1
			//Estimate green at blue pixel, but neither edge pixel nor side pixel 
			//practice note page(18)
			if ((row % 2) == 1 && (col % 2) == 0 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1)) {
				imageData_green_bl[row][col][BytesPerPixel - 1] = ((1.0 / 4.0) * inputImage[row - 1][col][BytesPerPixel - 1]
					+ (1.0 / 4.0) * inputImage[row][col - 1][BytesPerPixel - 1]
					+ (1.0 / 4.0) * inputImage[row + 1][col][BytesPerPixel - 1]
					+ (1.0 / 4.0) * inputImage[row][col + 1][BytesPerPixel - 1]);
			}

			//case2
			//Estimate green at red pixel, but neither edge pixel nor side pixel 
			//practice note page(18)
			else if ((row % 2) == 0 && (col % 2) == 1 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1)) {
				imageData_green_bl[row][col][BytesPerPixel - 1] = ((1.0 / 4.0) * inputImage[row - 1][col][BytesPerPixel - 1]
					+ (1.0 / 4.0) * inputImage[row][col - 1][BytesPerPixel - 1]
					+ (1.0 / 4.0) * inputImage[row + 1][col][BytesPerPixel - 1]
					+ (1.0 / 4.0) * inputImage[row][col + 1][BytesPerPixel - 1]);
			}

		}
	}

	//Red Channel Estimation
	//practice page(19)
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			//case1
			//Estimate red at green and blue line in horizontal sense also, neither edge nor side pixel
			//practice page(19)
			if ((row % 2) == 1 && (col % 2) == 1 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1)) {
				imageData_red_bl[row][col][BytesPerPixel - 1] = ((1.0 / 2.0) * inputImage[row - 1][col][BytesPerPixel - 1]
					+ (1.0 / 2.0) * inputImage[row + 1][col][BytesPerPixel - 1]);
			}

			//case2
			//Estimate red at green and red line in horizontal sense also, neither edge nor side pixel
			//practice page(19)
			else if ((row % 2) == 0 && (col % 2) == 0 && (row != 0 && row != (height - 1) && col != 0 && col != (width - 1))) {
				imageData_red_bl[row][col][BytesPerPixel - 1] = ((1.0 / 2.0) * inputImage[row][col - 1][BytesPerPixel - 1]
					+ (1.0 / 2.0) * inputImage[row][col + 1][BytesPerPixel - 1]);
			}

			//case3
			//Estimate red at blue also, neither edge nor side pixel
			//practice page(19)
			else if ((row % 2) == 1 && (col % 2) == 0 && (row != 0 && row != (height - 1) && col != 0 && col != (width - 1))) {
				imageData_red_bl[row][col][BytesPerPixel - 1] = ((1.0 / 4.0) * inputImage[row - 1][col - 1][BytesPerPixel - 1]
					+ (1.0 / 4.0) * inputImage[row - 1][col + 1][BytesPerPixel - 1]
					+ (1.0 / 4.0) * inputImage[row + 1][col - 1][BytesPerPixel - 1]
					+ (1.0 / 4.0) * inputImage[row + 1][col + 1][BytesPerPixel - 1]);
			}
		}
	}


	//Blue Channel Estimation

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			//case1
			//Estimate blue at green and blue line in horizontal sense also, neither edge nor side pixel
			//practice page(19)
			if ((row % 2) == 1 && (col % 2) == 1 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1)) {
				imageData_blue_bl[row][col][BytesPerPixel - 1] = ((1.0 / 2.0) * inputImage[row][col - 1][BytesPerPixel - 1]
					+ (1.0 / 2.0) * inputImage[row][col + 1][BytesPerPixel - 1]);
			}

			//case2
			//Estimate red at green and red line in horizontal sense also, neither edge nor side pixel
			//practice page(19)
			else if ((row % 2) == 0 && (col % 2) == 0 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1)) {
				imageData_blue_bl[row][col][BytesPerPixel - 1] = ((1.0 / 2.0) * inputImage[row - 1][col][BytesPerPixel - 1]
					+ (1.0 / 2.0) * inputImage[row + 1][col][BytesPerPixel - 1]);
			}

			//case3
			//Estimate blue at red also, neither edge nor side pixel
			//practice page(19)
			else if ((row % 2) == 0 && (col % 2) == 1 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1)) {
				imageData_blue_bl[row][col][BytesPerPixel - 1] = ((1.0 / 4.0) * inputImage[row - 1][col - 1][BytesPerPixel - 1]
					+ (1.0 / 4.0) * inputImage[row - 1][col + 1][BytesPerPixel - 1]
					+ (1.0 / 4.0) * inputImage[row + 1][col - 1][BytesPerPixel - 1]
					+ (1.0 / 4.0) * inputImage[row + 1][col + 1][BytesPerPixel - 1]);
			}
		}
	}


	//Save processed Image data to buffer
	//save buffer as raw file
	ImageWriteSave saveObj(height, width, savePathDogBi);

	saveObj.saveAsRawfileColor(imageData_red_bl, imageData_green_bl, imageData_blue_bl);

	//******************Bilinear Demosaicing End*******************************




	//******************MHC Demosaicing**********************************
	//get matrix data of raw image
	ImageLoadRead readObj1(height, width, BytesPerPixel, readPathDog);
	//load image
	readObj1.rawImgLoad();


	std::vector<std::vector<std::vector<double>>> imageData_green_MHC = readObj1.getImageData();
	std::vector<std::vector<std::vector<double>>> imageData_red_MHC = readObj1.getImageData();
	std::vector<std::vector<std::vector<double>>> imageData_blue_MHC = readObj1.getImageData();
	std::vector<std::vector<std::vector<double>>> inputImage_MHC = readObj1.getImageData();

	//Green Channel Estimation with MHC Algorithm
	//Problem1. (b)MHC Demosaicing
	//Green Channel Estimation
	//practice page(24)
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			//Case1(MHC filter1)
			//Estimate green at R location, but neither edge pixel nor side pixel nor second row and col nor width -2 and height -2 
			//practice note page(24)
			if ((row % 2) == 0 && (col % 2) == 1 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1) && row != 1 && row != (height - 2) && col != 1 && col != (width - 2)) {
				imageData_green_MHC[row][col][BytesPerPixel - 1] =
					(1.0 / 8.0) * ((-1.0) * (inputImage_MHC[row - 2][col][BytesPerPixel - 1] +
						inputImage_MHC[row][col - 2][BytesPerPixel - 1] +
						inputImage_MHC[row + 2][col][BytesPerPixel - 1] +
						inputImage_MHC[row][col + 2][BytesPerPixel - 1]) +
						(2.0) * (inputImage_MHC[row - 1][col][BytesPerPixel - 1] +
							inputImage_MHC[row][col - 1][BytesPerPixel - 1] +
							inputImage_MHC[row + 1][col][BytesPerPixel - 1] +
							inputImage_MHC[row][col + 1][BytesPerPixel - 1]) +
							(4.0) * inputImage_MHC[row][col][BytesPerPixel - 1]);


			}

			//case2(MHC filter2)
			//Estimate green at  B location, but neither edge pixel nor side pixel nor second row and col nor width -2 and height -2
			//practice note page(24)
			else if ((row % 2) == 1 && (col % 2) == 0 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1) && row != 1 && row != (height - 2) && col != 1 && col != (width - 2)) {
				imageData_green_MHC[row][col][BytesPerPixel - 1] =
					(1.0 / 8.0) * ((-1.0) * (inputImage_MHC[row - 2][col][BytesPerPixel - 1] +
						inputImage_MHC[row][col - 2][BytesPerPixel - 1] +
						inputImage_MHC[row + 2][col][BytesPerPixel - 1] +
						inputImage_MHC[row][col + 2][BytesPerPixel - 1]) +
						(2.0) * (inputImage_MHC[row - 1][col][BytesPerPixel - 1] +
							inputImage_MHC[row][col - 1][BytesPerPixel - 1] +
							inputImage_MHC[row + 1][col][BytesPerPixel - 1] +
							inputImage_MHC[row][col + 1][BytesPerPixel - 1]) +
							(4.0) * inputImage_MHC[row][col][BytesPerPixel - 1]);
			}
		}
	}




	//Red Channel Estimation with MHC Algorithm
	//practice page(24) and (25)
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			//Case1(MHC filter3)
			//Estimation R at green in R row, B column
			if ((row % 2) == 0 && (col % 2) == 0 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1) && row != 1 && row != (height - 2) && col != 1 && col != (width - 2)) {
				imageData_red_MHC[row][col][BytesPerPixel - 1] =
					(1.0 / 8.0) * ((1.0 / 2.0) * (inputImage_MHC[row - 2][col][BytesPerPixel - 1] +
						inputImage_MHC[row + 2][col][BytesPerPixel - 1]) +
						(-1.0) * (inputImage_MHC[row - 1][col - 1][BytesPerPixel - 1] +
							inputImage_MHC[row][col - 2][BytesPerPixel - 1] +
							inputImage_MHC[row + 1][col - 1][BytesPerPixel - 1] +
							inputImage_MHC[row + 1][col + 1][BytesPerPixel - 1] +
							inputImage_MHC[row][col + 2][BytesPerPixel - 1] +
							inputImage_MHC[row - 1][col + 1][BytesPerPixel - 1]) +
							(4.0) * (inputImage_MHC[row][col - 1][BytesPerPixel - 1] +
								inputImage_MHC[row][col + 1][BytesPerPixel - 1]) +
								(5.0) * (inputImage_MHC[row][col][BytesPerPixel - 1]));
			}

			//Case2(MHC filter4)
			//Estimation R at green in B row, R column
			else if ((row % 2) == 1 && (col % 2) == 1 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1) && row != 1 && row != (height - 2) && col != 1 && col != (width - 2)) {
				imageData_red_MHC[row][col][BytesPerPixel - 1] =
					(1.0 / 8.0) * ((1.0 / 2.0) * (inputImage_MHC[row][col - 2][BytesPerPixel - 1] +
						inputImage_MHC[row][col + 2][BytesPerPixel - 1]) +
						(-1.0) * (inputImage_MHC[row - 2][col][BytesPerPixel - 1] +
							inputImage_MHC[row - 1][col - 1][BytesPerPixel - 1] +
							inputImage_MHC[row + 1][col - 1][BytesPerPixel - 1] +
							inputImage_MHC[row + 2][col][BytesPerPixel - 1] +
							inputImage_MHC[row + 1][col + 1][BytesPerPixel - 1] +
							inputImage_MHC[row - 1][col + 1][BytesPerPixel - 1]) +
							(4.0) * (inputImage_MHC[row - 1][col][BytesPerPixel - 1] +
								inputImage_MHC[row + 1][col][BytesPerPixel - 1]) +
								(5.0) * (inputImage_MHC[row][col][BytesPerPixel - 1]));
			}

			//Case3(MHC filter5)
			//Estimation R at blue in B row , B column
			else if ((row % 2) == 1 && (col % 2) == 0 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1) && row != 1 && row != (height - 2) && col != 1 && col != (width - 2)) {
				imageData_red_MHC[row][col][BytesPerPixel - 1] =
					(1.0 / 8.0) * ((2.0) * (inputImage_MHC[row - 1][col - 1][BytesPerPixel - 1] +
						inputImage_MHC[row - 1][col + 1][BytesPerPixel - 1] +
						inputImage_MHC[row + 1][col - 1][BytesPerPixel - 1] +
						inputImage_MHC[row + 1][col + 1][BytesPerPixel - 1]) +
						(-3.0 / 2.0) * (inputImage_MHC[row - 2][col][BytesPerPixel - 1] +
							inputImage_MHC[row][col - 2][BytesPerPixel - 1] +
							inputImage_MHC[row + 2][col][BytesPerPixel - 1] +
							inputImage_MHC[row][col + 2][BytesPerPixel - 1]) +
							(6.0) * (inputImage_MHC[row][col][BytesPerPixel - 1]));
			}

		}
	}


	//Blue Channel Estimation with MHC Algorithm
	//practice page(24) and (25)
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			//Case1(MHC filter6)
			//Estimation B at green in B row, R column
			if ((row % 2) == 1 && (col % 2) == 1 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1) && row != 1 && row != (height - 2) && col != 1 && col != (width - 2)) {
				imageData_blue_MHC[row][col][BytesPerPixel - 1] =
					(1.0 / 8.0) * ((1.0 / 2.0) * (inputImage_MHC[row - 2][col][BytesPerPixel - 1] +
						inputImage_MHC[row + 2][col][BytesPerPixel - 1]) +
						(-1.0) * (inputImage_MHC[row - 1][col - 1][BytesPerPixel - 1] +
							inputImage_MHC[row][col - 2][BytesPerPixel - 1] +
							inputImage_MHC[row + 1][col - 1][BytesPerPixel - 1] +
							inputImage_MHC[row + 1][col + 1][BytesPerPixel - 1] +
							inputImage_MHC[row][col + 2][BytesPerPixel - 1] +
							inputImage_MHC[row - 1][col + 1][BytesPerPixel - 1]) +
							(4.0) * (inputImage_MHC[row][col - 1][BytesPerPixel - 1] +
								inputImage_MHC[row][col + 1][BytesPerPixel - 1]) +
								(5.0) * (inputImage_MHC[row][col][BytesPerPixel - 1]));
			}


			//Case2(MHC filter7)
			//Estimation B at green in R row, B column
			else if ((row % 2) == 0 && (col % 2) == 0 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1) && row != 1 && row != (height - 2) && col != 1 && col != (width - 2)) {
				imageData_blue_MHC[row][col][BytesPerPixel - 1] =
					(1.0 / 8.0) * ((1.0 / 2.0) * (inputImage_MHC[row][col - 2][BytesPerPixel - 1] +
						inputImage_MHC[row][col + 2][BytesPerPixel - 1]) +
						(-1.0) * (inputImage_MHC[row - 2][col][BytesPerPixel - 1] +
							inputImage_MHC[row - 1][col - 1][BytesPerPixel - 1] +
							inputImage_MHC[row + 1][col - 1][BytesPerPixel - 1] +
							inputImage_MHC[row + 2][col][BytesPerPixel - 1] +
							inputImage_MHC[row + 1][col + 1][BytesPerPixel - 1] +
							inputImage_MHC[row - 1][col + 1][BytesPerPixel - 1]) +
							(4.0) * (inputImage_MHC[row - 1][col][BytesPerPixel - 1] +
								inputImage_MHC[row + 1][col][BytesPerPixel - 1]) +
								(5.0) * (inputImage_MHC[row][col][BytesPerPixel - 1]));
			}

			//Case3(MHC filter8)
			//Estimation B at red in R row , R column
			else if ((row % 2) == 0 && (col % 2) == 1 && row != 0 && row != (height - 1) && col != 0 && col != (width - 1) && row != 1 && row != (height - 2) && col != 1 && col != (width - 2)) {
				imageData_blue_MHC[row][col][BytesPerPixel - 1] =
					(1.0 / 8.0) * ((2.0) * (inputImage_MHC[row - 1][col - 1][BytesPerPixel - 1] +
						inputImage_MHC[row - 1][col + 1][BytesPerPixel - 1] +
						inputImage_MHC[row + 1][col - 1][BytesPerPixel - 1] +
						inputImage_MHC[row + 1][col + 1][BytesPerPixel - 1]) +
						(-3.0 / 2.0) * (inputImage_MHC[row - 2][col][BytesPerPixel - 1] +
							inputImage_MHC[row][col - 2][BytesPerPixel - 1] +
							inputImage_MHC[row + 2][col][BytesPerPixel - 1] +
							inputImage_MHC[row][col + 2][BytesPerPixel - 1]) +
							(6.0) * (inputImage_MHC[row][col][BytesPerPixel - 1]));
			}

		}
	}



	//**Pre Processing Handle overflow issue between unsigned char and double type mannually
	
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if (imageData_red_MHC[row][col][0] < 0) {
				imageData_red_MHC[row][col][0] = 0.0;
			}
			else if (imageData_red_MHC[row][col][0] > 255.0) {
				imageData_red_MHC[row][col][0] = 255.0;
			}

		}
	}


	
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if (imageData_blue_MHC[row][col][0] < 0) {
				imageData_blue_MHC[row][col][0] = 0.0;
			}
			else if (imageData_blue_MHC[row][col][0] > 255.0) {
				imageData_blue_MHC[row][col][0] = 255.0;
			}

		}
	}

	
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if (imageData_green_MHC[row][col][0] < 0) {
				imageData_green_MHC[row][col][0] = 0.0;
			}
			else if (imageData_green_MHC[row][col][0] > 255.0) {
				imageData_green_MHC[row][col][0] = 255.0;
			}

		}
	}



	//Save the result image as raw file
	ImageWriteSave saveObj1(height, width, savePathDogMHC);

	saveObj1.saveAsRawfileColor(imageData_red_MHC, imageData_green_MHC, imageData_blue_MHC);













	//**************************Image Denoising*******************************
	int BytesPerPixel_Corn = 1;
	int width_Corn = 320;
	int height_Corn = 320;

	//create a ImageLoadRead Object for Corn image
	ImageLoadRead readObjCorn(height_Corn, width_Corn, BytesPerPixel_Corn, readPathCorn1);
	//load and read corn image
	readObjCorn.rawImgLoad();

	//get image data
	std::vector<std::vector<std::vector<double>>> inputCorn = readObjCorn.getImageData();


	//Create Proessor for Denoising algorithms
	ImageProcessor cornImagProcessor(height_Corn, width_Corn, BytesPerPixel_Corn, inputCorn);


	//****************Mean filter****************************
	//define Mean filter
	int sizeOfFilter_mean = 5;
	int randomNum = 1;

	Filters filterObj(randomNum);

	std::vector<std::vector<double>> meanFilter_2D = filterObj.meanFilterGenerator(sizeOfFilter_mean);



	//expend image data
	std::vector<std::vector<std::vector<double>>> expended_inputCorn = cornImagProcessor.boundaryExtension(meanFilter_2D);

	//filter data
	std::vector<std::vector<std::vector<double>>> filtered_Corn = cornImagProcessor.convolution(meanFilter_2D, expended_inputCorn);

	//crop a extended matrix data
	std::vector<std::vector<std::vector<double>>> final_Corn = cornImagProcessor.matrixCropper(meanFilter_2D, filtered_Corn);

	//save corn_mean
	//save buffer as raw file
	ImageWriteSave saveObjCorn(height_Corn, width_Corn, savePathCorn1);

	saveObjCorn.saveAsRawfile(final_Corn);


	//********************Gaussian Filter********************
	//get gaussian filter
	std::vector<std::vector<double>> gaussianFilter = filterObj.gaussianFilterGenerator();

	//Expend Image
	std::vector<std::vector<std::vector<double>>> expended_inputCorn_gaus = cornImagProcessor.boundaryExtension(gaussianFilter);

	//Filtering Process witl convolution operation
	std::vector<std::vector<std::vector<double>>> filtered_Corn_gaus = cornImagProcessor.convolution(gaussianFilter, expended_inputCorn_gaus);

	//Crop the filtered matrix
	std::vector<std::vector<std::vector<double>>> final_Corn_gaus = cornImagProcessor.matrixCropper(gaussianFilter, filtered_Corn_gaus);


	////Save corn_gaus
	ImageWriteSave saveObjCorn_gaus(height_Corn, width_Corn, savePathCorn2);

	saveObjCorn_gaus.saveAsRawfile(final_Corn_gaus);

	////compare PSNR between mean filter and gaussian filter

	ImageLoadRead readObjCorn_NoNoise(height_Corn, width_Corn, BytesPerPixel_Corn, readPathCornNoNoise);
	////load and read noise free corn image
	readObjCorn_NoNoise.rawImgLoad();

	////get noise free image data
	std::vector<std::vector<std::vector<double>>> noiseFreeCorn = readObjCorn_NoNoise.getImageData();

	cout << "PSNR by mean filter: " << cornImagProcessor.PSNRcalculator(noiseFreeCorn, final_Corn) << " (dB)" << endl;

	cout << "PSNR by gaussian filter: " << cornImagProcessor.PSNRcalculator(noiseFreeCorn, final_Corn_gaus) << " (dB)" << endl;



	//*************************Bilateral Denoising****************************
	int sizeOfFilter_Bi_Deno = sizeOfFilter_mean; // use the same size of filter in this case 5x5 size

	//Get a expend Image from a image extended by gaussian filter size
	std::vector<std::vector<std::vector<double>>> expended_inputCorn_Bi_Deno = expended_inputCorn_gaus; //it is okay to copy this value because it is just a extended matrix by filter size

	//Filter noise of the image by bilateral denoising technique
	double final_sigma_c = 0.0;
	double final_sigma_s = 0.0; //these two value can be decided by user who applies this filtering algorithm.

	std::vector<double> sigma_c_Storage(10);
	std::vector<double> sigma_s_Storage(10);


	
	

	//Create and save possible deviation values to strogages
	for (int i = 0; i < sigma_c_Storage.size(); i++) {
		sigma_c_Storage[i] = (i + 1.0) * 100.0; //e.g. possible values for sigma c and sigma s are from 100 to 1000
		sigma_s_Storage[i] = (i + 1.0) * 100.0;
	}


	//iterate Bilateral denoising and compare PSNR and choose a (sigma_c,sigma_s) pair that gives highest PSNR value.
	for (int i_th_sigma_c = 0; i_th_sigma_c < sigma_c_Storage.size(); i_th_sigma_c++) {
		for (int j_th_sigma_s = 0; j_th_sigma_s < sigma_s_Storage.size(); j_th_sigma_s++) {
			std::vector<std::vector<std::vector<double>>> filtered_Corn_Bi_Deno = cornImagProcessor.BilateralDenoising(expended_inputCorn_Bi_Deno, sizeOfFilter_Bi_Deno, sigma_c_Storage[i_th_sigma_c], sigma_s_Storage[j_th_sigma_s]);

			//Crop the filtered matrix
			//Gaussian filter is used as a parameter. This is okay because in the function it only uses size of the gaussian filter.
			std::vector<std::vector<std::vector<double>>> cropped_Corn_Bi_Deno = cornImagProcessor.matrixCropper(gaussianFilter, filtered_Corn_Bi_Deno);

			cout << "PSNR by bilateral filter : " << cornImagProcessor.PSNRcalculator(noiseFreeCorn, cropped_Corn_Bi_Deno) << " (dB)" << endl;
			cout << "Corresponding sigma c value: " << sigma_c_Storage[i_th_sigma_c] << endl;
			cout << "Corresponding sigma s value: " << sigma_s_Storage[j_th_sigma_s] << endl;
			cout << endl;

		}
	}

	//assign the fianl sigma c and sigma s value that give highest PSNR
	final_sigma_c = 100.0;
	final_sigma_s = 100.0;

	//image processing aggain with final sigmal c and sigma s
	std::vector<std::vector<std::vector<double>>> final_filtered_Corn_Bi_Deno = cornImagProcessor.BilateralDenoising(expended_inputCorn_Bi_Deno, sizeOfFilter_Bi_Deno, final_sigma_c, final_sigma_s);

	//Crop the filtered matrix
	//Gaussian filter is used as a parameter. This is okay because in the function it only uses size of the gaussian filter.
	std::vector<std::vector<std::vector<double>>> final_cropped_Corn_Bi_Deno = cornImagProcessor.matrixCropper(gaussianFilter, final_filtered_Corn_Bi_Deno);


	//Save final cropped corn_Bi_Deno
	ImageWriteSave saveObjCorn_Bi_Deno(height_Corn, width_Corn, savePathCornBiDenoise);

	saveObjCorn_Bi_Deno.saveAsRawfile(final_cropped_Corn_Bi_Deno);
	cout << "PSNR by final bilateral filter : " << cornImagProcessor.PSNRcalculator(noiseFreeCorn, final_cropped_Corn_Bi_Deno) << " (dB)" << endl;








	//***********************Non Local Mean Denoising Algorithm
	int sizeOfLargeWindow = 5;
	int sizeOfSmallWindow = 3;
	double final_hyperparameter_a = 0;
	double final_hyperparameter_h = 0;
	std::vector<double> hyperparameter_a_Storage(10);
	std::vector<double> hyperparameter_h_Storage(10);

	//Create and save possible hyper parameter values to strogages
	for (int i = 0; i < hyperparameter_a_Storage.size(); i++) {
		hyperparameter_a_Storage[i] = (i + 1.0) * 10.0; //e.g. possible values for a and h are from 10 to 100
		hyperparameter_h_Storage[i] = (i + 1.0) * 10.0; 
	}

	//iterate NLM denoising and compare PSNR and choose a (a,h) pair that gives highest PSNR value.
	for (int i_th_a = 0; i_th_a < hyperparameter_a_Storage.size(); i_th_a++) {
		for (int j_th_h = 0; j_th_h < hyperparameter_h_Storage.size(); j_th_h++) {
			std::vector<std::vector<std::vector<double>>> filteredCorn_NLM = cornImagProcessor.NonLocalMeanDenoising(inputCorn, sizeOfLargeWindow, sizeOfSmallWindow, hyperparameter_h_Storage[j_th_h], hyperparameter_a_Storage[i_th_a]);
			cout << "PSNR by NLM filter : " << cornImagProcessor.PSNRcalculator(noiseFreeCorn, filteredCorn_NLM) << " (dB)" << endl;
			cout << "Corresponding a value: " << hyperparameter_a_Storage[i_th_a] << endl;
			cout << "Corresponding h value: " << hyperparameter_h_Storage[j_th_h] << endl;
			cout << endl;

		}
	}
	//assign the fianl a and h value that give highest PSNR
	final_hyperparameter_a = 20.0;
	final_hyperparameter_h = 10.0;

	std::vector<std::vector<std::vector<double>>> final_filteredCorn_NLM = cornImagProcessor.NonLocalMeanDenoising(inputCorn, sizeOfLargeWindow, sizeOfSmallWindow, final_hyperparameter_h, final_hyperparameter_a);


	//save corn_NLM
	//save buffer as raw file
	ImageWriteSave saveObjCorn_NLM(height_Corn, width_Corn, savePathCornNLMDenoise);

	saveObjCorn_NLM.saveAsRawfile(final_filteredCorn_NLM);



	



	

	



	//****************************Histogram Manipulation*******************************(Since Method B takes 6 mins, code for it is located at the end.)
	int BytePerPixel_Toy = 3;
	int width_Toy = 400;
	int height_Toy = 560;
	int BytePerPixel_Toy_OneChannel = 1;
	//Create a ImageLoadRead Object for Toy image
	ImageLoadRead readObjToy(height_Toy, width_Toy, BytePerPixel_Toy, readPathToy);
	//load and read Toy image
	readObjToy.rawImgLoad();

	//get image data for each channel such as red, green, blue;
	std::vector<std::vector<std::vector<double>>> redChannelToy = readObjToy.getRedChannel();
	std::vector<std::vector<std::vector<double>>> greenChannelToy = readObjToy.getGreenChannel();
	std::vector<std::vector<std::vector<double>>> blueChannelToy = readObjToy.getBlueChannel();

	//Create ImageProcessor Object for each channel
	//For Red Channel
	ImageProcessor toyImageProcessor_Red(height_Toy, width_Toy, BytePerPixel_Toy_OneChannel, redChannelToy);
	//For Green Channel
	ImageProcessor toyImageProcessor_Green(height_Toy, width_Toy, BytePerPixel_Toy_OneChannel, greenChannelToy);
	//For Blue Channel
	ImageProcessor toyImageProcessor_Blue(height_Toy, width_Toy, BytePerPixel_Toy_OneChannel, blueChannelToy);

	//************************Method B(Filling Buckets)*****************(It takes about 7mins)
	//Perform Histogram Manipulation Method B for each channel
	//For Red Channel
	std::vector<std::vector<std::vector<double>>> manipulated_redChannelToy_MethodB = toyImageProcessor_Red.hisManipulatorMethodB(redChannelToy);
	//For Green Channel
	std::vector<std::vector<std::vector<double>>> manipulated_greenChannelToy_MethodB = toyImageProcessor_Green.hisManipulatorMethodB(greenChannelToy);
	//For Blue Channel
	std::vector<std::vector<std::vector<double>>> manipulated_blueChannelToy_MethodB = toyImageProcessor_Blue.hisManipulatorMethodB(blueChannelToy);

	//save buffer as raw file
	ImageWriteSave saveObjToy_MethodB(height_Toy, width_Toy, savePathToyMethodB);

	saveObjToy_MethodB.saveAsRawfileColor(manipulated_redChannelToy_MethodB, manipulated_greenChannelToy_MethodB, manipulated_blueChannelToy_MethodB);


	//************************Method A(Transfer Function)*****************
	//Perfor Histogram Manipulation Method A for each channel
	//Method A Step1
	//Count number of each intensity in one channel and return a corresponding 1D vector that includes the number of appearance of each intensity
	//For Red Channel
	std::vector<int> intensityFreq_RedChannel = toyImageProcessor_Red.intensityCounterOneChannel(redChannelToy);
	//For Green Channel
	std::vector<int> intensityFreq_GreenChannel = toyImageProcessor_Green.intensityCounterOneChannel(greenChannelToy);
	//For Blue Channel
	std::vector<int> intensityFreq_BlueChannel = toyImageProcessor_Blue.intensityCounterOneChannel(blueChannelToy);


	//Method A Step2
	//Calculate Normalized Probability and return a 1D vector that includes normalized probability for each intensity
	//For Red Channel
	std::vector<double> normalizedProb_RedChannel = toyImageProcessor_Red.normalizedProbCalculator(intensityFreq_RedChannel);
	//For Green Channel
	std::vector<double> normalizedProb_GreenChannel = toyImageProcessor_Green.normalizedProbCalculator(intensityFreq_GreenChannel);
	//For Blue Channel
	std::vector<double> normalizedProb_BlueChannel = toyImageProcessor_Blue.normalizedProbCalculator(intensityFreq_BlueChannel);


	//Method A Step3
	//Calculate Cummulative Density Function(CDF) for each intensity and return a 1D vector than includes CDF for each intensity
	//For Red Channel
	std::vector<double> CDF_RedChannel = toyImageProcessor_Red.CDFCalculator(normalizedProb_RedChannel);
	//For Green Channel
	std::vector<double> CDF_GreenChannel = toyImageProcessor_Green.CDFCalculator(normalizedProb_GreenChannel);
	//For Blue Channel
	std::vector<double> CDF_BlueChannel = toyImageProcessor_Blue.CDFCalculator(normalizedProb_BlueChannel);


	//Method A Step4
	//Generate and return transfer function(i.e., mapping function) that maps original intensity for each pixel to manipulated intensity 
	//For Red Channel
	std::vector<double> mappingVec_RedChannel = toyImageProcessor_Red.mappingFuncGenerator(CDF_RedChannel);
	//For Green Channel
	std::vector<double> mappingVec_GreenChannel = toyImageProcessor_Green.mappingFuncGenerator(CDF_GreenChannel);
	//For Blue Channel
	std::vector<double> mappingVec_BlueChannel = toyImageProcessor_Blue.mappingFuncGenerator(CDF_BlueChannel);


	//Method A Step5
    //Perform histogram manipulation method A to input one channel image and return it.
	//For Red Channel
	std::vector<std::vector<std::vector<double>>> manipulated_RedCh_MethodA = toyImageProcessor_Red.hisManipulatorMethodA(mappingVec_RedChannel, redChannelToy);
	//For Green Channel
	std::vector<std::vector<std::vector<double>>> manipulated_GreenCh_MethodA = toyImageProcessor_Green.hisManipulatorMethodA(mappingVec_GreenChannel, greenChannelToy);
	//For Blue Channel
	std::vector<std::vector<std::vector<double>>> manipulated_BlueCh_MethodA = toyImageProcessor_Blue.hisManipulatorMethodA(mappingVec_BlueChannel, blueChannelToy);


	//save buffer as raw file
	ImageWriteSave saveObjToy_MethodA(height_Toy, width_Toy, savePathToyMethodA);

	saveObjToy_MethodA.saveAsRawfileColor(manipulated_RedCh_MethodA, manipulated_GreenCh_MethodA, manipulated_BlueCh_MethodA);


	
	

	std::cout << "succeed" << endl;
	

	return 0;
}



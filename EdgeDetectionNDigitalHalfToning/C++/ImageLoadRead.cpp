#include "ImageLoadRead.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

ImageLoadRead::ImageLoadRead(int height, int width, int BytesPerPixel, std::string path1) {
	imgHeight = height;
	imgWidth = width;
	imgBytesPerPixel = BytesPerPixel;
	imgPath1 = path1;


}

ImageLoadRead::~ImageLoadRead() {}


void ImageLoadRead::readImgFromBuffer(unsigned char* buffer) {
	int buff_index = 0;
	std::vector<double> image_1D(imgBytesPerPixel);
	std::vector<std::vector<double>> image_2D(imgWidth, image_1D);
	std::vector<std::vector<std::vector<double>>> image_3D(imgHeight, image_2D);

	imgData = image_3D;



	for (int h = 0; h < imgHeight; h++) {
		for (int w = 0; w < imgWidth; w++) {
			for (int b = 0; b < imgBytesPerPixel; b++) {
				imgData[h][w][b] = buffer[buff_index];
				buff_index++;
			}
		}
	}



}


void ImageLoadRead::rawImgLoad() {
	FILE* file;
	if (!(file = fopen(imgPath1.data(), "rb"))) {
		std::cout << "Cannot open file: " << imgPath1 << std::endl;
		exit(1);
	}
	//load .raw file to buffer 
	unsigned char* buffer = new unsigned char[imgWidth * imgHeight * imgBytesPerPixel];


	fread(buffer, sizeof(unsigned char), imgWidth * imgHeight * imgBytesPerPixel, file);
	fclose(file);

	this->readImgFromBuffer(buffer);


}


std::vector<std::vector<std::vector<double>>> ImageLoadRead::getRedChannel() {
	int oneChannel = 1;
	int redChannel = 0; //it represents red channel in imgData variable
	std::vector<double> red_1D(oneChannel);
	std::vector<std::vector<double>> red_2D(imgWidth, red_1D);
	std::vector<std::vector<std::vector<double>>> red_3D(imgHeight, red_2D);

	for (int row = 0; row < imgHeight; row++) {
		for (int col = 0; col < imgWidth; col++) {
			red_3D[row][col][oneChannel - 1] = imgData[row][col][redChannel];//Always be careful with index for Byte per pixel part
		}
	}

	return red_3D;
}

std::vector<std::vector<std::vector<double>>> ImageLoadRead::getGreenChannel() {
	int oneChannel = 1;
	int greenChannel = 1; //it represents green channel in imgData variable
	std::vector<double> green_1D(oneChannel);
	std::vector<std::vector<double>> green_2D(imgWidth, green_1D);
	std::vector<std::vector<std::vector<double>>> green_3D(imgHeight, green_2D);

	for (int row = 0; row < imgHeight; row++) {
		for (int col = 0; col < imgWidth; col++) {
			green_3D[row][col][oneChannel - 1] = imgData[row][col][greenChannel]; //Always be careful with index for Byte per pixel part
		}
	}

	return green_3D;
}


std::vector<std::vector<std::vector<double>>> ImageLoadRead::getBlueChannel() {
	int oneChannel = 1;
	int blueChannel = 2; //it represents blue channel in imgData variable
	std::vector<double> blue_1D(oneChannel);
	std::vector<std::vector<double>> blue_2D(imgWidth, blue_1D);
	std::vector<std::vector<std::vector<double>>> blue_3D(imgHeight, blue_2D);

	for (int row = 0; row < imgHeight; row++) {
		for (int col = 0; col < imgWidth; col++) {
			blue_3D[row][col][oneChannel - 1] = imgData[row][col][blueChannel]; //Always be careful with index for Byte per pixel part
		}
	}
	return blue_3D;
}



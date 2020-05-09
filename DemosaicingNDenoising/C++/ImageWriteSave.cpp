#include "ImageWriteSave.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>


//constructor
ImageWriteSave::ImageWriteSave(int height, int width, std::string path) {
	imgHeight = height;
	imgWidth = width;
	savePath = path;
	
}

//destructor
ImageWriteSave::~ImageWriteSave() {}

//savig and writing method for one channel image file
unsigned char* ImageWriteSave::writeToBuffer(unsigned char* buffer, std::vector<std::vector<std::vector<double>>> oneChannel) {
	int buff_index = 0;
	unsigned char* result_buffer = buffer;
	std::vector<std::vector<std::vector<double>>> oneChannelData = oneChannel;
	int bytePerPixel = 1;
	for (int h = 0; h < imgHeight; h++) {
		for (int w = 0; w < imgWidth; w++) {
		
			

			result_buffer[buff_index] = oneChannelData[h][w][bytePerPixel-1] ;

			buff_index++;
		}

	}
	return result_buffer;
}

void ImageWriteSave::saveAsRawfile(std::vector<std::vector<std::vector<double>>> oneChannel) {
	int bytePerPixel = 1;
	std::vector<std::vector<std::vector<double>>> oneChannelData = oneChannel;
	FILE* file;
	if (!(file = fopen(savePath.data(), "wb"))) {
		std::cout << "Cannot open file: " << savePath << std::endl;
		exit(1);
	}
	//save buffer to .raw file 
	unsigned char* buffer = new unsigned char[imgWidth * imgHeight * bytePerPixel];

	unsigned char* result_buffer = this->writeToBuffer(buffer, oneChannelData);

	fwrite(result_buffer, sizeof(unsigned char), imgWidth * imgHeight * bytePerPixel, file);
	fclose(file);
}


//savig and writing method for 3 channels color image file
unsigned char* ImageWriteSave::writeToBufferColor(unsigned char* buffer, std::vector<std::vector<std::vector<double>>> red, std::vector<std::vector<std::vector<double>>> green, std::vector<std::vector<std::vector<double>>> blue) {
	int buff_index = 0;
	unsigned char* result_buffer = buffer;
	std::vector<std::vector<std::vector<double>>> redChannelData = red;
	std::vector<std::vector<std::vector<double>>> greenChannelData = green;
	std::vector<std::vector<std::vector<double>>> blueChannelData = blue;
	int bytePerPixel = 1;

	for (int h = 0; h < imgHeight; h++) {
		for (int w = 0; w < imgWidth; w++) {
	
		

			result_buffer[buff_index] = redChannelData[h][w][bytePerPixel - 1];
			buff_index++;

			result_buffer[buff_index] = greenChannelData[h][w][bytePerPixel - 1];
			buff_index++;

			result_buffer[buff_index] = blueChannelData[h][w][bytePerPixel - 1];
			buff_index++;
		}

	}
	return result_buffer;
}


void ImageWriteSave::saveAsRawfileColor(std::vector<std::vector<std::vector<double>>> red, std::vector<std::vector<std::vector<double>>> green, std::vector<std::vector<std::vector<double>>> blue) {
	int finalImgBytePerPixel = 3; //this is 3 for color image(RGB per pixel)
	std::vector<std::vector<std::vector<double>>> redChannelData = red;
	std::vector<std::vector<std::vector<double>>> greenChannelData = green;
	std::vector<std::vector<std::vector<double>>> blueChannelData = blue;

	FILE* file;
	if (!(file = fopen(savePath.data(), "wb"))) {
		std::cout << "Cannot open file: " << savePath << std::endl;
		exit(1);
	}
	//save buffer to .raw file 
	unsigned char* buffer = new unsigned char[imgWidth * imgHeight * finalImgBytePerPixel];

	unsigned char* result_buffer = this->writeToBufferColor(buffer, redChannelData, greenChannelData, blueChannelData);

	fwrite(result_buffer, sizeof(unsigned char), imgWidth * imgHeight * finalImgBytePerPixel, file);
	fclose(file);
}



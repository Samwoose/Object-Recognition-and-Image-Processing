#pragma once
#ifndef IMAGELOADREAD_H
#define IMAGELOADREAD_H
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

class ImageLoadRead {
private:
	int imgHeight;
	int imgWidth;
	int imgBytesPerPixel;
	std::string imgPath1;

	std::vector<std::vector<std::vector<double>>> imgData;



public:
	//constructor 
	ImageLoadRead(int height, int width, int BytePerPixel, std::string imgPath1);

	//destructor
	~ImageLoadRead();

	void readImgFromBuffer(unsigned char* buffer);

	void rawImgLoad();

	//return image data
	std::vector<std::vector<std::vector<double>>> getImageData() {
		return imgData;
	}

	void setImageData(std::vector<std::vector<std::vector<double>>> imageData) {
		imgData = imageData;
	}

	std::vector<std::vector<std::vector<double>>> getRedChannel();

	std::vector<std::vector<std::vector<double>>> getGreenChannel();

	std::vector<std::vector<std::vector<double>>> getBlueChannel();

	int getImgHeight() { return imgHeight; }

	int getImgWidth() { return imgWidth; }

	int getImgBytePerPixel() { return imgBytesPerPixel; }

};


#endif // !	IMAGELOADREAD_H

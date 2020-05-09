#pragma once

#ifndef IMAGEWRITESAVE_H
#define IMAGEWRITESAVE_H
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

class ImageWriteSave
{
private:
	int imgHeight;
	int imgWidth;
	std::string savePath;

public:
	//constructor
	ImageWriteSave(int height, int width, std::string path);

	//destructor
	~ImageWriteSave();

	//class methodes for gray raw file
	void saveAsRawfile(std::vector<std::vector<std::vector<double>>> oneChannel);

	unsigned char* writeToBuffer(unsigned char* buffer, std::vector<std::vector<std::vector<double>>> oneChannel);

	//class methodes for color image raw file
	void saveAsRawfileColor(std::vector<std::vector<std::vector<double>>> red, std::vector<std::vector<std::vector<double>>> green, std::vector<std::vector<std::vector<double>>> blue);

	unsigned char* writeToBufferColor(unsigned char* buffer, std::vector<std::vector<std::vector<double>>> red, std::vector<std::vector<std::vector<double>>> green, std::vector<std::vector<std::vector<double>>> blue);



};



#endif
// This sample code reads in image data from a RAW image file and 
// writes it into another file

// NOTE:	The code assumes that the image is of size 256 x 256 and is in the
//			RAW format. You will need to make corresponding changes to
//			accommodate images of different sizes and/or types

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include "ImageProcessor.h"

using namespace std;

int main(int argc, char* argv[])

{
	// Define file pointer and variables
	int BytesPerPixel = 1; 
	int width = 390;
	int height = 300;
	
	//string readPath1 = argv[1];
	//string savePath2 = argv[2];

	string readPath1 = "C:\Users\tjdtk\HW1_Images\cat";
	string savePath2 = "C:\Users\tjdtk\HW1_Images\cat1";
	
	ImageProcessor obj(height, width, BytesPerPixel, readPath1, savePath2);
	//load image
	obj.rawImgLoad();
	
	obj.saveAsRawfile();



	cout << "succeed" << endl;

	return 0;
}



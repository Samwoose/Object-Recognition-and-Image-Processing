#include "Filters.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>




Filters::Filters(int tep) {
	temp = tep;
}
Filters::~Filters() {}

std::vector<std::vector<double>> Filters::meanFilterGenerator(int filterSizeOfOneside) {
	
	if ((filterSizeOfOneside % 2) == 0) {
		std::cout << "Please use odd number for filter size of one size" << std::endl;
	}

	std::vector<double> meanFilter_1D(filterSizeOfOneside);
	std::vector<std::vector<double>> meanFilter_2D(filterSizeOfOneside, meanFilter_1D);

	for (int i = 0; i < filterSizeOfOneside; i++) {
		for (int j = 0; j < filterSizeOfOneside; j++) {
			meanFilter_2D[i][j] = 1.0;

		}
	}

	return meanFilter_2D;
}


std::vector<std::vector<double>> Filters::gaussianFilterGenerator() {
	//For now 5*5 size gaussian
	std::vector<double> gaussianFilter_1D(5);
	std::vector<std::vector<double>> gaussianFilter_2D(5, gaussianFilter_1D);

	gaussianFilter_2D[0][0] = 1;
	gaussianFilter_2D[0][1] = 4;
	gaussianFilter_2D[0][2] = 7;
	gaussianFilter_2D[0][3] = 4;
	gaussianFilter_2D[0][4] = 1;

	gaussianFilter_2D[1][0] = 4;
	gaussianFilter_2D[1][1] = 16;
	gaussianFilter_2D[1][2] = 26;
	gaussianFilter_2D[1][3] = 16;
	gaussianFilter_2D[1][4] = 4;

	gaussianFilter_2D[2][0] = 7;
	gaussianFilter_2D[2][1] = 26;
	gaussianFilter_2D[2][2] = 41;
	gaussianFilter_2D[2][3] = 26;
	gaussianFilter_2D[2][4] = 7;

	gaussianFilter_2D[3][0] = 4;
	gaussianFilter_2D[3][1] = 16;
	gaussianFilter_2D[3][2] = 26;
	gaussianFilter_2D[3][3] = 16;
	gaussianFilter_2D[3][4] = 4;

	gaussianFilter_2D[4][0] = 1;
	gaussianFilter_2D[4][1] = 4;
	gaussianFilter_2D[4][2] = 7;
	gaussianFilter_2D[4][3] = 4;
	gaussianFilter_2D[4][4] = 1;

	return gaussianFilter_2D;

}
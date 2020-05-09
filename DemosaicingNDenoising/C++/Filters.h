#pragma once

#ifndef FILTERS_H
#define FILTERS_H
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>



class Filters
{


private:
	int temp;
	

public:
	//constructor 
	Filters(int tep);

	//destructor
	~Filters();

	//Low Pass Filter(Mean)
	//Parameter, filterSizeOfOneside should be always odd number
	std::vector<std::vector<double>> meanFilterGenerator(int filterSizeOfOneside);

	//Low Pass Filter(Gaussian)
	//Parameter, filterSizeOfOneside should be always odd number. For now 5X5 gaussian filter
	std::vector<std::vector<double>> gaussianFilterGenerator();

	
};



#endif // ! IMAGEPROCESSOR_H


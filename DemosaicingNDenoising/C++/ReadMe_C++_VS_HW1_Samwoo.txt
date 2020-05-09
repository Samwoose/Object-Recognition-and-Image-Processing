# EE569 Homework Assignment #1
# Date: January 28, 2020
# Name: Samwoo Seong
# ID: 7953-6137-66
# email: samwoose@usc.edu
#
# Software: Visual Studio 2019
========================================================================
    CONSOLE APPLICATION : [Problem1 and Problem2] Project Overview
========================================================================

This file contains a summary of what you will find in each of the files that
make up your [Problem1 and Problem2] application.

Steps:

1. Open HW1.sln.
	This is the solution for the project.

2. Open main_HW1.cpp.
	This is the main program for problem 1 and 2.
3. All the header files and the source files have already been included in the project. 

4. Change paths of the input file in lines 27, 32, 36, 41 and the output files in lines 28, 29, 33, 34, 37, 38, 42, 43 on command line argument.
**Instruction to change command line arguments in visual studio 2019
Click "Project" -> "name of solution properties..." -> "Configuration Properties" -> "Debugging" -> "Command Arguments" -> "<Edit...>"

**Summary for each line
Paths for the input files
Line 27: path for Dog.raw
Line 32: path for Corn_noisy.raw
Line 36: path for Corn_gray.raw
Line 41: path for Toy.raw

Paths for the output files
Line 28: path for bilinear demosaicing result file
Line 29: path for MHC demosaicing result file
Line 33: path for mean filter result file
Line 34: path for gaussain filter result file
Line 37: path for Bilateral filter result file
Line 38: path for Non Local Mean filter result file
Line 42: path for historgam manipulation method A result file
Line 43: path for historgam manipulation method B result file

example of command line arguments in visual studio 2019
"C:\Users\tjdtk\HW1_Images\Dog.raw" "C:\Users\tjdtk\HW1_Images\Dog_Bi.raw" "C:\Users\tjdtk\HW1_Images\Dog_MHC.raw" "C:\Users\tjdtk\HW1_Images\Corn_noisy.raw" "C:\Users\tjdtk\HW1_Images\Corn_mean.raw" "C:\Users\tjdtk\HW1_Images\Corn_gaus.raw" "C:\Users\tjdtk\HW1_Images\Corn_gray.raw" "C:\Users\tjdtk\HW1_Images\Corn_Bi_Deno.raw" "C:\Users\tjdtk\HW1_Images\Corn_NLM.raw" "C:\Users\tjdtk\HW1_Images\Toy.raw" "C:\Users\tjdtk\HW1_Images\Toy_MethodA.raw" "C:\Users\tjdtk\HW1_Images\Toy_MethodB.raw" 

6. Run the program and wait until program is done. It may take around 15 minutes depending on computer specification. This is due to optimization and sorting process in some algorithms 
Note: 
When you run program for the first time errors related to fopen and fread can occur. 
This is because in visual studio 2019 encourage users to use other safer functions to open and read file.
This error can be resolved as follows.

Select your project and click "Properties" in the context menu.

In the dialog, chose Configuration Properties -> C/C++ -> Preprocessor

In the field PreprocessorDefinitions add ;_CRT_SECURE_NO_WARNINGS to turn those warnings off

Reference: https://stackoverflow.com/questions/21873048/getting-an-error-fopen-this-function-or-variable-may-be-unsafe-when-complin
   

/////////////////////////////////////////////////////////////////////////////
Other standard files:
ImageLoadRead.cpp
 This is used to load raw file data to buffer in c++  and read the data to 2D matrix.
 It also includes methods to get each channel such as red, green and blue.
 It can handle both gray and color image.

ImageProcessor.cpp
 This is used to perform any image processing algorithm that is needed in problem 1 and 2 except for plotting and BM3D.   

ImageWriteSave.cpp
 This is used to write 2D matrix image data to buffer in c++ and save them as raw file
 It can handle both gray and color image.

Filters.cpp
 This is used to get filters for some algorithms such as mean and gaussian filter
/////////////////////////////////////////////////////////////////////////////
Other notes:

Please contact me via my email or phone in case any running error occurs.
Phone number : 213-245-6235
/////////////////////////////////////////////////////////////////////////////

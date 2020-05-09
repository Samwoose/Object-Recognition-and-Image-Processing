# EE569 Homework Assignment #2
# Date: February 16, 2020
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

1. Open HW2_EE569.sln.
	This is the solution for the project.

2. Open HW2_EE569.cpp.
	This is the main program for problem 1 and 2.
3. All the header files and the source files have already been included in the project. 

4. Change paths of the input file in lines mentioned below and the output files in lines mentioned below on command line argument.
**Instruction to change command line arguments in visual studio 2019
Click "Project" -> "name of solution properties..." -> "Configuration Properties" -> "Debugging" -> "Command Arguments" -> "<Edit...>"

**Summary for each line
Paths for the input files
Line 29: path for Dogs.raw
Line 33:  path for Dogs magnitude edge map

Line 37: path for Gallery.raw
Line 41:  path for Gallery magnitude edge map

Line 46: path for LightHouse.raw
Line 57: Path for Rose.raw

Paths for the output files
Line 30:  result file for Dogs x direction edge map
Line 31:  result file for Dogs y direction edge map
Line 32:  result file for Dogs magnitude edge map
Line 34:  result file for Dogs tunned edge map

Line 38:  result file for Gallery x direction edge map
Line 39:  result file for Gallery y direction edge map
Line 40:  result file for Gallery magnitude edge map
Line 42:  result file for Gallery tunned edge map

Line 47:  result file for fixed thresholding method
Line 48:  result file for random thresholding method
Line 49:  result file for I 2x2 method
Line 50:  result file for I 8x8 method
Line 51:  result file for I 32x32 method
Line 52:  result file for floyd method
Line 53:  result file for JJN method
Line 54:  result file for stucki method

Line 58:  result file for seperable color error diffusion method


example of command line arguments in visual studio 2019
"C:\Users\tjdtk\HW2_Images\Pr1\Dogs.raw" "C:\Users\tjdtk\HW2_Images\Pr1\Dogs_xEdge.raw" "C:\Users\tjdtk\HW2_Images\Pr1\Dogs_yEdge.raw" "C:\Users\tjdtk\HW2_Images\Pr1\Dogs_MagnitudeEdge.raw" "C:\Users\tjdtk\HW2_Images\Pr1\Gallery.raw" "C:\Users\tjdtk\HW2_Images\Pr1\Gallery_xEdge.raw" "C:\Users\tjdtk\HW2_Images\Pr1\Gallery_yEdge.raw" "C:\Users\tjdtk\HW2_Images\Pr1\Gallery_MagnitudeEdge.raw" "C:\Users\tjdtk\HW2_Images\Pr1\Gallery_TunnedEdge.raw" "C:\Users\tjdtk\HW2_Images\Pr1\Dogs_TunnedEdge.raw" "C:\Users\tjdtk\HW2_Images\Pr2\LightHouse.raw" "C:\Users\tjdtk\HW2_Images\Pr2\LightHouse_Fixed.raw" "C:\Users\tjdtk\HW2_Images\Pr2\LightHouse_Random.raw" "C:\Users\tjdtk\HW2_Images\Pr2\LightHouse_2X2.raw" "C:\Users\tjdtk\HW2_Images\Pr2\LightHouse_8X8.raw" "C:\Users\tjdtk\HW2_Images\Pr2\LightHouse_32X32.raw" "C:\Users\tjdtk\HW2_Images\Pr2\LightHouse_floyd.raw" "C:\Users\tjdtk\HW2_Images\Pr2\LightHouse_jjn.raw" "C:\Users\tjdtk\HW2_Images\Pr2\LightHouse_stucki.raw" "C:\Users\tjdtk\HW2_Images\Pr2\Rose.raw" "C:\Users\tjdtk\HW2_Images\Pr2\Rose_seperable.raw"

6. Run the program and wait until program is done. It may take around 5 minutes depending on computer specification.  
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
 This is written based on the given code readraw.cpp

ImageProcessorHW2.cpp
 This is used to perform any image processing algorithm that is needed in problem 1 and 2 except for problem1(b),(c),(d) and problem2(c-2).   

ImageWriteSave.cpp
 This is used to write 2D matrix image data to buffer in c++ and save them as raw file
 It can handle both gray and color image.
 This is written based on the given code writeraw.cpp

Filters.cpp
 This is used to get filters for some algorithms such as Floyd, JJN, and Stucki masks
/////////////////////////////////////////////////////////////////////////////
Other notes:

Please contact me via my email or phone in case any running error occurs.
Phone number : 213-245-6235
/////////////////////////////////////////////////////////////////////////////

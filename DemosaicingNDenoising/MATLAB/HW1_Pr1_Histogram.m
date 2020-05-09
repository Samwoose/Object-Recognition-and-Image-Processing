%Read .raw file

filename_Toy = "Toy.raw";
imgHeight_Toy = 560;
imgWidth_Toy = 400;
imgBytePerPixel_Toy = 3;

oriData_Toy = readraw_Color(filename_Toy, imgHeight_Toy, imgWidth_Toy, imgBytePerPixel_Toy);

%get each channel data
redChannel_data = getRedChannel(oriData_Toy,imgHeight_Toy, imgWidth_Toy);
greenChannel_data = getGreenChannel(oriData_Toy, imgHeight_Toy, imgWidth_Toy);
blueChannel_data = getBlueChannel(oriData_Toy, imgHeight_Toy, imgWidth_Toy);

%Count intensity to plot histogram for each channel
x_axis = [0:1:255];
%for red channel
redChannel_frequency = intensityCounterOneChannel(redChannel_data,imgHeight_Toy,imgWidth_Toy);
%for green channel
greenChannel_frequency = intensityCounterOneChannel(greenChannel_data,imgHeight_Toy,imgWidth_Toy);
%for blue channel
blueChannel_frequency = intensityCounterOneChannel(blueChannel_data,imgHeight_Toy,imgWidth_Toy);

%%methodA
%Find mapping function for each channel
%Red Channel
redChannel_mappingVector = mappingFuncGenerator(redChannel_frequency, imgHeight_Toy, imgWidth_Toy);
%Green Channel
greenChannel_mappingVector = mappingFuncGenerator(greenChannel_frequency, imgHeight_Toy, imgWidth_Toy);
%Blue Channel
blueChannel_mappingVector = mappingFuncGenerator(blueChannel_frequency, imgHeight_Toy, imgWidth_Toy);

%%Method B 
%Find cumulative histogram for each channel
numOfGrayScale = 256;
%for red channel
cumulativeHistoVector_Red = cumulativeHistogram(imgHeight_Toy,imgWidth_Toy,numOfGrayScale);
%for green channel
cumulativeHistoVector_Green = cumulativeHistogram(imgHeight_Toy,imgWidth_Toy,numOfGrayScale);
%for blue channel
cumulativeHistoVector_Blue = cumulativeHistogram(imgHeight_Toy,imgWidth_Toy,numOfGrayScale);


%plot histogram for each channel
%intensity values VS number of pixels for each intensity
figure(1)

plot(x_axis,redChannel_frequency)
title('Histogram for Red Channel')
xlabel('Intensity') 
ylabel('Frequency of Each Intensity')

figure(2)
plot(x_axis,greenChannel_frequency)
title('Histogram for Green Channel')
xlabel('Intensity') 
ylabel('Frequency of Each Intensity')


figure(3)
plot(x_axis,blueChannel_frequency)
title('Histogram for Blue Channel')
xlabel('Intensity') 
ylabel('Frequency of Each Intensity')

%plot transfer function for each channel 
%(intensity values VS mapped intensity Values)
figure(4)
plot(x_axis, redChannel_mappingVector)
title('Transfer Function for Red Channel')
xlabel('Original Intensity') 
ylabel('Mapping Manipulated Intensity')

figure(5)
plot(x_axis, greenChannel_mappingVector)
title('Transfer Function for Green Channel')
xlabel('Original Intensity') 
ylabel('Mapping Manipulated Intensity')

figure(6)
plot(x_axis, blueChannel_mappingVector)
title('Transfer Function for Blue Channel')
xlabel('Original Intensity') 
ylabel('Mapping Manipulated Intensity')

%plot cumulative histogram
figure(7)
plot(x_axis, cumulativeHistoVector_Red)
title('Cumulative Historam for Red Channel')
xlabel('Intensity') 
ylabel('Cumulated Frequency of Each Intensity')

figure(8)
plot(x_axis, cumulativeHistoVector_Green)
title('Cumulative Historam for Green Channel')
xlabel('Intensity') 
ylabel('Cumulated Frequency of Each Intensity')

figure(9)
plot(x_axis, cumulativeHistoVector_Blue)
title('Cumulative Historam for Blue Channel')
xlabel('Intensity') 
ylabel('Cumulated Frequency of Each Intensity')


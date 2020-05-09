function G = readraw_Color(filename,imgHeight,imgWidth,imgBytePerPixel)
%readraw - read RAW format grey scale image of square size into matrix G
% Usage:	G = readraw(filename)

	disp(['	Retrieving Image ' filename ' ...']);

	% Get file ID for file
	fid=fopen(filename,'rb');

	% Check if file exists
	if (fid == -1)
	  	error('can not open input image file press CTRL-C to exit \n');
	  	pause
	end

	% Get all the pixels from the image
	pixel = fread(fid, inf, 'uchar');

	% Close file
	fclose(fid);

	
	% Construct matrix
	G = zeros(imgHeight,imgWidth,imgBytePerPixel);

	% Write pixels into matrix
	buffer_index = 1;
    for row = 1:imgHeight
        for col = 1:imgWidth
            for bytePerPixel = 1:imgBytePerPixel
                G(row,col,bytePerPixel) = pixel(buffer_index); 
                buffer_index = buffer_index + 1;
            end
        end
    end

	
end %function

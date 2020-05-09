% Demo for Structured Edge Detector (please see readme.txt first).
%%additionally added part
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW2\HW2_MATLAB\Pr1_c\piotr_toolbox\toolbox\matlab')
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW2\HW2_MATLAB\Pr1_c\piotr_toolbox\toolbox\channels')
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW2\HW2_MATLAB\Pr1_c\piotr_toolbox\toolbox\images')
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW2\HW2_MATLAB\Pr1_c\edges-master')
addpath('C:\Users\tjdtk\Desktop1\EE 569\HW2\HW2_MATLAB\Pr1_c')
%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts); toc; % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=100;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=1;                 % set to true to enable nms

%% evaluate edge detector on BSDS500 (see edgesEval.m)
if(0), edgesEval( model, 'show',1, 'name','' ); end

%% detect edge and visualize results
DogsJPG = imread('C:\Users\tjdtk\Desktop1\EE 569\HW2\HW2_MATLAB\Pr1_c\Dogs.jpg');
tic, probabilityDogs=edgesDetect(DogsJPG,model); toc
figure(1); im(DogsJPG); figure(2); im(1-probabilityDogs);

GalleryJPG = imread('C:\Users\tjdtk\Desktop1\EE 569\HW2\HW2_MATLAB\Pr1_c\Gallery.jpg');
tic, ProbabilityGallery=edgesDetect(GalleryJPG,model); toc
figure(3); im(GalleryJPG); figure(4); im(1-ProbabilityGallery);

HelicopterJPG = imread('C:\Users\tjdtk\Desktop1\EE 569\HW2\HW2_MATLAB\Pr1_c\Helicopter.jpg');
tic, ProbabilityHelicopter=edgesDetect(HelicopterJPG,model); toc
figure(5); im(HelicopterJPG); figure(6); im(1-ProbabilityHelicopter);

%Binarize probability Edge Map and get binary edge map
probabilityThreshold = 0.2; %It is from the HW2 Description.

binaryEdgeMap_Dogs = ProbabilityBinarizer(probabilityDogs,probabilityThreshold);
binaryEdgeMap_Gallery = ProbabilityBinarizer(ProbabilityGallery,probabilityThreshold);
binaryEdgeMap_Helicopter = ProbabilityBinarizer(ProbabilityHelicopter,probabilityThreshold);

%plot binary edge map
figure(7)
im(binaryEdgeMap_Dogs)
figure(8)
im(binaryEdgeMap_Gallery)
figure(9)
im(1-binaryEdgeMap_Helicopter) %Inversed 0:Edge, 1:Background in the plot

%Convert 0~1 sacle to 0~255 to save the results as raw files;
scaled_EdgeMap_Dogs = scalerChanger255(binaryEdgeMap_Dogs);
scaled_EdgeMap_Gallery = scalerChanger255(binaryEdgeMap_Gallery);
%Inverse gray scale value. 0->Edges, 255->Background
inversed_scaled_EdgeMap_Dogs = inverseEdgeNBackground(scaled_EdgeMap_Dogs);
inversed_scaled_EdgeMap_Gallery = inverseEdgeNBackground(scaled_EdgeMap_Gallery);
%save two probability edge maps as raw files
probability_EdgeMap_Dogs = writeraw_gray(probabilityDogs,'Dogs_probability_SE.raw');
probability_EdgeMap_Gallery = writeraw_gray(ProbabilityGallery,'Gallery_probability_SE.raw');


%Save two images as raw files
scaled_EdgeMap_Dogs = writeraw_gray(inversed_scaled_EdgeMap_Dogs,'Dogs_Edge_SE.raw');
scaled_EdgeMap_Gallery = writeraw_gray(inversed_scaled_EdgeMap_Gallery,'Gallery_Edge_SE.raw');


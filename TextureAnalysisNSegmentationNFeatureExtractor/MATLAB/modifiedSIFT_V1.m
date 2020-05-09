function [featureFrame,descriptor] = modifiedSIFT_V1(I)
%MODIFIEDSIFT_V1 Summary of this function goes here
%Perform SIFT to extract features
%I : input image
%featureFrame: each column is feature frame. Its format: [X;Y;S;TH]
%X,Y = (fractional) center of frame
%S = scale
%TH = orientation in radians
%descriptor: Each
%   column of D is the descriptor of the corresponding frame in F. A
%   descriptor is a 128-dimensional vector of class UINT8.
%   Detailed explanation goes here



% Modified VL_DEMO_SIFT_BASIC  Demo: SIFT: basic functionality



% --------------------------------------------------------------------
%                                                        Load a figure
% --------------------------------------------------------------------


%image(I) ; 
% colormap gray ;
% axis equal ; axis off ; axis tight ;
% vl_demo_print('sift_basic_0') ;

% --------------------------------------------------------------------
%                                       Convert the to required format
% --------------------------------------------------------------------
I = single(rgb2gray(I)) ;

% clf ; %imagesc(I)
% axis equal ; axis off ; axis tight ;
% vl_demo_print('sift_basic_1') ;

% --------------------------------------------------------------------
%                                                             Run SIFT
% --------------------------------------------------------------------
[featureFrame,descriptor] = vl_sift(I) ;

% hold on ;
% perm = randperm(size(featureFrame,2)) ;
% sel  = perm(1:50) ;
% h1   = vl_plotframe(featureFrame(:,sel)) ; set(h1,'color','k','linewidth',3) ;
% h2   = vl_plotframe(featureFrame(:,sel)) ; set(h2,'color','y','linewidth',2) ;
% 
% vl_demo_print('sift_basic_2') ;
% 
% delete([h1 h2]);
% 
% h3 = vl_plotsiftdescriptor(descriptor(:,sel),featureFrame(:,sel)) ;
% set(h3,'color','k','linewidth',2) ;
% h4 = vl_plotsiftdescriptor(descriptor(:,sel),featureFrame(:,sel)) ;
% set(h4,'color','g','linewidth',1) ;
% h1   = vl_plotframe(featureFrame(:,sel)) ; set(h1,'color','k','linewidth',3) ;
% h2   = vl_plotframe(featureFrame(:,sel)) ; set(h2,'color','y','linewidth',2) ;
% 
% vl_demo_print('sift_basic_3') ;



end


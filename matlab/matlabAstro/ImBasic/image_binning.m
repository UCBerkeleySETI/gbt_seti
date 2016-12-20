function Image=image_binning(Image,Bin)
%--------------------------------------------------------------------------
% image_binning function                                           ImBasic
% Description: Bin an image (whole pixel binning)
% Input  : - Image to bin.
%          - [Y binning factor, X binning factor].
% Output : - 
% License: GNU general public license version 3
% Tested : Matlab R2015a
%     By : Eran O. Ofek                    Aug 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Image=image_binning(rand(10,10),[2 2])
% Reliable: 2
%--------------------------------------------------------------------------

[M,N]=size(Image); 

Image = sum(reshape(Image,Bin(1),[]) ,1 );
Image = reshape(Image,M./Bin(1),[]).'; 

Image = sum(reshape(Image,Bin(2),[]) ,1);
Image = reshape(Image,N./Bin(2),[]).';


function OutImage=medfilt_circ(Image,Radius)
%--------------------------------------------------------------------------
% medfilt_circ function                                            ImBasic
% Description: Run a 2-D circular median filter (ignoring NaN) on an image.
%              This function is very slow.
% Input  : - A 2D matrix.
%          - Radius of median filter.
% Output : - A 2D matrix representing the image after the median filtering.
% Tested : Matlab 7.13
%     By : Eran O. Ofek                    Jun 2012
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: OutImage=medfilt_circ(rand(1000,1000),5);
% Reliable: 2
%--------------------------------------------------------------------------


OutImage    = zeros(size(Image));
[MatX,MatY] = meshgrid((1:1:size(Image,2)),(1:1:size(Image,1)));

Npix = length(Image(:));
for Ipix=1:1:Npix,
   IndF =  ((MatX(Ipix)-MatX(:)).^2 + (MatY(Ipix)-MatY(:)).^2)<(Radius.^2) ;
   OutImage(Ipix) = nanmedian(Image(IndF));
end


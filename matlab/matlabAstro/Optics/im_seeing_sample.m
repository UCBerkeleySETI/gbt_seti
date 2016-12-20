function [PixImS,FWHMx,FWHMy,CutX,CutY]=im_seeing_sample(Im,varargin)
%--------------------------------------------------------------------------
% im_seeing_sample function                                         Optics
% Description: Given a 2D image (e.g., from a ray tracing simulation),
%              convolve the image with the seeing disk, and resample the
%              image (pixelize).
% Input  : - Image to convolve with seeing and to pixelize.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'SigmaX' - Sigma of seeing disk in X direction (FWHM/2.35) in
%                       original pixels units.
%                       Default is 6.5.
%            'SigmaY' - Sigma of seeing disk in X direction (FWHM/2.35).
%                       Default is SigmaX.
%            'Rho'    - Correlation between X and Y. Default is 0.
%            'Scale'  - Scale of pixelized image relative to the original
%                       image. Degfault is 1./6.5.
% Output : - The convolved and pixelized image.
%          - FWHM of brightest source in x direction [pixels].
%          - FWHM of brightest source in y direction [pixels].
%          - Cut of PSF through X axis.
%          - Cut of PSF through Y axis.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    May 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [PixImS,FWHMx,FWHMy]=im_seeing_sample(FFT_PSF_r2);
% Reliable: 2
%--------------------------------------------------------------------------


DefV.SigmaX = 6.5;
DefV.SigmaY = [];
DefV.Rho    = 0;
DefV.Scale  = 1./6.5;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (isempty(InPar.SigmaY)),
    InPar.SigmaY = InPar.SigmaX;
end


MaxSigma = max(InPar.SigmaX,InPar.SigmaY);
% convolve image with seeing disk
ImS = conv2_gauss(Im,'gauss',6.*MaxSigma,[InPar.SigmaX, InPar.SigmaY, InPar.Rho, 1]);
% sample image (pixelize)
PixImS = imresize(ImS,InPar.Scale,'box');
[~,MaxInd] = maxnd(PixImS);
CutX  = PixImS(MaxInd(1),:).';
CutY  = PixImS(:,MaxInd(2));
FWHMx = get_fwhm([],CutX);
FWHMy = get_fwhm([],CutY);






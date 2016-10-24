function [Lines]=prep_stripe82_deep(RA,Dec,Width,Filters,JD_Range,Seeing_Range,Bck_Range,Flux20_Range,OutputName);
%--------------------------------------------------------------------------
% prep_stripe82_deep function                                         sdss
% Description: Download all sdss stripe 82 images within a given box,
%              a given JD range and seeing, and combine the images using
%              swarp. The ouput images are named
%              coadd.fits and coadd.weight.fits.
% Input  : - J2000.0 R.A. [rad or sexagesimal string or H M S].
%          - J2000.0 Dec. [rad or sexagesimal string or Sign D M S].
%          - Output image width [arcmin].
%          - Filters to retrieve, default is 'r'.
%          - Two element vector containing JD range for images to download
%            [Min Max], default is [-Inf Inf].
%            min(JD) = 2453243; max(JD)=2454425
%          - Two element vector containing seeing (HWHM) range for images
%            to download [Min Max], default is [-Inf Inf].
%            95% of the images has HWHM~<1"
%          - Two element vector containing background level range for
%            images to download [Min Max], default is [-Inf Inf].
%          - Two element vector containing Flux20 range for
%            images to download [Min Max], default is [-Inf Inf].
%            For example, good images have flux20>1700.
%          - Optional output file name root.
%            If empty or not given then use default Swarp names:
%            'coadd.fits' & 'coadd.weight.fits'.
%            If given the use it as the file name and add the extensions
%            '.fits' and '.weight.fits', respectively.
% Output : - All the lines in the SDSS82_Fields.mat file found within the
%            search box.
% Tested : Matlab 7.3
%     By : Eran O. Ofek                    August 2008
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [Lines]=prep_stripe82_deep([01 00 00],0,10,'r',[-Inf Inf],[0 1],[-Inf Inf],[1700 Inf]);
% Reliable: 2
%--------------------------------------------------------------------------
PixelScale = 0.25;
ImageSize  = ceil(Width.*60./PixelScale);
MP         = 'n';       % multi-processor

Def.Coadd  = 'coadd.fits';
Def.CoaddW = 'coadd.weight.fits';

if (nargin==9),
   % do nothing
else
   OutputName = [];
end


SexRA      = convertdms(RA,'gH','SH');
SexDec     = convertdms(Dec,'gD','SD');

[Lines,Ind,FitsName]=get_all_sdss82(RA,Dec,Width,Filters,JD_Range,Seeing_Range,Bck_Range,Flux20_Range);

% Combine images using swarp
[OutImage,OutWImage] = run_swarp(FitsName, [], MP, 'CENTER_TYPE','MANUAL',...
	                                       'PIXELSCALE_TYPE','MANUAL',...
	                                       'PIXEL_SCALE',sprintf('%5.3f',PixelScale),...
	                                       'RESAMPLING_TYPE','LANCZOS4',...
	                                       'OVERSAMPLING','3,3',...
	                                       'COMBINE_TYPE','WEIGHTED',...
	                                       'IMAGE_SIZE',sprintf('%d,%d',ImageSize,ImageSize),...
	                                       'CENTER',sprintf('%s,%s',SexRA,SexDec));



if (isempty(OutputName)==0),
   % rename output and delete all fpC* images

   movefile(Def.Coadd,sprintf('%s.fits',OutputName));
   movefile(Def.CoaddW,sprintf('%s.weight.fits',OutputName));

   delete('fpC*.fit');
end

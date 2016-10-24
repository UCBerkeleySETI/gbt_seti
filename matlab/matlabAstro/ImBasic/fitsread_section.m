function Im=fitsread_section(ImageName,StartPix,EndPix)
%--------------------------------------------------------------------------
% fitsread_section function                                        ImBasic
% Description: Read a rectangular region of interest from a single
%              FITS image.
% Input  : - String containing FITS image name.
%          - Start pixels position [x, y].
%            Alternatively, this can be [xmin, xmax, ymin, ymax].
%            In this case the third input argument should not provided.
%          - End pixels position [x, y].
% Output : - FITS image sub section.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jun 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Im=fitsread_section('Image.fits',[11 11],[70 70]);
%          Im=fitsread_section('Image.fits',[11 70 11 70]);
% Reliable: 2
%--------------------------------------------------------------------------

if (nargin==2),
    EndPix   = StartPix([2,4]);
    StartPix = StartPix([1,3]);
end
    
import matlab.io.*
Fptr = fits.openFile(ImageName);
Im = fits.readImg(Fptr,StartPix,EndPix);
fits.closeFile(Fptr);



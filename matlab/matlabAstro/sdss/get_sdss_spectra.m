function [Links,FitsName,Spec]=get_sdss_spectra(ID,Save,DataRelease);
%-----------------------------------------------------------------------------
% get_sdss_spectra function                                              sdss
% Description: Given an SDSS soectra ID [Plate, MJD, Fiber], get the
%              link to, and the SDSS 1d spectra in FITS format.
%              Furthermore, read the spectra into matlab matrix.
%              The program first check if the FITS image is exist in
%              current directory and if so, it reads the image from the disk.
%              Note that if nargout>1 then the fits file is retrieved.
% Input  : - Matrix of spectra ID [Plate, MJD, Fiber].
%          - Save FITS images to disk {'y' | 'n'}, default is 'y'.
%          - Data release {'DRsup'|'DR5'}, default is 'DR7'.
% Output : - Cell array of links to each fits image.
%            Rows for each field, columns for each band.
%          - Cell array of fits images name.
%          - Cell array containing the structure of spectra.
%            Each cell containing the following fields:
%            .Wave        - Wavelength [Ang].
%            .Pix         - pixel index
%            .Flux        - Flux [erg cm^-2 s^-1 A^-1]
%            .FluxContSub - Continuum subtracted flux [erg cm^-2 s^-1 A^-1]
%            .Error       - Error in flux [erg cm^-2 s^-1 A^-1]
%            .Mask        - Mask containing flags (e.g., 0 if OK).
% Reference: http://www.sdss.org/dr6/dm/flatFiles/spSpec.html
%            http://www.sdss.org/dr6/products/spectra/read_spSpec.html
% See also : read_sdss_spec.m
% Tested : Matlab 7.0
%     By : Eran O. Ofek          June 2005
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [Link,FN,Spec]=get_sdss_spectra([266 51630 003],'y');
% Reliable: 1
%-----------------------------------------------------------------------------
RAD = 180./pi;

SDSS_Server = 'http://das.sdss.org/';

DefSave        = 'y';
DefDataRelease = 'DR7';
if (nargin==1),
   Save         = DefSave;
   DataRelease  = DefDataRelease;
elseif (nargin==2),
   DataRelease  = DefDataRelease;
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end


Nim    = size(ID,1);

Links  = cell(Nim,1);
for Iim=1:1:Nim,
   Plate     = ID(Iim,1);
   MJD       = ID(Iim,2);
   Fiber     = ID(Iim,3);

   %URL       = sprintf('%s%s/data/spectro/1d_25/%04d/1d/',SDSS_Server,DataRelease,Plate);
   URL       = sprintf('%s/spectro/1d_26/%04d/1d/',SDSS_Server,Plate);
   
   ImageName = sprintf('spSpec-%05d-%04d-%03d.fit',MJD,Plate,Fiber);
   Links{Iim}= sprintf('%s%s',URL,ImageName);
 
   FitsName{Iim}= ImageName;

   %--- Check if file is already on disk ---
   %FID1 = fopen(ImageName,'r');

   if (exist(ImageName,'file')==0),
      % file is not on disk - get file
      system(sprintf('wget -q %s',Links{Iim}));
   end
   %fclose(FID1);

   if (nargout>2),
      SpecRead = read_sdss_spec(FitsName{Iim},'old');
      Spec(Iim) = SpecRead(1);
   end
   switch Save
    case 'y'
       % do nothing
    case 'n'
       % delete file
       delete(FitsName{Iim});
    otherwise
       error('Unknown Save option');
   end
end



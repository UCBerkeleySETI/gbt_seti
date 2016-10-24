function Info=mag_calib_tsfield(Images,CatCol);
%------------------------------------------------------------------------------
% mag_calib_tsfield function                                              sdss
% Description: Use (or get and use) the SDSS tsField fits file to calculate
%              the photometric zero point for a set of an SDSS FITS
%              images.
% Input  : - Matrix of images ID [Run, ReRun, CamCol, Field].
%            If ReRun is NaN, then the program will look
%            for the most recent rerun in the SDSS archive directory.
%            Alternatively, this can be a cell array of images returned
%            by get_sdss_tsfield.m
%          - Cell array of all columns in FITS table.
%            If first input is matrix of images ID, then this is
%            a string indicating if to save the tsField fits file
%            {'y' | 'n'}, default is 'y'. indicating if to
% Output : - Vector in which each element (per image) is a structure
%            containing the following photometric information.
%              Info(Iim).AA       - photometric zp coef.
%              Info(Iim).KK       - extinction coef
%              Info(Iim).AirMass  - Airmass
%              Info(Iim).Counts20 - Counts for 20mag (AB) source
%              Info(Iim).Counts0  - Counts for 0mag (AB) source
%              Info(Iim).Sky      - Sky counts per sq. arcsec.
%              Info(Iim).MJD      - MJD
%            Note that all these fields are five element vectors
%            (per filter).
% Algorithm: http://home.fnal.gov/~neilsen/documents/sdss_data/
% Tested : Matlab 7.10
%     By : Eran O. Ofek                     July 2010
%    URL: http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: Info=mag_calib_tsfield([1339 40 1 11]);
% Reliable: 2
%------------------------------------------------------------------------------


ExpTime  = 53.907456;   % Exposure time for SDSS
Filters  = {'u','g','r','i','z'};
Filter_b = [0.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10];

Save = 'y';
if (iscell(Images)),
   % Images is already in cell format
else
   [Link,FN,Images,Header,CatCol1]=get_sdss_tsfield([1339 40 1 11],'y');
   if (nargin==1),
      Save = 'y';
   else
      Save = CatCol;
   end
   CatCol = CatCol1;
end

Nim = length(Images);

for Iim=1:1:Nim,
   % for each image
   AA       = Images{Iim}{find(strcmpi(CatCol,'aa')==1)};
   KK       = Images{Iim}{find(strcmpi(CatCol,'kk')==1)};
   AirMass  = Images{Iim}{find(strcmpi(CatCol,'airmass')==1)};
   Sky      = Images{Iim}{find(strcmpi(CatCol,'sky_frames_sub')==1)};
   MJD      = Images{Iim}{find(strcmpi(CatCol,'mjd')==1)};
   
   M20      = 20;
   FF0      = 2.*Filter_b.*sinh( (-log(10).*M20/2.5)-log(Filter_b) );
   Counts20 = ExpTime.*FF0./(10.^(0.4.*(AA+KK.*AirMass)));
   
   % Get conversion from maggies to counts
   M0 = 0;
   FF0= 2.*Filter_b.*sinh( (-log(10).*M0/2.5)-log(Filter_b) );
   RefCounts = ExpTime.*FF0./(10.^(0.4.*(AA+KK.*AirMass)));
   
   PixScale  = 0.396;   % ["/pix]
   SkyCounts = PixScale.^2.*RefCounts.*Sky;   % [counts/s]
   
   Info(Iim).AA       = AA;
   Info(Iim).KK       = KK;
   Info(Iim).AirMass  = AirMass;
   Info(Iim).Counts20 = Counts20;
   Info(Iim).Counts0  = RefCounts;
   Info(Iim).Sky      = SkyCounts;
   Info(Iim).MJD      = MJD;

   switch lower(Save)
    case 'n'
       delete(FN{Iim});
    otherwise
       % do nothing
   end
end

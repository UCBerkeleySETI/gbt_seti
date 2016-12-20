function [Lines,Ind]=find_all_sdss82(RA,Dec,Width)
%---------------------------------------------------------------------------
% find_all_sdss82 function                                             sdss
% Description: Look for all SDSS strip 82 images within a given box
%              region. This script search the SDSS82_Fields.mat file
% Input  : - J2000.0 R.A., in [rad], sexagesimal string or [H M S] format.
%          - J2000.0 Dec., in [rad], sexagesimal string or
%            [sign D M S] format.
%          - Search box width [arcmin].
% Output : - All the lines in the SDSS82_Fields.mat file found within the
%            search box.
%          - Indices in the lines of SDSS82_Fields.mat file found within the
%            search box.
% Tested : Matlab 7.6
%     By : Eran O. Ofek                   January 2007
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%---------------------------------------------------------------------------
RAD = 180./pi;

% SDSS unique field corners:
Corners = [1      65;   1    1425;   2048 1425;   2048   65];

RA  = convertdms(RA,'gH','r');
Dec = convertdms(Dec,'gD','R');

SearchRad = (Width+20)./(60.*RAD);   % initial search radius [rad]
MinDec    = Dec - 0.5.*Width./(60.*RAD);
MaxDec    = Dec + 0.5.*Width./(60.*RAD);
MinRA_l   = RA - 0.5.*Width./(60.*RAD)./cos(MinDec);
MinRA_u   = RA - 0.5.*Width./(60.*RAD)./cos(MaxDec);
MaxRA_l   = RA + 0.5.*Width./(60.*RAD)./cos(MinDec);
MaxRA_u   = RA + 0.5.*Width./(60.*RAD)./cos(MaxDec);
SearchCor = [MinRA_l MinDec; MinRA_u MaxDec; MaxRA_u MaxDec; MaxRA_l MinDec];



load SDSS82_Fields.mat
Col.N       = 1;
Col.Run     = 2;
Col.ReRun   = 3;
Col.CamCol  = 4;
Col.Field   = 5;
Col.JD      = 6;
Col.HWHM    = 7;
Col.Sky     = 8;
Col.Flux20  = 9;
Col.CRPIX1  = 10;
Col.CRPIX2  = 11;
Col.CRVAL1  = 12;
Col.CRVAL2  = 13;
Col.CD1_1   = 14;
Col.CD1_2   = 15;
Col.CD2_1   = 16;
Col.CD2_2   = 17;


Dist = sphere_dist(RA,Dec,SDSS82_Fields(:,Col.CRVAL1)./RAD,SDSS82_Fields(:,Col.CRVAL2)./RAD);
Is   = find(Dist<SearchRad);    % indices of nearby images to search
Nim  = length(Is);              % number of mearby images
K    = 0;
Ind  = [];
for Iim=1:1:Nim,
   WCS.CRPIX1   = SDSS82_Fields(Is(Iim),Col.CRPIX1);
   WCS.CRPIX2   = SDSS82_Fields(Is(Iim),Col.CRPIX2);
   WCS.CRVAL1   = SDSS82_Fields(Is(Iim),Col.CRVAL1);
   WCS.CRVAL2   = SDSS82_Fields(Is(Iim),Col.CRVAL2);
   WCS.CD1_1    = SDSS82_Fields(Is(Iim),Col.CD1_1);
   WCS.CD1_2    = SDSS82_Fields(Is(Iim),Col.CD1_2);
   WCS.CD2_1    = SDSS82_Fields(Is(Iim),Col.CD2_1);
   WCS.CD2_2    = SDSS82_Fields(Is(Iim),Col.CD2_2);

   CorCoo = wcs2corners(WCS,Corners);

   [IntPolyX,IntPolyY] = polyxpoly(SearchCor(:,1),SearchCor(:,2),...
			           CorCoo(:,1),CorCoo(:,2),'unique');
   if (isempty(IntPolyX)==0),
      % intersection found
      K = K + 1;
      Ind(K) = Is(Iim);
   end
end


if (isempty(Ind)==1),
   Lines = [];
else
   Ind = Ind.';
   Lines = SDSS82_Fields(Ind,:);
end

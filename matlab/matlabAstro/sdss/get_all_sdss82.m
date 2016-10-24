function [Lines,Ind,FitsName]=get_all_sdss82(RA,Dec,Width,Filters,JD_Range,Seeing_Range,Bck_Range,Flux20_Range);
%------------------------------------------------------------------------------
% get_all_sdss82 function                                                 sdss
% Description: Download all SDSS strip 82 images within a given box
%              region, in a given JD range and with a given seeing.
% Input  : - J2000.0 R.A., in [rad], sexagesimal string or [H M S] format.
%          - J2000.0 Dec., in [rad], sexagesimal string or
%            [sign D M S] format.
%          - Search box width [arcmin].
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
% Output : - All the lines in the SDSS82_Fields.mat file found within the
%            search box.
%          - Indices in the lines of SDSS82_Fields.mat file found within the
%            search box.
%          - Cell array containing list of all downloaded images.
% Tested : Matlab 7.6
%     By : Eran O. Ofek                       Jan 2006
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 1
%------------------------------------------------------------------------------
RAD = 180./pi;

DataRelease      = 'DRSN';

Def.Filters      = 'r';
Def.JD_Range     = [-Inf Inf];
Def.Seeing_Range = [-Inf Inf];
Def.Bck_Range    = [-Inf Inf];
Def.Flux20_Range = [-Inf Inf];
if (nargin==3),
   Filters       = Def.Filters;
   JD_Range      = Def.JD_Range;
   Seeing_Range  = Def.Seeing_Range;
   Bck_Range     = Def.Bck_Range;
   Flux20_Range  = Def.Flux20_Range;
elseif (nargin==4),
   JD_Range      = Def.JD_Range;
   Seeing_Range  = Def.Seeing_Range;
   Bck_Range     = Def.Bck_Range;
   Flux20_Range  = Def.Flux20_Range;
elseif (nargin==5),
   Seeing_Range  = Def.Seeing_Range;
   Bck_Range     = Def.Bck_Range;
   Flux20_Range  = Def.Flux20_Range;
elseif (nargin==6),
   Bck_Range     = Def.Bck_Range;
   Flux20_Range  = Def.Flux20_Range;
elseif (nargin==7),
   Flux20_Range  = Def.Flux20_Range;
elseif (nargin==8),
   % do nothing
else
   error('Illegal number of input arguments');
end

Col.Run     = 2;
Col.ReRun   = 3;
Col.CamCol  = 4;
Col.Field   = 5;
Col.JD      = 6;
Col.HWHM    = 7;
Col.Sky     = 8;
Col.Flux20  = 9;

% Find fields
[Lines,Ind]=find_all_sdss82(RA,Dec,Width);

% selection criteria
I = find(Lines(:,Col.JD)>=JD_Range(1) & Lines(:,Col.JD)<=JD_Range(2) & ...
         Lines(:,Col.HWHM)>=Seeing_Range(1) & Lines(:,Col.HWHM)<=Seeing_Range(2) & ...
         Lines(:,Col.Sky)>=Bck_Range(1) & Lines(:,Col.Sky)<=Bck_Range(2) & ...
         Lines(:,Col.Flux20)>=Flux20_Range(1) & Lines(:,Col.Flux20)<=Flux20_Range(2));

Lines = Lines(I,:);
Ind   = Ind(I,:);

% get images
[Links,FitsName]=get_sdss_corrim(Lines(:,[Col.Run, Col.ReRun, Col.CamCol, Col.Field]),'y',Filters,DataRelease);

Links
Links{1}

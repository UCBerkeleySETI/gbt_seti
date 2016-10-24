function [AperPhot,InCoo]=ds9_phot(Coo,varargin);
%------------------------------------------------------------------------------
% ds9_phot function                                                        ds9
% Description: Interactive photometry for ds9 display. Allow the user to
%              mark objects on the ds9 display, and perform centeroiding,
%              and simple aperture photometry.
% Input  : - List of coordinates to measure [X,Y].
%            If empty (i.e., []), then interactive mode.
%          * Arbitrary number of pair of arguments: ...,keyword,value,...
%            --- Zero point paarmeters ---
%            'ACS'       - HST/ACS FITS image name. If specified,
%                          then use the values of the
%                          gain, readout noise and zero point,
%                          appear in the HST/ACS image header,
%                          and use the AB magnitude zero point.
%                          Default is [].
%            'AperCorr'  - {'y'|'n'} apply aperture correction to HST/ACS image
%                          using the Sirianni et al. (2005) aperture
%                          corrections. Default is 'y'.
%                          The image scale is read from the header.
%            'SDSS'      - *** NOT WORKING ***
%                          SDSS FITS image name. If specified,
%                          then use the appropriate zero point
%                          from the SDSS image header,
%                          and use the AB magnitude zero point.
%                          Default is [].
%            --- Aperture photometry parameters --- see aperphot.m
%            'MaxIter'    - Maximum number of centering iterations,
%                           (0 for no centering) default is 1.
%            'CenBox'     - Width of centering box, default is 7.
%            'Aper'       - Radius of aperture in [pix] (scalar or vector),
%                           default is 5.
%            'ZP'         - Photometry zero point [mag], default is 22.
%            'Gain'       - gain, default is 1.
%            'ReadNoise'  - Readout noise [e], default is 6.5.
%            'Annu'       - Radius of sky annulus [pix],
%                           default is 12.
%            'DAnnu'      - Width of sky annulus [pix],
%                           default is 10.
%            'SkyAlgo'    - Sky subtraction algorithm:
%                           'Mean'     - mean
%                           'Median'   - median, default.
%                           'PlaneFit' - plane fit.
%            'SkySigClip' - Sigma clipping for sky rejuction,
%                           default is [3.5 3.5] sigmas,
%                           for lower and upper sigma clip.
%                           If NaN, then no sigma clipping.
%            'Print'      - print results to screen: {'y' | 'n'},
%                           default is 'y'.
%
%
% Output : - Structure containing the following fields:
%            Mag      : Magnitude for all stars (row for stars, column
%                       for aperture).
%            MagErr   : Magnitude Err for all stars (row for stars,
%                       column for aperture).
%            Flux     : Flux for all stars (row for stars, column for
%                       aperture).
%            FluxErr  : Flux Err for all stars (row for stars, column
%                       for aperture).
%            Area     : Aperure area for all stars (row for stars,
%                       column for aperture).
%            Info     : Object information:
%                       [M1(1),M1(2),M2(1,1),M2(2,2),M2(1,2),Sky,SkySTD,Nsky];
%          - Input coordinates [X,Y] (if interactive, then contains the
%            selected coordinates ('image' coordinates).
%
% Tested : Matlab 7.0
%     By : Eran O. Ofek                       Feb 2007
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [AperPhot,InCoo]=ds9_phot;   % interactive mode
% Reliable: 2
%------------------------------------------------------------------------------
CooType   = 'image';
Method    = 'center';

if (nargin==0),
   Coo    = [];
end

if (isempty(Coo)==1),
   % Interactive mode - select coordinates
   [CooX,CooY,Value] = ds9_getcoo(1,CooType);
   Coo = [CooX, CooY];
end


%--- Default values ---
ACS       = [];
SDSS      = [];
AperCorr  = 'y';
% default values from aperphot.m
% use to estaimate databox size
Aper      = 5;
Annu      = 12;
DAnnu     = 10;

Narg = length(varargin);
for Iarg=1:2:Narg-1,
   switch varargin{Iarg}
    case 'ACS'
       ACS      = varargin{Iarg+1};
    case 'SDSS'
       SDSS     = varargin{Iarg+1};
    case 'AperCorr'
       AperCorr = varargin{Iarg+1};
    case 'Aper'
       Aper     = varargin{Iarg+1};
    case 'Annu'
       Annu     = varargin{Iarg+1};
    case 'DAnnu'
       DAnnu    = varargin{Iarg+1};
    otherwise
       % do nothing - not for errornous keywords
   end
end


%--- Set Box size ---
InCoo      = Coo;
Ncoo       = size(InCoo,1);
SemiWidthX = max([Aper;Annu+DAnnu+1]);
SemiWidthY = max([Aper;Annu+DAnnu+1]);
Coo = [InCoo, SemiWidthX.*ones(Ncoo,1), SemiWidthY.*ones(Ncoo,1)];


%---------------------------
%--- Special zero points ---
%---------------------------
if (isempty(ACS)==0),
   ImageName = ACS;
   %---------------
   Key.Detector  = 'DETECTOR';
   Key.Filter1   = 'FILTER1';
   Key.Filter2   = 'FILTER2';
   Key.Units     = 'D001OUUN';    % should be 'cps'
   Key.CD1_1     = 'CD1_1';
   Key.CD1_2     = 'CD1_2';
   Key.PhotFLam  = 'PHOTFLAM';
   Key.PhotZPT   = 'PHOTZPT';
   Key.PhotPLam  = 'PHOTPLAM';
   Key.PhotBW    = 'PHOTBW';

   Keywords = get_fits_keyword(ImageName,{Key.Filter1; Key.Filter2; Key.CD1_1; Key.CD1_2; Key.PhotFLam; Key.PhotZPT; Key.PhotPLam});
   Filter    = Keywords{2}; % Keywords{1};
   CD1_1     = Keywords{3};
   CD1_2     = Keywords{4};
   PhotFLam  = Keywords{5};
   PhotZPT   = Keywords{6};
   PhotPLam  = Keywords{7};

   Scale = sqrt(CD1_1.^2+CD1_2.^2);

   % ST mag zero-point
   %ZP = -2.5.*log10(PhotFLam) - PhotZPT;
   % AB mag zero-point
   ZP = -2.5.*log10(PhotFLam) - 21.10 - 5.*log10(PhotPLam) + 18.6921;

   switch AperCorr
    case 'y'
       % HST/ACS Aperture correction
       [Energy,EnergyErr]=hst_acs_zp_apcorr(Aper.*Scale,Filter);
       ZP = ZP + 2.5.*log10(Energy);
    case 'n'
       % do nothing
    otherwise
       error('Unknown AperCorr option');
   end

   % set new zero point
   varargin{end+1} = 'ZP';
   varargin{end+1} = ZP;
elseif (isempty(SDSS)==0),
   ImageName = SDSS;
   %---------------
   error('SDSS option is not available');
else
   % do nothing - no ZP
end


for Icoo=1:1:Ncoo,
   %--- Get pixels value around chose position ---
   [MatVal,MatX,MatY] = ds9_getbox(Coo(Icoo,:),Method,CooType);

   %--- Aperture photometry ---
   [AllMag,AllMagErr,AllFlux,AllFluxErr,AllArea,ObjInfo]=aperphot(MatVal,[SemiWidthX+1, SemiWidthY+1],...
                            'VerifyKey','n',...
			    'OffsetX',InCoo(Icoo,1)-Coo(Icoo,3)-1,...
                            'OffsetY',InCoo(Icoo,2)-Coo(Icoo,4)-1,...
                            varargin{:});


   %--- Store APer. phot. result into AperPhot structure ---
   AperPhot.Mag(Icoo)      = AllMag;
   AperPhot.MagErr(Icoo)   = AllMagErr;
   AperPhot.Flux(Icoo)     = AllFlux;
   AperPhot.FluxErr(Icoo)  = AllFluxErr;
   AperPhot.Area(Icoo)     = AllArea;
   AperPhot.Info{Icoo}     = ObjInfo;
end

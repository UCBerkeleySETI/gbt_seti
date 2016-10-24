function [Cat,ThreshIm]=mextractor(Sim,varargin)
%--------------------------------------------------------------------------
% mextractor function                                               ImPhot
% Description: Source extractor written in MATLAB.
%              Given an input images, and matched filter, this function
%              filter the images by their matched filter and search for
%              sources above a detection threshold. For each source the
%              program measure its basic properties.
%              The main difference between this program and SExtractor
%              is that the noise ("sigma") is measured in the filtered
%              image, rather than the unfiltered image.
% Input  : - SIM/FITS images. See images2sim.m for details regarding
%            possible input.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'ColCell' - A list of columns to returm for each source.
%                 possible columns are:
%                 'NUMBER' - serial index of sources.
%                 'XPEAK_IMAGE'- Image X position of source brightest pixel.
%                 'YPEAK_IMAGE'- Image Y position of source brightest pixel.
%                 'FLUX_PEAK'  - Background subtracted peak flux of source.
%                 'FLUXTOT_PEAK'- Non-Background subtracted peak flux of source.
%                 'SN' - S/N of source.
%                 'SN_NF' - S/N of source in the unfiltered image
%                 'BACKGROUND'- Background level at source position.
%                 'SKY_STD'-StD at source position.
%                 'NEAREST_SRCIND'-Number of nearest source.
%                 'NEAREST_SRCDIST'-Distance to nearest source (pixels).
%                 'MAG_PEAK'- Background subtracted peak magnitude.
%                 'FLUX_PSFC'- PSF flux estimated from the cross
%                           correkation. This is reliable if the matched
%                           filter is accurate and the image is not
%                           confused.
%                 'MAG_PSFC'- PSF magnitude estimated from the cross
%                           correkation.
%                 'MAGERR_PSFC'- Error in PSF magnitude estimated from
%                           the cross-correlation S/N.
%                 'ALPHA_PEAK_J2000' - J2000.0 R.A. of peak flux position.
%                 'DELTA_PEAK_J2000' - J2000.0 Dec. of peak flux position.
%                 'ISFORCE' - A flag indicating if source was forced
%                             (i.e., came from a ForcePos list).
%                 'AZ'      - Azimuth of each source [in CUnits].
%                 'ALT'     - Altitude of each source [in CUnits].
%                 'AIRMASS' - Hardie air mass of each source.
%                 'PARANG'  - Parallactic angle of each source [in CUnits].
%                 Default is {'NUMBER','XPEAK_IMAGE','YPEAK_IMAGE','FLUX_PEAK','FLUXTOT_PEAK','SN',...
%                          'BACKGROUND','SKY_STD','NEAREST_SRCIND','NEAREST_SRCDIST',...
%                          'PEAK_MAG','PSFC_FLUX','PSFC_MAG','PSFC_MAGERR'}.
%            'CUnits' - Units for output coordinates {'rad','deg'}. 
%                   Default is 'deg'.
%            'Gain' - CCD gain header keyword or gain value.
%                   Default is 'GAIN'. If empty don't correct for gain.
%            'ZP' - Zero point for magnitudes. Default is 22.
%            'DetectThresh' - Detection threshold in units of sigma.
%                   Default is 5.
%            'ForcePos' - A list of position in which to force a position
%                   of sources and calculate the output parameters.
%                   This may be a a two column matrix [X, Y],
%                   or a cell array of such matrices (one per image).
%                   Default is empty.
%            'OnlyForce' - A flag indicating if to use sources only
%                   from the 'ForcePos list (true) or to use both
%                   the 'ForcePos' list and search for sources. 
%                   Default is false.
%            'MF' - A matrix of matched filter. Default is empty.
%                   Alternatively, this can be  a cell array of filters,
%                   one per image.
%            'MFpar' - If MF is not given, then use this cell array of
%                   parameters to pass to construct_matched_filter2d.m.
%                   Default is {'Gauss',[1.5 1.5 0 1],'B',0,'RN',0,...
%                               'FiltHalfSize',[5 5],'Rad0',3,'OnlyPSF',true}
%            'Noise' - Noise images for each images. See images2sim.m
%                   for input options. If empty then will calculate
%                   the noise using sim_std.m.
%                   Default is empty.
%            'SubBack' - Subtract background {true|false}. Deafult is true.
%            'FiltIm' - Filter image prior to thresholding and segmentation
%                   {true|false}. Default is true.
%            'MethodStD' - Method by which to estimate the image StD.
%                   See sim_std.m for options. Default is 'fit'.
%            'StdPar' - Cell array of key,val arguments to pass to
%                   sim_std.m. Default is {}.
%                   Note that if MethodStD is specified here it will
%                   override the MethodStD parameter.
%            'BackMethod' - Method by which to estimate the image background.
%                   See sim_background.m for options. Default is
%                   'mode_fit'.
%            'BackPar' - Cell array of key,val arguments to pass to
%                   sim_background.m. Default is {}.
%                   Note that if BackMethod is specified here it will
%                   override the BackMethod parameter.
%            'MinArea' - Min area of source above threshold in the filtered
%                   image. Default is 1. Note that values above 1 will 
%                   change the meaning of the detection threshold.
%            'AreaOpenConn' - connectivity for bwareaopen.m Default is 8.
%            'RegionMaxConn' - connectivity for imregionalmax. Default is 8.
%            'SortBy' - Column name by which to sort the output catalog.
%                    If empty do not sort. Default is 'YPEAK_IMAGE'.
%                    Note that catalog can be sorted only by columns
%                    that appears in 'ColCell'.
%            'MagInLuptitude' - Return all magnitudes in luptitude.
%                    Default is true.
%            'LuptSoft' - Luptitude softenning parameter. Default is 1e-10.
%            'HDUnum' - HDU number from which to read WCS. Default is 1.
%            'STtype' - Sidereal time type:
%                       'a' - apparent; 'm' - mean. Default is 'm'.
%            'Verbose' - Verbose. Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m, image2sim.m, sim_std.m, sim_background.m,
%            sim_julday.m, sim_geodpos.m
% Output : - Structure array of catalogs.
% See also: sextractor.m
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Apr 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Cat]=mextractor('*.fits');
% Reliable: 2
%--------------------------------------------------------------------------

RAD   = 180./pi;  % 1 radian = 57.295... deg
SN2DM = 2.5./log(10);  % S/N to mag error factor - 1.086...

% SIM class fields
ImageField      = 'Im';
HeaderField     = 'Header';
BackImField     = 'BackIm';
ErrImField      = 'ErrIm';
CatField        = 'Cat';
CatColField     = 'Col';
CatColCellField = 'ColCell';

% Default values of key,val input arguments
DefV.ColCell           = {'XPEAK_IMAGE','YPEAK_IMAGE','FLUX_PEAK','FLUXTOT_PEAK',...
                          'SN','SN_NF','ANNU_FLUX',...
                          'BACKGROUND','SKY_STD','NEAREST_SRCIND','NEAREST_SRCDIST',...
                          'MAG_PEAK','FLUX_PSFC','MAG_PSFC','MAGERR_PSFC','AZ','ALT','AIRMASS','PARANG','XWIN_IMAGE','YWIN_IMAGE'};
DefV.CUnits            = 'deg';  % Units for output coordinates {'rad','deg'}. 
DefV.Gain              = 'GAIN'; % Gain header keyword or gain value
DefV.ZP                = 22;     % electrons to magnitude photometric zero point
DefV.DetectThresh      = 5;
DefV.ForcePos          = [];
DefV.OnlyForce         = false;
DefV.MF                = [];
DefV.MFpar             = {'Gauss',[1.5 1.5 0 1],'B',0,'RN',0,'FiltHalfSize',[5 5],'Rad0',5,'OnlyPSF',true};
DefV.SubBack           = true;
DefV.Noise             = [];
DefV.MethodStD         = 'fit';
DefV.StdPar            = {};
DefV.BackMethod        = 'mode_fit';
DefV.BackPar           = {};
DefV.WidthB0           = 2;    % number of pixels around bounderies to set to zero
DefV.FiltIm            = true;
DefV.GlobalNoiseFilter = false;
DefV.MinArea           = 1;
DefV.AreaOpenConn      = 8;  % [4 | 8]
DefV.RegionMaxConn     = 8;  % [4 | 8]
DefV.SortBy            = 'YPEAK_IMAGE';  % must be present in ColCell - empty don't sort
DefV.MagInLuptitude    = true;
DefV.LuptSoft          = 1e-10;
DefV.HDUnum            = 1;
DefV.STtype            = 'm';   % sidereal time type m|a
DefV.AperRad           = 3;
DefV.ReplaceVal        = 0;
DefV.Verbose           = true;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (~iscell(InPar.ForcePos)),
    InPar.ForcePos = {InPar.ForcePos};
end
Nfplists = numel(InPar.ForcePos);

% Convert RA/Dec to rad/deg
ConvCoo = convert_units('rad',InPar.CUnits);   % conversion factor 2 output coordinates
Coo2rad = convert_units(InPar.CUnits,'rad');   % conversion factor from output coordinates to radians

% read images
Sim  = images2sim(Sim,varargin{:});
Nim  = numel(Sim);
Ncol = numel(InPar.ColCell);


% read images Gain and correct for Gain
% i.e., convert image units from ADU to electrons
if (isempty(InPar.Gain)),
    Gain = ones(Nim,1);
else
    [Gain,Sim] = sim_gain(Sim,'Gain',InPar.Gain);
end

% constrct/read match filter
if (isempty(InPar.MF)),
    MF = construct_matched_filter2d(InPar.MFpar{:},'Norm',1);
else
    MF = InPar.MF;
end

% construct aperture photometry filters
if (any(strcmp_cell(InPar.ColCell,{'FLUX_APER','FLUXERR_APER','MAG_APER','MAGERR_APER','XWIN_IMAGE','YWIN_IMAGE'}))),
    % construct sky annulus filters
    MF_PSF = construct_matched_filter2d(InPar.MFpar{:},'FiltHalfSize',InPar.AperRad,'Norm',[],'OnlyPSF',true, 'Rad0',InPar.AperRad);
    SizeMF = size(MF_PSF);
    VX = (1:1:SizeMF(2))' - (SizeMF(2)+1).*0.5;
    VY = (1:1:SizeMF(1))' - (SizeMF(1)+1).*0.5;
    [MF_X,MF_Y] = meshgrid(VX,VY);
    
    MF_EV  = construct_matched_filter2d(InPar.MFpar{:},'FiltHalfSize',InPar.AperRad,'Norm',[],'OnlyPSF',false,'Rad0',InPar.AperRad);
    
    MF_Aper = construct_matched_filter2d('MFtype','circ','FiltHalfSize',InPar.AperRad,'ParMF',InPar.AperRad,'Norm',[],'OnlyPSF',true);
    MF_Annu = construct_matched_filter2d('MFtype','annulus','FiltHalfSize',25,'ParMF',[15 25],'Norm',[],'OnlyPSF',true);
    MF_cx   = construct_matched_filter2d('MFtype','circx','FiltHalfSize',InPar.AperRad,'ParMF',InPar.AperRad,'Norm',[],'OnlyPSF',true);
    MF_cy   = construct_matched_filter2d('MFtype','circy','FiltHalfSize',InPar.AperRad,'ParMF',InPar.AperRad,'Norm',[],'OnlyPSF',true);
    MF_cx2  = construct_matched_filter2d('MFtype','circx2','FiltHalfSize',InPar.AperRad,'ParMF',InPar.AperRad,'Norm',[],'OnlyPSF',true);
    MF_cy2  = construct_matched_filter2d('MFtype','circy2','FiltHalfSize',InPar.AperRad,'ParMF',InPar.AperRad,'Norm',[],'OnlyPSF',true);
    MF_cxy  = construct_matched_filter2d('MFtype','circxy','FiltHalfSize',InPar.AperRad,'ParMF',InPar.AperRad,'Norm',[],'OnlyPSF',true);
    
    
end



% check if observatory coordinates and time is needed
if (any(strcmp_cell(InPar.ColCell,{'AZ','ALT','AIRMASS','PARANG'}))),
    % get time (UTC JD)
    [JD,ExpTime] = sim_julday(Sim,varargin{:});
    % get observatory geodetic position
    GeodPos      = sim_geodpos(Sim,varargin{:});
    LST          = lst(JD,[GeodPos.Long].',InPar.STtype);   % [fracday]
    %[Az,Alt,AM]=ha2az(HA,Dec,Lat);
end
    

if (~isempty(InPar.ReplaceVal)),
    Sim = sim_replace(Sim,'Range',[InPar.ReplaceVal InPar.ReplaceVal],'Value',NaN);
end

% estimate the background and std
SimB = sim_back_std(Sim,varargin{:});

% read noise from InPar.Noise (user provided)
if (~isempty(InPar.Noise)),
    SimNoise = images2sim(InPar.Noise,varargin{:});
    SimB.(ErrImField) = SimNoise.(ImageField);
end

% % background subtraction
% if (InPar.SubBack),
%     SimB = sim_background(Sim,'BackMoethod',InPar.BackMethod,InPar.BackPar{:},'SubBack',true,'StoreBack',true);
% else
%     SimB = Sim;
% end


if (InPar.WidthB0>0),
    % set pixels near boundery to zero in SimB
    for Iim=1:1:Nim,
        SimB(Iim).(ImageField)(1:InPar.WidthB0,:)         = 0;
        SimB(Iim).(ImageField)(end-InPar.WidthB0+1:end,:) = 0;
        SimB(Iim).(ImageField)(:,1:InPar.WidthB0)         = 0;
        SimB(Iim).(ImageField)(:,end-InPar.WidthB0+1:end) = 0;
    end
end
   


% filtering
if (InPar.FiltIm),
    SimF = sim_filter(SimB,'Filter',MF);
    %SimF.(ImageField) = fftshift(fftshift(ifft2(fft2(SimB.(ImageField)).*conj(fft2(MF))),1),2);
    
    %[B,S] = mode_fit(SimF.Im);
    %SimF.Im = SimF.Im - B;
else
    SimF = SimB;
end

% estimate image noise
if (InPar.GlobalNoiseFilter),
    % image noise estimated from the entire image - return a scalar
    SimF = sim_std(SimF,'MethodStD',InPar.MethodStD,InPar.StdPar{:});
else
    % image noise estimated per pixel
    SimF = sim_back_std(SimF,varargin{:});
    % can estimate the StD using: sqrt(sumnd(MF.^2).*Back)
end

% find sources in each image
Cat = struct_def({CatField,CatColCellField,CatColField},Nim,1);
% run over all images
for Iim=1:1:Nim,
    %----------------
    % locate sources
    %----------------
    Ifpl = min(Iim,Nfplists);
    if (~isempty(InPar.ForcePos{Ifpl})),
        % ForcePos list supplied by user - read it
        PeakXF = InPar.ForcePos{Ifpl}(:,1);
        PeakYF = InPar.ForcePos{Ifpl}(:,2);
        PeaksIndF = sub2ind(size(Sim(Iim).(ImageField)),PeakYF,PeakXF);
        IsForceF   = true(size(PeakXF));
    else
        % ForcePos list wasn't provided - set to empty
        PeakXF = [];
        PeakYF = [];
        PeaksIndF = [];
        IsForceF = [];
    end
    
    ThreshIm(Iim).(ImageField) = SimF(Iim).(ImageField)./SimF(Iim).(ErrImField);
    
    if (~InPar.OnlyForce),
        % locate new sources using filtering and thresholding
        
        % searching for pixels which are N-sigma above background
        DetSrc = SimF(Iim).(ImageField)>InPar.DetectThresh.*SimF(Iim).(ErrImField);
        
        % threshold relative to noise in un-filtered image
        %DetSrc = SimF(Iim).(ImageField)>InPar.DetectThresh.*SimB(Iim).(ErrImField);
        
        % remove detection with number of pixels smaller than MinArea
        % For example if MinArea is 2 DetSrc with 1 pixel will be removed,
        % but those with 2 pixels will be left as they are
        DetSrc = bwareaopen(DetSrc,InPar.MinArea,InPar.AreaOpenConn);

        % locate local maxima in the filtered image
        %--- KNOWN PROBLEML if two identical value next to each other than
        %--- imregionalmax will find both
        MaxImage = DetSrc.*SimF(Iim).(ImageField);
        if (~isempty(InPar.ReplaceVal)),
            MaxImage(isnan(MaxImage)) = InPar.ReplaceVal;
        end
        PeaksIm  = imregionalmax(MaxImage,InPar.RegionMaxConn);
            
        %PeaksInd = find(PeaksIm>0);
        PeaksInd = find(PeaksIm);
        % Read peaks coordinates
        [PeakY,PeakX]    = ind2sub(size(PeaksIm),PeaksInd);
        IsForce   = false(size(PeakXF));
    else
        % don't locate new source - use only ForcePos list
        PeakX = [];
        PeakY = [];
        PeaksInd = [];
        IsForce = [];
    end
    
    % merge forced sources and searched sources
    PeakX    = [PeakX;PeakXF];
    PeakY    = [PeakY;PeakYF];
    PeaksInd = [PeaksInd;PeaksIndF];
    IsForce  = [IsForce;IsForceF];
    Nsrc     = numel(PeaksInd);   % total number of sources found+force
    
    if (InPar.Verbose),
        fprintf('mextracting image number %d out of %d - %d sources found\n',Iim,Nim,Nsrc);
    end
    
    %SizeIm  = size(Sim(Iim).(ImageField));
    %SizeErr = size(SimF(Iim).(ErrImField));
     
    % populate catalog table of image
    Cat(Iim).(CatField) = zeros(Nsrc,Ncol);
    % write table columns to Cat
    Flag_CooPeak = true;
    Alt          = [];
    Az           = [];
    AirMass      = [];
    for Icol=1:1:Ncol,
        switch InPar.ColCell{Icol}
            case 'NUMBER'
                % source serial index
                Cat(Iim).(CatField)(:,Icol) = (1:1:Nsrc).';
            case 'ISFORCE'
                % flag indicating of the source was found (0) or was forced by
                % user (1)
                Cat(Iim).(CatField)(:,Icol) = double(IsForce);
            case 'XPEAK_IMAGE'
                % X position of source peak
                Cat(Iim).(CatField)(:,Icol) = PeakX;
            case 'YPEAK_IMAGE'
                % Y position of source peak
                Cat(Iim).(CatField)(:,Icol) = PeakY;
            case 'FLUX_PEAK'
                % background subtracted source count at peak position
                Cat(Iim).(CatField)(:,Icol) = SimB(Iim).(ImageField)(PeaksInd);
            case 'FLUXTOT_PEAK'
                % non-background subtracted source count at peak position
                Cat(Iim).(CatField)(:,Icol) = Sim(Iim).(ImageField)(PeaksInd);
            case 'FLUX_PSFC'
                % PSF flux estimated from the cross correlation with the
                % matched filter.
                % This is reliable if the matched filter is accurate and
                % the image is not confused.
                Cat(Iim).(CatField)(:,Icol) = SimF(Iim).(ImageField)(PeaksInd);
            case 'MAG_PSFC'
                % PSF magnitude estimated from the cross correlation with the
                % matched filter.
                Cat(Iim).(CatField)(:,Icol) = flux2mag(SimF(Iim).(ImageField)(PeaksInd), InPar.ZP, InPar.MagInLuptitude,InPar.LuptSoft);
                %Cat(Iim).(CatField)(:,Icol) = InPar.ZP - 2.5.*log10(SimF(Iim).(ImageField)(PeaksInd));
            case 'MAGERR_PSFC'
                % PSF magnitude error estimated from the cross correlation with the
                % matched filter.
                if (numel(SimF(Iim).(ErrImField))==1),
                    Cat(Iim).(CatField)(:,Icol) = SN2DM.*SimF(Iim).(ErrImField)./SimF(Iim).(ImageField)(PeaksInd);
                else
                    Cat(Iim).(CatField)(:,Icol) = SN2DM.*SimF(Iim).(ErrImField)(PeaksInd)./SimF(Iim).(ImageField)(PeaksInd);
                end
            case 'ANNU_FLUX'
                % flux in annulus around source in original image
                Tmp = sim_filter(Sim(Iim),'Filter',MF_Annu);
                Cat(Iim).(CatField)(:,Icol) = Tmp(Iim).(ImageField)(PeaksInd);
                
            case 'SN'
                % S/N as measured from the peak in the filtered image
                % normalized by the noise in the filtered image
                if (numel(SimF(Iim).(ErrImField))==1),
                    Cat(Iim).(CatField)(:,Icol) = SimF(Iim).(ImageField)(PeaksInd)./SimF(Iim).(ErrImField);
                else
                    Cat(Iim).(CatField)(:,Icol) = SimF(Iim).(ImageField)(PeaksInd)./SimF(Iim).(ErrImField)(PeaksInd);
                end
            case 'SN_NF'
                % S/N as measured from the peak in the un-filtered image
                % normalized by the noise in the un-filtered image
                if (numel(SimB(Iim).(ErrImField))==1),
                    Cat(Iim).(CatField)(:,Icol) = SimB(Iim).(ImageField)(PeaksInd)./SimB(Iim).(ErrImField);
                else
                    Cat(Iim).(CatField)(:,Icol) = SimB(Iim).(ImageField)(PeaksInd)./SimB(Iim).(ErrImField)(PeaksInd);
                end    
            %case 'FLUX_APER'
                % aperture photometry flux
                
            case 'BACKGROUND'
                % Background at peak position
                if (numel(SimB(Iim).(BackImField))==1),
                    Cat(Iim).(CatField)(:,Icol) = SimB(Iim).(BackImField);
                else
                    Cat(Iim).(CatField)(:,Icol) = SimB(Iim).(BackImField)(PeaksInd);
                end   
            case 'SKY_STD'
                % StD at peak position at the original image
                if (numel(SimB(Iim).(ErrImField))==1),
                    Cat(Iim).(CatField)(:,Icol) = SimB(Iim).(ErrImField);
                else
                    Cat(Iim).(CatField)(:,Icol) = SimB(Iim).(ErrImField)(PeaksInd);
                end   
            case 'NEAREST_SRCIND'
                % Index of nearest source
                % sort catalog...
                
                % search_cat(...'SearchMethod','binms1','CooType','plane')
            case 'NEAREST_SRCDIST'
                % Distance to nearest source   
                
            case 'MAG_PEAK'
                % Magnitude of peak
                Cat(Iim).(CatField)(:,Icol) = flux2mag(SimB(Iim).(ImageField)(PeaksInd), InPar.ZP, InPar.MagInLuptitude,InPar.LuptSoft);
                %Cat(Iim).(CatField)(:,Icol) = InPar.ZP - 2.5.*log10(SimB(Iim).(ImageField)(PeaksInd));
                
            case 'ALPHA_PEAK_J2000'
                % J2000.0 RA of peak
                if (Flag_CooPeak),
                    [ALPHA_PEAK_J2000,DELTA_PEAK_J2000]=xy2sky(Sim(Iim),PeakX,PeakY,InPar.HDUnum);
                    Flag_CooPeak = false;
                end
                Cat(Iim).(CatField)(:,Icol) = ALPHA_PEAK_J2000.*ConvCoo;
                
            case 'DELTA_PEAK_J2000'    
                % J2000.0 Dec of peak
                if (Flag_CooPeak),
                    [ALPHA_PEAK_J2000,DELTA_PEAK_J2000]=xy2sky(Sim(Iim),PeakX,PeakY,InPar.HDUnum);
                    Flag_CooPeak = false;
                end
                Cat(Iim).(CatField)(:,Icol) = DELTA_PEAK_J2000.*ConvCoo;
                
            case 'XWIN_IMAGE'
                % First moment X position
                %MF_PSF = MF_PSF./sum(MF_PSF(:));
                OverShot = 1.15;
                %SimMF  = sim_filter(SimB(Iim),'Filter',MF_PSF, 'FiltNorm',[],'Back',false,'MaskFilter',false);
                SimXW  = sim_filter(SimF(Iim),'Filter',MF_cx,  'FiltNorm',[],'Back',false,'MaskFilter',false);
                SimXWn = sim_filter(SimF(Iim),'Filter',MF_Aper,'FiltNorm',[],'Back',false,'MaskFilter',false);
                %SimXW  = sim_filter(SimB(Iim),'Filter',MF_cx,  'FiltNorm',[],'Back',false,'MaskFilter',false);
                %SimXWn = sim_filter(SimB(Iim),'Filter',MF_Aper,'FiltNorm',[],'Back',false,'MaskFilter',false);
                Cat(Iim).(CatField)(:,Icol) = PeakX + OverShot.*SimXW.(ImageField)(PeaksInd) ./ SimXWn.(ImageField)(PeaksInd);
                
            case 'YWIN_IMAGE'
                % First moment X position
                %MF_PSF = MF_PSF./sum(MF_PSF(:));
                OverShot = 1.15;
                %SimMF  = sim_filter(SimB(Iim),'Filter',MF_PSF, 'FiltNorm',[],'Back',false,'MaskFilter',false);
                SimYW  = sim_filter(SimF(Iim),'Filter',MF_cy,  'FiltNorm',[],'Back',false,'MaskFilter',false);
                SimYWn = sim_filter(SimF(Iim),'Filter',MF_Aper,'FiltNorm',[],'Back',false,'MaskFilter',false);
                %SimYW  = sim_filter(SimB(Iim),'Filter',MF_cy,  'FiltNorm',[],'Back',false,'MaskFilter',false);
                %SimYWn = sim_filter(SimB(Iim),'Filter',MF_Aper,'FiltNorm',[],'Back',false,'MaskFilter',false);
                Cat(Iim).(CatField)(:,Icol) = PeakY + OverShot.*SimYW.(ImageField)(PeaksInd) ./ SimYWn.(ImageField)(PeaksInd);
         
            case 'AZ'
                % calculate Azimuth of each source
                if (Flag_CooPeak),
                    [ALPHA_PEAK_J2000,DELTA_PEAK_J2000]=xy2sky(Sim(Iim),PeakX,PeakY,InPar.HDUnum);
                    Flag_CooPeak = false;
                end
                if (isempty(Az)),
                    [Az,Alt,AirMass]=ha2az(2.*pi.*LST(Iim)-ALPHA_PEAK_J2000, DELTA_PEAK_J2000, GeodPos(Iim).Lat);
                end
                Cat(Iim).(CatField)(:,Icol) = Az.*ConvCoo;
                
            case 'ALT'
                % calculate Altitude of each source
                if (Flag_CooPeak),
                    [ALPHA_PEAK_J2000,DELTA_PEAK_J2000]=xy2sky(Sim(Iim),PeakX,PeakY,InPar.HDUnum);
                    Flag_CooPeak = false;
                end
                if (isempty(Alt)),
                    [Az,Alt,AirMass]=ha2az(2.*pi.*LST(Iim)-ALPHA_PEAK_J2000, DELTA_PEAK_J2000, GeodPos(Iim).Lat);
                end
                Cat(Iim).(CatField)(:,Icol) = Alt.*ConvCoo;
                
            case 'AIRMASS'
                % calculate hardie airmass for each source
                if (Flag_CooPeak),
                    [ALPHA_PEAK_J2000,DELTA_PEAK_J2000]=xy2sky(Sim(Iim),PeakX,PeakY,InPar.HDUnum);
                    Flag_CooPeak = false;
                end
                if (isempty(AirMass)),
                    [Az,Alt,AirMass]=ha2az(2.*pi.*LST(Iim)-ALPHA_PEAK_J2000, DELTA_PEAK_J2000, GeodPos(Iim).Lat);
                end
                Cat(Iim).(CatField)(:,Icol) = AirMass;

            case 'PARANG'
                % calculate parallactic angle for each source
                if (Flag_CooPeak),
                    [ALPHA_PEAK_J2000,DELTA_PEAK_J2000]=xy2sky(Sim(Iim),PeakX,PeakY,InPar.HDUnum);
                    Flag_CooPeak = false;
                end
                ParAng = parallactic_angle(ALPHA_PEAK_J2000, DELTA_PEAK_J2000, LST(Iim), GeodPos(Iim).Lat);
                Cat(Iim).(CatField)(:,Icol) = ParAng.*ConvCoo;
                
            otherwise
                error('Unknown Column option: %s',InPar.ColCell{Icol});
        end
    end
    Cat(Iim).(CatColCellField) = InPar.ColCell;
    Cat(Iim).(CatColField)     = cell2struct(num2cell(1:1:Ncol),InPar.ColCell,2);
    
    
    % sort output
    if (~isempty(InPar.SortBy)),
        % sort by specifc column that exisit in the output table
        Cat(Iim).(CatField) = sortrows(Cat(Iim).(CatField),Cat(Iim).(CatColField).(InPar.SortBy));
    end
        
end


% 
% [SimSeg,NumSeg]=sim_segmentation(SimF,'SubBack',false,'Thresh',5,'ErrIm',SimStd.(ErrImField));
% OutPar = {'Iseg','X','Y','X2','Y2','XY','NPIX','X_PEAK','Y_PEAK','FLUX_PEAK','FLUX'};
% 
% Cat=sim_seg_moments(Sim,SimSeg,varargin{:},'OutPar',OutPar);
% Isrc = Cat(1).Cat(:,Cat(1).Col.NPIX)>=1;
% Cat(1).Cat = Cat(1).Cat(Isrc,:);
% Cat(1).Cat = sortrows(Cat(1).Cat,Cat(1).Col.Y);
% Res = search_cat(Cat(1).Cat,Cat(1).Cat(:,[Cat(1).Col.X, Cat(1).Col.Y]),[],...
%                  'ColX',Cat(1).Col.X,'ColY',Cat(1).Col.Y,'SearchMethod','binmdup','CooType','plane','SearchRad',3);
% Cat(1).Cat = Cat(1).Cat([Res.Nfound]==0,:);
% 
% NumSeg
% 
% 




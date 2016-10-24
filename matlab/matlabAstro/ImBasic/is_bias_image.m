function [IsBias,IsGoodNoise,IsGoodMean,IsBiasKey]=is_bias_image(Sim,varargin)
%--------------------------------------------------------------------------
% is_bias_image function                                           ImBasic
% Description: Given a list of FITS images or SIM, look for good
%              bias images. The search is done by looking for specific
%              keyword values in the image headers, and also by checking
%              the noise and mean properties of the images.
% Input  : - List of images to check for bias images.
%            The following inputs are possible:
%            (1) Cell array of image names in string format.
%            (2) String containing wild cards (see create_list.m for
%                option). E.g., 'lred00[15-28].fits' or 'lred001*.fits'.
%            (3) Structure array of images (SIM).
%                The image should be stored in the 'Im' field.
%                This may contains also mask image (in the 'Mask' field),
%                and an error image (in the 'ErrIm' field).
%            (4) A file contains a list of image (e.g., '@list').
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'ReadHead'- Read header information {true|false}.
%                       Default is true.
%            'ImType' - Image type. One of the following:
%                       'FITS'   - fits image (default).
%                       'imread' - Will use imread to read a file.
%                       'mat'    - A matlab file containing matrix,
%                                  or structure array.
%            'FitsPars'- Cell array of additonal parameters to pass to
%                        fitsread.m. Default is {}.
%            'ImreadPars' - Cell array of additonal parameters to pass to
%                        imread.m. Default is {}.
%            'FieldName' - Field name containing the header in the
%                        structure returned by fitsinfo.m.
%                        Default is 'PrimaryData'. If empty then use
%                        default. If NaN then will attempt to look for 
%                        the correct field.
%            'Ind'     - Index of image header, in header structure.
%                        Default is 1.
%            'BiasKeyList' - List of header keywords which may contain
%                        the image type.
%                        Default is {'object','imgtype','imtype','type','imagetyp'}.
%                        All these keywords will be searched.
%            'BiasValList' - The list of expected values of the image type
%                        of a bias image. Default is {'bias'}.
%            'ExpTimeKey' - If not empty, then will check that the image
%                        exposure time is zero. If not empty, then this
%                        need to be the exposure time image header keyword.
%                        Default is 'EXPTIME'.
%            'DateKey' - If not empty, then will check if candidate bias
%                        image is not isolated in time. If not empty,
%                        then this need to be the UTC date image
%                        header keyword (usually 'DATE' containing the
%                        UTC date; e.g., '2009-02-18T08:10:13').
%                        Default is [].
%                        This parameter is useful in cases in which the
%                        first bias image is noisy.
%            'TimeDiffIso' - If 'DateKey' is not empty, then this parameter
%                        defining the maximum allowed time between a
%                        bias image and the previous image.
%                        Default is 2 [min].
%            'CheckImage' - Check the statistics of the candidate bias
%                        images to see if it is consistent with
%                        expectation {true|false}. Default is true.
%            'RN'      - CCD readnoise [e-]. Default is 10.
%            'Gain'    - CCD gain. Default is 1.
%            'Nsigma'  - Number of sigma in which the mean bias image
%                        std is allowed to be above the readnoise.
%                        Default is 3.
%            'StdFun'  - An handle to the function for std estimation.
%                        (e.g., @nanstd). Default is @nanrstd.
%            'MaxBiasRatio' - Bias image uniformaity criterion.
%                        If the median of a bias image deviates by more
%                        than 1+/- this parameter from the median
%                        of all candidate bias images then its IsGoodMean
%                        flag is set to false.
%            'Verbose' - Print progress messages {true|false}.
%                        Default is false.
% Output : - A flag vector (IsBias) indicating if the image is a bias
%            image or not.
%          - A flag vector (IsGoodNoise) indicating if the candidate bias
%            image have a good noise properties (i.e., comparable with
%            read noise).
%          - A flag vector (IsGoodMean) indicating if the candidate bias
%            image have a good mean value relative to the mean bias image.
%          - A flag vector (IsBiasKey) indicating if the image have
%            a bias keyword.
%            A good bias images are: IsBias & IsGoodNoise & IsGoodMean.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [IsBias,IsGoodNoise,IsGoodMean]=is_bias_image('red*.fits');
% Reliable: 2
%--------------------------------------------------------------------------


MIN_IN_DAY  = 1440;

ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

% input parameters
% images2sim.m parameters
DefV.ReadHead   = true;
DefV.ImType     = 'FITS';
DefV.FitsPars   = {};
DefV.ImreadPars = {};
% read header
DefV.FieldName  = [];
DefV.Ind        = 1;
% bias identification
DefV.BiasKeyList   = {'object','imgtype','imtype','type','imagetyp'};
DefV.BiasValList   = {'bias'};
DefV.ExpTimeKey    = 'EXPTIME';  % if empty then don't check
DefV.DateKey       = [];         %'DATE';
DefV.TimeDiffIso   = 2;          % [min]
% check images
DefV.CheckImage    =  true;
DefV.RN            = 10;
DefV.Gain          = 1;
DefV.Nsigma        = 3;
DefV.StdFun        = @nanrstd;   
DefV.MaxBiasRatio  = 0.05;
DefV.Verbose       = false;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (isstruct(Sim)),
    % do nothing
    InputSim = true;
    Nim      = numel(Sim);
else
    [~,ImageListCell] = create_list(Sim,NaN);
    InputSim = false;
    Nim      = numel(ImageListCell);
end


% go over images
IsBias      = false(Nim,1);
IsBiasKey   = false(Nim,1);
IsExpTime0  = false(Nim,1);
IsGoodNoise = false(Nim,1);
IsGoodMean  = false(Nim,1);
Mean        = zeros(Nim,1).*NaN;
JD          = zeros(Nim,1).*NaN;
for Iim=1:1:Nim,
    % read image
    if (InputSim);
        Header = Sim(Iim).(HeaderField);
        if (InPar.CheckImage),
            Image = Sim(Iim).(ImageField);
        end
    else
        % read from FITS
        %Header = fits_header_cell(ImageListCell{Iim},InPar.FieldName,InPar.Ind);
        Header = fits_get_head(ImageListCell{Iim},InPar.Ind);
        if (InPar.CheckImage),
            Image = fitsread(ImageListCell{Iim},InPar.FitsPars{:});
        end
    end
    
    % check candidate image header
    NewCellHead = cell_fitshead_getkey(Header,InPar.BiasKeyList,NaN);
    Vals  = NewCellHead(:,2);
    Nvals = numel(Vals);
    for Ivals=1:1:Nvals,
        if (~isempty(find(strcmpi(spacedel(Vals{Ivals}),InPar.BiasValList), 1))),
            IsBias(Iim)    = true;
            IsBiasKey(Iim) = true;
        end
    end
    
    % check ExpTime
    if (~isempty(InPar.ExpTimeKey)),
        NewCellHead = cell_fitshead_getkey(Header,InPar.ExpTimeKey,NaN);
        IsExpTime0(Iim) = str2num_nan(NewCellHead{1,2})==0;
    end
    
    % check that the time is not isolated
    if (~isempty(InPar.DateKey)),
        NewCellHead = cell_fitshead_getkey(Header,InPar.DateKey,NaN);
        JD(Iim) = julday(NewCellHead{1,2});
    end            
    
    if (InPar.CheckImage),
        % check image noise
        %Ratio(Iim)       = nanstd(Image(:)).*InPar.Gain;
        IsGoodNoise(Iim) = InPar.StdFun(Image(:)).*InPar.Gain<(InPar.RN.*InPar.Nsigma);
        Mean(Iim)        = nanmedian(Image(:));
    end
end

% check that the time is not isolated
if (~isempty(InPar.DateKey)),
    Diff = [Inf; diff(JD)];
    IsNotIsolated = Diff<InPar.TimeDiffIso./MIN_IN_DAY;
    IsBias        = IsBias & IsNotIsolated;
end

if (~isempty(InPar.ExpTimeKey)),
    IsBias        = IsBias & IsExpTime0;
end
    
% check images median in compared to the typical selected bias images
if (InPar.CheckImage),
     MeanMean   = nanmedian(Mean(IsBias & IsGoodNoise));
     IsGoodMean = abs(Mean - MeanMean)./MeanMean < InPar.MaxBiasRatio;
end

if (InPar.Verbose),
    fprintf('Bias images search include total of %d images\n',Nim);
    if (InPar.CheckImage),
        fprintf('Found %d good bias images out of %d candidate bias images\n',length(find(IsBias & IsGoodNoise & IsGoodMean)),length(find(IsBias)));
    else
        fprintf('Found %d candidate bias images\n',length(find(IsBias)));
    end
end
    

if (~InPar.CheckImage),
    IsGoodNoise = true(Nim,1);
    IsGoodMean  = true(Nim,1);
end
    
    


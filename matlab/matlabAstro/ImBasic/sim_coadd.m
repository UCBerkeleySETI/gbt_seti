function [Coadd]=sim_coadd(Sim,varargin)
%--------------------------------------------------------------------------
% sim_coadd function                                               ImBasic
% Description: Image caoddition, with optional offset, scaling, weighting
%              and filtering.
% Input  : - Input images can be FITS, SIM or other types of
%            images. For input options see images2sim.m.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'Align'  - {true|false}. Default is false.
%                       If true then will align the images prior to
%                       coaddition using sim_align_shift.m.
%            'ScaleMethod' - Scale flux method.
%                       See sim_scaleflux.m for details.
%                       options are:
%                       'Const' - Scaling (divide the image ) by a user
%                                 defined constant. Default.
%                       (Value in 'ScaleConst').
%                       Additionl options include:
%                       'mode_fit', 'none', 'median', 'mean', 'std', 'var'
%                       Note that the scaling is done prior to background
%                       subtraction.
%            'ScaleConst' - Scale flux constant. Default is 1.
%                       Since the program require the images to have
%                       gain of 1, you can use this to convert the images
%                       gain to 1.
%            'BackMethod' - Background subtraction method.
%                       See sim_back_std.m for options. 
%                       options include:
%                       'none' - no background subtraction.
%                       'mode_fit' - Fit a Gaussian to the pixels
%                                histogram. Default.
%            'BackMethod'- Background subtraction method.
%                       See sim_back_std.m for options. 
%                       Default is 'mode_fit'.
%            'WeightMethod' - Weight flux method (this is identical to
%                       Scaling, but with additionla options).
%                       'none'  - no weighting.
%                       'const' - Weights provided in WeightConst.
%                       'ev'    - optimal E/V weights, calculated using
%                                 weights4coadd.m
%                       Default is 'const. If 'const',
%                       will use weight in WeightConst.
%            'WeightConst' - Vector of weights (per image). Default is 1.

%            'Filter' - Filters to use on each image.
%                       This can be a cell array of matrices,
%                       a single matrix, a singel cell or a SIM
%                       array of filters.
%                       Alternatively if 'psf' then will measure the PSF
%                       in each image using psf_builder.m and will filter
%                       each image with its PSF.
%                       Filter must contain odd by odd number of pixels.
%                       If empty then image will not be filtered.
%                       Default is empty.
%            'FiltNorm'- Normalize filters such their sum equal to this
%                        number. Default is 1. If empty, do not normalize.
%                        Default is 1.
%            'CombMethod' - Combine image method. Options are:
%                    'none' - do not combine image.
%                    'sum'  - sum images (default).
%                    'nnotnan' - number of not nan for each pixel.
%                    'fracnotnan' - fraction of not nan for each pixel.
%                    'median' - median of images.
%                    'mean'   - mean of images.
%                    'std'    - std of images.
%                    'var'    - variance of images.
%                    'min'    - minimum of images.
%                    'max'    - maximum of images.
%                    'range'  - range of values.
%                    'rss'    - square rott of sum of squares.
%                    'prod'   - product of images.
%                    'expsumlog' - exp(sum(log(Images))) - this is like
%                               'prod' by more stable numerically.
%                    'iqr'    - interquartile range between the 75th to
%                               the 25th percentiles.
%                    'prctile'- prctile with 'Prctile' parameter.
%                    'quantile' - quantile with 'Quantile' parameter.
%            'Prctile'    - Prctile parameter for prctile coaddition.
%            'Quantile'   - Quantile parameter for quantile coaddition.
%            'CombMethodWeight' - Method to use for constructing weight
%                           image. See 'CombMethod' for options.
%                           Default is 'fracnotnan'.
%            'CombMethodErr' - Method to use for constructing error
%                           image. See 'CombMethod' for options.
%                           Default is 'rss'.
%            'CombMethodBack' - Method to use for constructing background
%                           image. See 'CombMethod' for options.
%                           Default is 'sum'.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m, image2sim.m, sim_background.m, sim_scaleflux.m,
%            sim_filter.m, sim2cube.m, psf_builder.m, sim_align_shift.m
% Output : - Coadded image in SIM format.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Apr 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Coadd=sim_coadd('*.fits','Align',true);
%          Coadd=sim_coadd('*.fits','Align',true,'Filter','psf');
% Reliable: 2
%--------------------------------------------------------------------------

ImageField  = 'Im';
%HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';
WeightImField = 'WeightIm';


DefV.Align              = false;
DefV.ScaleMethod        = 'none';   % {'none'|'mode_fit'|...}
DefV.ScaleConst         = 1;
DefV.BackMethod         = 'mode_fit'; %none';   % {'none'|'mode_fit'|...}
DefV.WeightMethod       = 'const';
DefV.WeightConst        = 1;
DefV.Filter             = {};
DefV.FiltNorm           = 1;
DefV.CombMethod         = 'sum';
DefV.Prctile            = 0.0;
DefV.Quantile           = 0.5;
DefV.CombMethodWeight   = 'fracnotnan';
DefV.CombMethodErr      = 'rss';   
DefV.CombMethodBack     = 'sum';
DefV.Verbose            = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% read images
Sim   = images2sim(Sim,varargin{:});
Nim   = numel(Sim);

InPar.WeightConst = InPar.WeightConst.*ones(Nim,1);

% align images
if (InPar.Align),
    if (InPar.Verbose),
        fprintf('Align all images\n');
    end
    [Sim]=sim_align_shift(Sim,[],varargin{:});
end


% Scale images
switch lower(InPar.ScaleMethod)
    case 'none'
        % do nothing
    otherwise
        Sim = sim_scaleflux(Sim,varargin{:},'ScaleMethod',InPar.ScaleMethod,'ScaleConst',InPar.ScaleConst);
end

% remove background
switch lower(InPar.BackMethod)
    case 'none'
        % do nothing
    otherwise
        if (InPar.Verbose),
            fprintf('Subtract background from all images\n');
        end
        %Sim   = sim_background(Sim,varargin{:},'BackMethod',InPar.BackMethod);
        Sim   = sim_back_std(Sim,varargin{:},'BackStdAlgo',InPar.BackMethod);
end

% Weight images
switch lower(InPar.WeightMethod)
    case 'none'
        % do nothing
    case 'const'
        Sim = sim_scaleflux(Sim,varargin{:},'ScaleMethod',InPar.WeightMethod,'ScaleConst',InPar.WeightConst);
    case 'ev'
        if (InPar.Verbose),
            fprintf('Calculate weights using weights4coadd\n');
        end
        % optimal E/V weights calcualted using weights4coadd.m
        OutP = weights4coadd(Sim,'CalcVar',true,'CalcAbsZP',true,'PSFsame',true,varargin{:});
        InPar.WeightConst = OutP.AbsTran./OutP.Var;
        Sim = sim_scaleflux(Sim,varargin{:},'ScaleMethod','const','ScaleConst',InPar.WeightConst);
    otherwise
        error('Unknown WeightMethod option');
end

% measure the PSF in all images
if (ischar(InPar.Filter)),
    switch lower(InPar.Filter)
        case 'psf'
            if (exist('OutP','var')),
                InPar.Filter = OutP.PSF;
            else
                
                [PSF] = psf_builder(Sim,varargin{:});
                InPar.Filter = {PSF.PSF};
            end
        otherwise
            error('Unknown Filter option');
    end
end

% filter images    
if (~isempty(InPar.Filter)),
    %Sim = sim_filter(Sim,varargin{:},'Back',false, 'Filter',InPar.Filter, 'FiltPrep',InPar.FiltPrep, 'FiltNorm',InPar.FiltNorm);
    Sim = sim_filter(Sim,varargin{:},'Back',false, 'Filter',InPar.Filter, 'FiltNorm',InPar.FiltNorm);
end

Cube   = sim2cube(Sim,varargin{:});
DimIm  = 1;

% Combine images
Coadd = SIM;
Coadd.(ImageField)    = combine_imcube(Cube,InPar.CombMethod,DimIm,InPar);
Coadd.(WeightImField) = combine_imcube(Cube,InPar.CombMethodWeight,DimIm,InPar);
Coadd.(ErrImField)    = combine_imcube(Cube,InPar.CombMethodErr,DimIm,InPar);
Coadd.(BackImField)   = combine_imcube(Cube,InPar.CombMethodBack,DimIm,InPar);


end

%--------------------------------------------------------------------------
function Coadd=combine_imcube(Cube,Method,DimIm,InPar)
%--------------------------------------------------------------------------

% sigma clip prior to coaddition
%[Mean,StD,NpixUse]=clip_image_mean(Mat,varargin)

% coaddition
Nim = size(Cube,DimIm);
switch lower(Method)
    case 'none'
        % do nothing
        Coadd = [];
    case 'sum'
        Coadd = nansum(Cube,DimIm);
    case 'nnotnan'
        % number of values not equal NaN for each pixel
        Coadd = sum(~isnan(Cube),DimIm);
    case 'fracnotnan'
        % fraction of values not equal NaN for each pixel
        Coadd = sum(~isnan(Cube),DimIm)./Nim;
    case 'median'
        % median - allow sigma clip
%         [Coadd,~,NpixUse]=clip_image_mean(Cube,'MeanFun',@nanmedian,...
%                                                'RejectMethod',InPar.RejectMethod,...
%                                                'Reject',InPar.Reject,...
%                                                'MaxIter',InPar.MaxIter);
        Coadd = nanmedian(Cube,DimIm);
    case 'mean'
        Coadd = nanmean(Cube,DimIm);
    case 'std'
        Coadd = nanstd(Cube,[],DimIm);
    case 'var'
        Coadd = nanvar(Cube,[],DimIm);
    case 'min'
        Coadd = min(Cube,[],DimIm);
    case 'max'
        Coadd = max(Cube,[],DimIm);
    case 'range'
        Coadd = range(Cube,DimIm);
    case 'rss'
        % root of of sum of squares
        Coadd = sqrt(nansum(Cube.^2,DimIm));
    case 'prod'
        Coadd = prod(Cube,DimIm);
    case 'expsumlog'
        Coadd = exp(sum(log(Cube,DimIm)));
    case 'iqr'
        % returns the interquartile range between the 75th to the 25th percentiles
        Coadd = iqr(Cube,DimIm);
    case 'prctile'
        % 
        Coadd = prctile(Cube,InPar.Prctile.*100,DimIm);
    case 'quantile'
        % the value for which Quantile percent of the data is larger
        Coadd = quantile(Cube,InPar.Quantile,DimIm);
    otherwise
        error('Unknown CombMethod option');
end
Coadd = squeeze(Coadd);

end
function Sim=sim_maskstars(Sim,varargin)
%--------------------------------------------------------------------------
% sim_maskstars function                                           ImBasic
% Description: mask stars or high level pixels in a set of structure
%              array images.
% Input  : - Multiple images in one of the following forms:
%            (1) A structure array (SIM).
%            (2) A string containing wild cards (see create_list.m).
%            (3) A cell array of matrices.
%            (4) A file name that contains a list of files
%                (e.g., '@list', see create_list.m)
%            (5) A cell array of file names.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'ReplaceValue'- Replace value. Default is NaN.
%            'MaskStars' - One of the follwong mask stars algorithms:
%                       'no' - no stars masking.
%                       'val' - mask all pixels above a certain value.
%                       'prct' - mask all pixels at a specific upper
%                                percentile.
%                       'noise' - mask by S/N.
%                       'sex' - mask using sextractor segmentation image.
%            'MSPrct' - Percentile for 'prct' algorithm. Default is 0.01.
%            'MSVal'  - value for 'val' algorithm. Default is 1e4.
%            'Nsigma' - Number of sigma above background for masking.
%                       Default is 3.
%            --- images2sim.m parameters
%            'R2Field'- Name of the structure array field in which to
%                       store the image. Default is 'Im'.
%            'R2Name' - Name of the structure array field in which to
%                       store the image name. Default is 'ImageFileName'.
%            'ReadHead'- Read header information {true|false}.
%                       Default is true.
%            'ImType' - Image type. One of the following:
%                       'FITS'   - fits image (default).
%                       'imread' - Will use imread to read a file.
%                       'mat'    - A matlab file containing matrix,
%                                  or structure array.
%            'FitsPars'- Cell array of additonal parameters to pass to
%                        fitsread.m. Default is {}.
%            'ImreadPars'- Cell array of additonal parameters to pass to
%                        imread.m. Default is {}.
%            ---- image_background.m parameters
%            'BackMethod'- One of the following background estimation
%                       method:
%                       'mode'  - Mode of the image (default).
%                       'mean'  - Mean og the image.
%                       'median'- Median of the image.
%                       'medfilt'- 2D median filter of the image.
%                       'ordfilt'- 2D order filter of the image.
%                       'polys' - Polynomial surface of the image.
%            'MedFiltSize' - Median and order filter box size.
%                       Default is [65 65].
%            'Order'  - Fraction of the order filter. Default is 0.3.
%            'PolyX'  - Row vector of the orders of X polynomials in the fit.
%                     For example: [1 3], will add a term Px1.*X + Px2.*X.^3.
%                     Default is [1 2].
%            'PolyY'  - Row vector of the orders of Y polynomials in the fit.
%                     Default is [1 2].
%            'PolyXY'- A two column matix of the cross terms orders.
%                     For example: [1 1; 3 4], will add a term:
%                     Pxy1.*X.*Y + Pxy2.*X.^3.*Y.^4.
%                     Default is [1 1].
%            'MaxIter'- Maximum number of sigma clipping iterations.
%                     Default is 0 (i.e., no sigma clipping).
%            'ClipMethod'- Sigma clipping method (see clip_resid.m for details).
%                     Default is 'StD'.
%            'ClipMean' - Method by which to calculate the "mean" of the sample
%                     in the sigma clipping process
%                     (see clip_resid.m for details). Default is 'median'.
%            'Clip' - Two elements vector containing the lower and upper
%                     values for the sigma clipping. This is [Lower, Upper]
%                     number of sigmas (positive) below/above the mean
%                     (see clip_resid.m for details).
%                     See clip_resid.m for defaults.
%            --- image_noise.m parameters
%            'NoiseMethod' - One of the following background estimation
%                       method:
%                       'poisson' - Estimated poisson noise (default).
%                       'std'  - Mode of the image.
%                       'order'- Use two order filters to estimate
%                                the 50-percentile, and estimate the std.
%            'NoiseFiltSize' - Order filter box size.
%                       Default is [65 65].
%            'Gain'   - CCD Gain. Default is 1.
%            'RN'     - CCD readout noise. Default is 10 e-.
%            --- run_sextractor.m parameters
%            'DETECT_THRESH' - SExtractor detection threshold.
%                       Default is 4.0.
%            'DETECT_MINAREA' - SExtractor minimum area threshold.
%                       Default is 5.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=sim_maskstars('lred012*.fits','MaskStars','Noise');
%          Sim=sim_maskstars(Sim);
%          Sim=sim_maskstars('lred012[1-2].fits','MaskStars','sex');
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
%HeaderField = 'Header';
FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

% input parameters
% images2sim.m parameters
DefV.R2Field    = ImageField;
DefV.R2Name     = FileField;
DefV.ReadHead   = true;
DefV.ImType     = 'FITS';
DefV.FitsPars   = {};
DefV.ImreadPars = {};
% mask stars
DefV.MaskStars       = 'prct';  %  {'no','val','prct','noise','sex'}
DefV.MSPrct          = 0.01;
DefV.MSVal           = 1e4;
DefV.Nsigma          = 3;
% image_background.m parameters
DefV.BackMethod      = 'mode';     
DefV.MedFiltSize     = [65 65];
DefV.Order           = 0.3;
DefV.PolyX           = [1 2];
DefV.PolyY           = [1 2];
DefV.PolyXY          = [1 1];
DefV.MaxIter         = 0;
DefV.ClipMethod      = 'StD';
DefV.ClipMean        = 'median';
DefV.Clip            = [];
% image_noise.m parameters
DefV.NoiseMethod     = 'poisson';     
DefV.NoiseFiltSize   = [65 65];
DefV.Gain            = 1;
DefV.RN              = 10;
% run_sextractor.m
DefV.DETECT_THRESH   = 4.0;
DefV.DETECT_MINAREA  = 5;
%
DefV.ReplaceValue    = NaN;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

Sim=images2sim(Sim,varargin{:});

Nim = numel(Sim);  % number of images
for Iim=1:1:Nim,
    switch lower(InPar.MaskStars)
        case 'no'
            % do nothing
        case 'prct'
            Val = quantile(Sim(Iim).(InPar.R2Field)(:),1-InPar.MSPrct);
            Sim(Iim).(InPar.R2Field)(Sim(Iim).(InPar.R2Field)>Val) = InPar.ReplaceValue;
        case 'val'
            Sim(Iim).(InPar.R2Field)(Sim(Iim).(InPar.R2Field)>InPar.MSVal) = InPar.ReplaceValue;
        case 'noise'
            Back  = image_background(Sim(Iim).(InPar.R2Field),varargin{:});
            Noise = image_noise(Sim(Iim).(InPar.R2Field),varargin{:});
            Sim(Iim).(InPar.R2Field)(Sim(Iim).(InPar.R2Field)>(Back+Noise.*InPar.Nsigma)) = InPar.ReplaceValue;
            
        case 'sex'
            TmpSegIm = tempname;
            TmpImage = tempname;
            fitswrite(Sim(Iim).(InPar.R2Field),TmpImage);
            run_sextractor(TmpImage,...
                               [],[],[],[],...
                               'DETECT_THRESH',sprintf('%f',InPar.DETECT_THRESH),...
                               'DETECT_MINAREA',sprintf('%d',InPar.DETECT_MINAREA),...
                               'CHECKIMAGE_TYPE','SEGMENTATION',...
                               'CHECKIMAGE_NAME',TmpSegIm);
            SegIm = fitsread(TmpSegIm);   % read segmentation image
            delete(TmpSegIm);             % delete segmentation FITS file
            delete(TmpImage);
            Sim(Iim).(InPar.R2Field)(SegIm>0) = InPar.ReplaceValue;
        otherwise
            error('Unknown MaskStars option');
    end
end   
            
            
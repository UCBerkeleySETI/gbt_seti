function BiasSim=bias_construct(Images,varargin)
%--------------------------------------------------------------------------
% bias_construct function                                          ImBasic
% Description: Construct a bias (or dark) image from a set bias (or dark)
%              images.
%              This function is not responsible for selecting good
%              bias images.
% Input  : - Bias images from which to construct the bias image.
%            The following inputs are possible:
%            (1) Cell array of image names in string format.
%            (2) String containing wild cards (see create_list.m for
%                option). E.g., 'lred00[15-28].fits' or 'lred001*.fits'.
%            (3) Structure array of images (SIM).
%                The image should be stored in the 'Im' field.
%                This may contains also mask image (in the 'Mask' field),
%                and an error image (in the 'ErrIm' field).
%            (4) Cell array of matrices.
%            (5) A file contains a list of image (e.g., '@list').
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
%            'BiasFile' - Bias FITS file to save.
%                        Default is 'Bias.fits'. If empty then do not save.
%            'MaskFile' - Mask bit file to save. Default is empty.
%                        If empty then do not save.
%            'ErrImFile' - Relative error image to save. Default is empty.
%                         If empty then do not save.
%            'DataType' - FITS image type. Default is 'float32'.
%            'MaskType' - Mask image type. Default is 'uint16'.
%            'CreateErr'- Create error image and store it in the
%                         output structure image {false|true}.
%                         Default is true.
%            'Mask'     - Input mask file in which to add the bit mask.
%                         Default is empty. If empty will create a new
%                         mask.
%            'CreateMask'- Create a mask file and store it in the
%                         output structure image {false|true}.
%                         Default is true.
%            'Bit_BiasNoisy' - Mask bit index (or function handle) for
%                         noisy pixels.
%            'Bit_BiasNoise0' - Mask bit index (or function handle) for
%                         pixels with zero noise.
%            'MinNimPix' - Minimum number of images in flat construction.
%                          If number of images is smaller than set the
%                          Bit_FlatLowNim bit to true.
%            'MaxRelErr' - Maximum of relative error above to flag the
%                          'Bit_MaxRelErr' bit.
%            'RField'    - Field name in which the images are stored
%                          in the structure array. Default is 'Im'.
%            'Method'    - combining method:
%                          {'sum' | 'mean' | 'median' | 'std'},
%                          default is 'median'.
%            'Zero'      - Image offset (to add before scale),
%                          {'mean' | 'median' | 'mode' | 'constant' | 'none'},
%                          default is 'none'.
%                          Note that the zero willbe applied before the scaling.
%            'ZeroConst' - A scalar or vector of a constant offsets to be applied to
%                          each image. Default is [1].
%                          This is being used only if zero=constant.
%            'Scale'     - Image scaling {'mean' | 'median' | 'mode' | 'constant' | 'none'},
%                          default is 'none'.
%            'ScaleConst'- A scalar or vector of a constant scale to be applied to
%                          each image. Default is [1].
%                          This is being used only if scale=constant.
%            'Weight'    - Method by which to wheight the images.
%                          {'mean' | 'median' | 'mode' | 'constant' | 'images' | 'none'},
%                          default is 'none'.
%            'WeightConst'-A scalar or vector of a constant weight to be applied to
%                          each image. Default is [1].
%                          This is being used only if weight=constant.
%            'WeightFun' - Function name to apply to the weight value before weighting
%                          (e.g., @sqrt).
%                          If empty matrix (i.e., []) than donot apply a function,
%                          default is empty matrix.
%            'WeightIm'  - Set of images to use as weights.
%                          This can be a cell array of images, or a 3D cube.
%                          Default is [].
%            'Reject'    - Pixels rejection methods:
%                          'sigclip'   - std clipping.
%                          'minmax'    - minmax rejuction.
%                          'none'      - no rejuction, default.
%                          'sigclip' and 'minmax' are being applied only if
%                          Niter>1.
%            'RejPar'    - rejection parameters [low, high].
%                          for 'sigclip', [LowBoundSigma, HighBoundSigma], default is [Inf Inf].
%                          for 'minmax', [Number, Number] of pixels to reject default is [0 0].
%            'Niter'     - Number of iterations in rejection scheme. Default is 1.
%            'Verbose'   - Print progress messages {true|false}.
%                          Default is false.
% Output : - Structure image with the bias image, mask, error image and
%            header.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: BiasSim=bias_construct('lred012*.fits');
%          BiasSim=bias_construct(Sim(is_bias_image(Sim)));
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
MaskField   = 'Mask';
%BackImField = 'BackIm';
ErrImField  = 'ErrIm';

% input parameters
% images2sim.m parameters
DefV.ReadHead   = true;
DefV.ImType     = 'FITS';
DefV.FitsPars   = {};
DefV.ImreadPars = {};
% save fits
DefV.BiasFile    = 'Bias.fits';
DefV.MaskFile    = [];
DefV.ErrImFile   = [];
DefV.DataType   = 'float32';
DefV.MaskType   = 'uint16';
% bit mask
DefV.CreateErr      = true;
DefV.Mask           = [];
DefV.CreateMask     = true;   
DefV.Bit_BiasNoisy  = @def_bitmask_specpipeline;
DefV.Bit_BiasNoise0 = @def_bitmask_specpipeline;
DefV.MaxBiasErr      = 50;
DefV.MinBiasErr      = 0.001;
% sim_combine.m parameters
DefV.RField      = ImageField;
DefV.Method      = 'median';
DefV.Zero        = 'none';
DefV.Scale       = 'none';
DefV.Reject      = 'none';
DefV.RejPar      = [];
DefV.Niter       = 1;
% others
DefV.Verbose     = false;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% read images to SIM
Sim  = images2sim(Images,varargin{:});
Nim   = numel(Sim);

% convert Sim to a cube
%Cube = sim2cube(Sim,'ImDim',3);

%--- calculate bias field ---
%[BiasSim.(ImageField)] = imcombine(Sim,varargin{:},'RField',InPar.RField);
[BiasSim.(ImageField)] = sim_combine(Sim,varargin{:},'RField',InPar.RField);

% flat field error (std)
if (InPar.CreateErr),
    %BiasSim.(ErrImField) = imcombine(Sim,varargin{:},'RField',InPar.RField,'Method','std');
    BiasSim.(ErrImField) = sim_combine(Sim,varargin{:},'RField',InPar.RField,'Method','std');
end

if (InPar.Verbose),
    fprintf('Bias image constructed using %d images\n',Nim);
    fprintf('Median bias level = %f\n',nanmedian(BiasSim.(ImageField)(:)));
    fprintf('StD of bias       = %f\n',nanstd(BiasSim.(ImageField)(:)));
end

%--- Flag bad pixels ---
if (InPar.CreateMask),
    % get bit indices
    % Bit_BiasNoisy  Bit_BiasNoise0
    InPar.Bit_BiasNoisy  = get_bitmask_def(InPar.Bit_BiasNoisy,'Bit_BiasNoisy');
    InPar.Bit_BiasNoise0 = get_bitmask_def(InPar.Bit_BiasNoise0,'Bit_BiasNoise0');
    
    % set bits    
    BiasSim.(MaskField) = maskflag_set([],InPar.MaskType,...
                                       InPar.Bit_BiasNoisy,...
                                       BiasSim.(ErrImField) > InPar.MaxBiasErr,...
                                       InPar.Bit_BiasNoise0,...
                                       BiasSim.(ErrImField) < InPar.MinBiasErr);
                                   
end

%--- update header ---

CellHead = cell_fitshead_addkey([],'TYPE','bias','Image type',...
                                   'IMTYPE','bias','Image type',...
                                   'COMMENT','','Bias image generated by flat_construct.m',...
                                   'COMMENT','',sprintf('Combine method: %s',InPar.Method),...
                                   'COMMENT','',sprintf('Scale method: %s',InPar.Scale),...
                                   'NUM_IM',Nim,'Number of images from which bias was constructed');
                               
BiasSim.(HeaderField) = CellHead;

if (~isempty(InPar.BiasFile)),
    fitswrite_my(BiasSim.(ImageField),InPar.BiasFile,CellHead,InPar.DataType);
end
if (~isempty(InPar.MaskFile)),
    CellHead1 = cell_fitshead_addkey(CellHead,'COMMENT','','Bias Mask image');
    fitswrite_my(BiasSim.(MaskField),InPar.MaskFile,CellHead1,InPar.MaskType);
end
if (~isempty(InPar.ErrImFile)),
    CellHead1 = cell_fitshead_addkey(CellHead,'COMMENT','','Bias Error image');
    fitswrite_my(BiasSim.(ErrImField),InPar.ErrImFile,CellHead1,InPar.DataType);
end


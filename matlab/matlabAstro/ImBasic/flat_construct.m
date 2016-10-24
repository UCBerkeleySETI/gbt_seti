function FlatSim=flat_construct(Images,varargin)
%--------------------------------------------------------------------------
% flat_construct function                                          ImBasic
% Description: Construct a flat field image from a set of images.
%              This function is not responsible for selecting good
%              flat images.
% Input  : - Bias subtracted flat images from which to construct
%            the flat image.
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
%            --- All the parameters of sim_maskstars.m (not listed here).
%            'MaskStars' - Mask stars prior to combining images.
%                       See sim_maskstars.m for options. Default is 'no'.
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
%            'FlatFile' - Flat field FITS file to save.
%                        Default is 'Flat.fits'. If empty then do not save.
%            'MaskFile' - Mask bit file to save. Default is empty.
%                        If empty then do not save.
%            'ErrImFile' - Relative error image to save. Default is empty.
%                        If empty then do not save.
%            'DataType' - FITS image type. Default is 'float32'.
%            'MaskType' - Mask image type. Default is 'uint16'.
%            'EqDispDir'- Normalize the flat such that the mean along
%                         one of the dimesniosn will be one.
%                         Will use the same normalization function as for
%                         the flat {false|true}. Default is false.
%                         NOTE: This may produce a problem if the
%                         polynomial fit return values which are
%                         below or equal to zero.
%            'DispDir'  - Direction along to normalize {'x'|'y'}.
%                         For spectroscopic observations, this is the
%                         dispersion direction. Default is 'x'.
%            'DispPolyDeg'- Degree of polynomial to use in the dispersion
%                         direction normalization. Default is 11.
%                         If empty, the will use the median of the
%                         dispersion axis instead of the polynomial fit.
%            'NormFF'   - Flat field normalization 
%                         {'mean'|'median'|'std'|const}.
%                         Default is 'median'.
%            'CreateErr'- Create error image and store it in the
%                         output structure image {false|true}.
%                         Default is true.
%            'Mask'     - Input mask file in which to add the bit mask.
%                         Default is empty. If empty will create a new
%                         mask.
%            'CreateMask'- Create a mask file and store it in the
%                         output structure image {false|true}.
%                         Default is true.
%            'Bit_FlatNaN'- Index of the bit (in the bit mask) in which to
%                         indicate that the flat is NaN.
%                         Default is @def_bitmask_specpipeline.
%                         See get_bitmask_def.m for details.
%            'Bit_FlatLowNim'- Index of the bit (in the bit mask) in which
%                         to indicate that the number of images that
%                         were used to constrct the flat in that pixel are 
%                         smaller than the 'MinNimPix' parameter.
%                         Default is @def_bitmask_specpipeline.
%                         See get_bitmask_def.m for details.
%            'Bit_MaxRelErr'- Index of the bit (in the bit mask) in which to
%                         indicate that the flat relative error is larger
%                         than the 'MaxRelErr' parameter.
%                         Default is @def_bitmask_specpipeline.
%                         See get_bitmask_def.m for details.
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
%            'Scale'     - Image scaling {'mean' | 'median' | 'mode' | 'constant' | 'none'},
%                          default is 'median'.
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
% Output : - Structure image with the flat image, mask, error image and
%            header.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: FlatSim=flat_construct('lred012*.fits');
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
MaskField   = 'Mask';
%BackImField = 'BackIm';
ErrImField  = 'ErrIm';

DefV.MaskStars  = 'no';
% images2sim.m parameters
DefV.ReadHead   = true;
DefV.ImType     = 'FITS';
DefV.FitsPars   = {};
DefV.ImreadPars = {};
% save fits
DefV.FlatFile    = 'Flat.fits';
DefV.MaskFile    = [];
DefV.ErrImFile   = [];
DefV.DataType   = 'float32';
DefV.MaskType   = 'uint16';
% input arguments
DefV.EqDispDir      = false;
DefV.DispDir        = 'x';
DefV.DispPolyDeg    = []; %11;
% normalization
DefV.NormFF         = 'median';  % {'mean'|'median'|'std'|const}
% bit mask
DefV.CreateErr      = true;
DefV.Mask           = [];
DefV.CreateMask     = true;
DefV.Bit_FlatNaN    = @def_bitmask_specpipeline;
DefV.Bit_FlatLowNim = @def_bitmask_specpipeline;
DefV.MinNimPix      = 3;
DefV.Bit_MaxRelErr  = @def_bitmask_specpipeline;
DefV.MaxRelErr      = 0.1;
% sim_combine.m parameters
DefV.RField      = ImageField;
DefV.Method      = 'median';
DefV.Zero        = 'none';
DefV.Scale       = 'median';
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

%--- calculate mean bias ---
%[FlatSim.(ImageField),~,~,NN_Counter] = imcombine(Sim,varargin{:},'RField',InPar.RField);
[FlatSim.(ImageField),~,~,NN_Counter] = sim_combine(Sim,varargin{:},'RField',InPar.RField);
% bias error (std)
if (InPar.CreateErr),
    %FlatSim.(ErrImField) = imcombine(Sim,varargin{:},'RField',InPar.RField,'Method','std');
    FlatSim.(ErrImField) = sim_combine(Sim,varargin{:},'RField',InPar.RField,'Method','std');
end
%--- Flag bad pixels ---
if (InPar.CreateMask),
    % get bit indices
    InPar.Bit_MaxRelErr  = get_bitmask_def(InPar.Bit_MaxRelErr,'Bit_MaxRelErr');
    InPar.Bit_FlatNaN    = get_bitmask_def(InPar.Bit_FlatNaN,'Bit_FlatNaN');
    InPar.Bit_FlatLowNim = get_bitmask_def(InPar.Bit_FlatLowNim,'Bit_FlatLowNim');
    
    % set bits    
    FlatSim.(MaskField) = maskflag_set([],InPar.MaskType,...
                                       InPar.Bit_MaxRelErr,...
                                       FlatSim.(ErrImField)./FlatSim.(ImageField) > InPar.MaxRelErr,...
                                       InPar.Bit_FlatNaN,...
                                       isnan(FlatSim.(ImageField)),...
                                       InPar.Bit_FlatLowNim,...
                                       NN_Counter<InPar.MinNimPix);
                                       
end

%--- normalize flat ---
if (ischar(InPar.NormFF)),
    switch lower(InPar.NormFF)
        case 'median'
            NormFF = nanmedian(FlatSim.(ImageField)(:));
        case 'mean'
            NormFF = nanmean(FlatSim.(ImageField)(:));
        case 'std'
            NormFF = nanstd(FlatSim.(ImageField)(:));
        otherwise
            error('Unknown NormFF option');
    end
else
    NormFF = InPar.NormFF;
end
% normalize 
FlatSim.(ImageField) = FlatSim.(ImageField)./NormFF;
% normalize flat error
% NOTE: flat error is a relative error
if (InPar.CreateErr),
    FlatSim.(ErrImField) = FlatSim.(ErrImField)./NormFF;
end

% spectroscopy normalization
% normalize in dispersion direction
if (InPar.EqDispDir),
    %error('Not available yet');
    DispDirMap = {'x','y'};
    DispDirInd = find(strcmpi(DispDirMap,InPar.DispDir));
    SpecCut    = nanmedian(FlatSim.(ImageField),DispDirInd);
    if (isempty(InPar.DispPolyDeg)),
        % use the SuperCut instead of fitting a polynomial
        YY = SpecCut;
        %plot(YY)
    else
        XX = (1:1:length(SpecCut)).';
        if (DispDirInd==1),
            P  = polyfit(XX,SpecCut.',InPar.DispPolyDeg);
            YY = polyval(P,XX).';
        else
            P  = polyfit(XX,SpecCut,InPar.DispPolyDeg);
            YY = polyval(P,XX);
        end
        %plot(SpecCut)
        %hold on
        %plot(YY,'r-')
    end
    FlatSim.(ImageField) = bsxfun(@times,FlatSim.(ImageField),1./YY);
    if (InPar.CreateErr),
        FlatSim.(ErrImField) = bsxfun(@times,FlatSim.(ErrImField),1./YY);
    end
end

if (InPar.Verbose),
    fprintf('Flat image constructed using %d images\n',Nim);
    fprintf('Median of normalized flat = %f\n',nanmedian(FlatSim.(ImageField)(:)));
    fprintf('StD of normalized flat    = %f\n',nanstd(FlatSim.(ImageField)(:)));
end

%--- update header ---
% write to header:
% number of images
% max std
% min std
% mean image
% median image
% min image
% max image
CellHead = cell_fitshead_addkey([],'TYPE','flat','Image type',...
                                   'IMTYPE','flat','Image type',...
                                   'COMMENT','','Norm flat image generated by flat_construct.m',...
                                   'COMMENT','',sprintf('Combine method: %s',InPar.Method),...
                                   'COMMENT','',sprintf('Scale method: %s',InPar.Scale),...
                                   'COMMENT','',sprintf('Norm method: %s',num2str(InPar.NormFF)),...
                                   'NUM_IM',Nim,'Number of images from which flat was constructed');
                               
FlatSim.(HeaderField) = CellHead;

if (~isempty(InPar.FlatFile)),
    fitswrite_my(FlatSim.(ImageField),InPar.FlatFile,CellHead,InPar.DataType);
end
if (~isempty(InPar.MaskFile)),
    CellHead1 = cell_fitshead_addkey(CellHead,'COMMENT','','Flat Mask image');
    fitswrite_my(FlatSim.(MaskField),InPar.MaskFile,CellHead1,InPar.MaskType);
end
if (~isempty(InPar.ErrImFile)),
    CellHead1 = cell_fitshead_addkey(CellHead,'COMMENT','','Flat Relative Error image');
    fitswrite_my(FlatSim.(ErrImField),InPar.ErrImFile,CellHead1,InPar.DataType);
end


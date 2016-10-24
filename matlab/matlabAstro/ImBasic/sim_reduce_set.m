function [Sim,BiasSim,FlatSim,Is]=sim_reduce_set(Sim,varargin)
%--------------------------------------------------------------------------
% sim_reduce_set function                                          ImBasic
% Description: Reduce a set of images taken in the same configuration
%              (e.g., identical image size, filter, Gain and RN).
%              The reduction may include: flagging of saturated pixels,
%              bias subtraction, flat field correction, CR flagging
%              and removal.
% Input  : - Images to reduce.
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
%           'TrimSec' - Triming section [xmin xmax ymin ymax] or an header
%                       keyword name that contains the trim section.
%                       If emptythen donot trim image.
%                       Assumed to be identical for all images.
%           'MaskSaturated' - Flag in the mask image saturated pixels
%                       {true|false}. Default is true.
%           'BiasSec' - Overscan bias section [xmin xmax ymin ymax] or an
%                       header keyword name that contains the bias section.
%                       If emptythen donot subtract overscan bias.
%                       Assumed to be identical for all images.
%           'BitMaskFun' - Handle function that retuen the appropriate
%                       bit indices. See @def_bitmask_specpipeline
%                       for example. If empty then use internal defaults
%                       of each function. Default is empty. 
%           'Gain'    - CCD Gain. If string, then this is the header keyword
%                       that contains the gain. Default is 1.
%                       Assumed to be identical for all images.
%           'RN'      - CCD Read noise [e-]. If string, then this is the 
%                       header keyword that contains the gain. Default is 10.
%                       Assumed to be identical for all images.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m, sim_mask_saturated.m, is_bias_image.m
%            bias_construct.m, sim_bias.m, sim_suboverscan.m,
%            is_flat_image.m, flat_construct.m, sim_flat.m,
%            sim_trim.m, sim_crdetect.m
% Output : - Structure array of reduced images (bias subtract and flat
%            corrected) with their  mask images.
%          - Structure containing the master bias image and its error.
%          - Structure containing the master flat image and its error .
%          - A structure containing logical flags regarding the nature
%            of the input images e.g., IsBias, IsFlat, etc.).
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Sim,BiasSim,FlatSim,Is]=sim_reduce_set('red*.fits','BitMaskFun',@def_bitmask_specpipeline);
% Reliable: 2
%--------------------------------------------------------------------------


%ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';


DefV.TrimSec         = [];     % default - no trimming 
DefV.MaskSaturated   = true;
DefV.BiasSec         = [];     % if empty - no overscan subtraction
DefV.BitMaskFun      = [];     % if empty use default - if handle use uniform handle in all functions - @def_bitmask_specpipeline
DefV.Gain             = 1;
DefV.RN               = 10;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (isempty(InPar.BitMaskFun)),
    BitPars = {};
else
    BitPars = {'Bit_ImSaturated',InPar.BitMaskFun,...
               'Bit_BiasNoisy',InPar.BitMaskFun,...
               'Bit_BiasNoise0',InPar.BitMaskFun,...
               'Bit_FlatNaN',InPar.BitMaskFun,...
               'Bit_FlatLowNim',InPar.BitMaskFun,...
               'Bit_MaxRelErr',InPar.BitMaskFun,...
               'Bit_CR',InPar.BitMaskFun};
end
       
% read images and headers
Sim   =images2sim(Sim,varargin{:});
%Nim   = numel(Sim);

% since all images are identical get the trim section
% from the first image
TrimSec = get_ccdsec_head(Sim(1).(HeaderField),InPar.TrimSec);
    
% get RN and Gain
if (ischar(InPar.Gain)),
    Val = sim_getkeyval(Sim(1),InPar.Gain,'ConvNum',true);
    Gain = Val{1};
else
    Gain = InPar.Gain;
end
if (ischar(InPar.RN)),
    Val = sim_getkeyval(Sim(1),InPar.RN,'ConvNum',true);
    RN = Val{1};
else
    RN = InPar.RN;
end
    

% mark saturated images and create saturation mask for each image
if (InPar.MaskSaturated),
    Sim=sim_mask_saturated(Sim,varargin{:},BitPars{:}); %,'Bit_ImSaturated',InPar.BitMaskDef,'SatLevel',InPar.SatLevel);
end

% select bias images
[Is.IsBias,Is.IsGoodNoise,Is.IsGoodMean,Is.IsBiasKey]=is_bias_image(Sim,varargin{:},'Gain',Gain,'RN',RN); %'DateKey',InPar.DateKey,'ExpTimeKey',InPar.ExpTimeKey);
Is.IsGoodBias  = Is.IsBias & Is.IsGoodNoise & Is.IsGoodMean;
% construct bias
BiasSim        = bias_construct(Sim(Is.IsGoodBias),varargin{:},BitPars{:}); %,'Bit_FlatNaN',InPar.BitMaskDef,'Bit_FlatLowNim',InPar.BitMaskDef,'Bit_MaxRelErr',InPar.BitMaskDef);

% construct Readout noise image [e-]
%ReadNoiseImage = BiasSim.ErrIm.*Gain;

% subtract bias
Sim            = sim_bias(Sim,varargin{:},'BiasImage',BiasSim);  % return bias subtracted SIM
% subtract overscan bias
if (~isempty(InPar.BiasSec)),
    Sim=sim_suboverscan(Sim,varargin{:},'BiasSec',InPar.BiasSec); % return overscan bias subtracted SIM
end

% select flat
[Is.IsFlat,Is.IsNotSaturated]=is_flat_image(Sim,varargin{:}); %'SatLevel',InPar.SatLevel);
Is.IsGoodFlat     = Is.IsFlat & Is.IsNotSaturated;
% construct flat
FlatSim        = flat_construct(Sim(Is.IsGoodFlat),BitPars{:});
% correct for flat
Sim            = sim_flat(Sim,varargin{:},'FlatImage',FlatSim);

% trim images
Sim            = sim_trim(Sim,varargin{:},'TrimSec',InPar.TrimSec);    % return trimmed images
FlatSim        = sim_trim(FlatSim,varargin{:},'TrimSec',TrimSec);
BiasSim        = sim_trim(BiasSim,varargin{:},'TrimSec',TrimSec);

% select arc images
%[IsArc,ImageArcName]=is_arc_image(Sim,varargin{:}); %,'CompareArcKeyVal','bit');

% select science images
% all the images which are not bias or flat
Is.IsScience = ~Is.IsBiasKey & ~Is.IsFlat;

% search for cosmic rays
% can use ReadNoiseImage instead of InPar.RN
Sim(Is.IsScience)  = sim_crdetect(Sim(Is.IsScience),varargin{:},BitPars{:},'Gain',Gain,'RN',RN); %,'Clean',InPar.CleanCR,'Gain',Gain,'RN',RN,'BWmorphN',InPar.BWmorphN,'Bit_CR',InPar.BitMaskDef);

        

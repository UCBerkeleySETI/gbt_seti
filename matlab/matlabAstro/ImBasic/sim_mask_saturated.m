function Sim=sim_mask_saturated(Sim,varargin)
%--------------------------------------------------------------------------
% sim_mask_saturated function                                      ImBasic
% Description: For each image in a list of images, look for saturated
%              pixels and generate a bit mask image with the saturated
%              pixels marked.
% Input  : - Images in which to look for saturated pixels.
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
%            'SatLevel'- Image saturation level. This is either a number
%                        or an header keyword name that containing
%                        the saturation level.
%                        Default is 60000.
%            'SameSatLevel' - Assume that saturation level is identical
%                        for all images {true|false}. Default is false.
%            'MaskType' - If a mask file is created then
%                      this is the data type of the mask.
%                      Default is 'uint16'.
%            'Bit_ImSaturated' - The bit index which to flag the saturated
%                      pixels. Alternatively, this can be a function
%                      handle which get the the bit name
%                      (i.e., 'Bit_ImSaturated') and return the bit index.
%                      Default is @def_bitmask_specpipeline.
%            'FlagOp' - Bit binary operation between the input mask and
%                      saturated pixels mask {'or'|'and'|'onlysat'}.
%                      Default is 'or'.
%                      'onlysat' will use only the saturated pixels mask.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - An array with the number of saturated pixels in each image.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% See also: is_saturated_image.m
% Example: Sim=sim_mask_saturated('*.fits');
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

DefV.SatLevel         = 60000;
DefV.SameSatLevel     = false;
DefV.MaskType         = 'uint16';
DefV.Bit_ImSaturated  = @def_bitmask_specpipeline;
DefV.FlagOp           = 'or';    % {'or'|'and'|'onlysat'}

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

InPar.Bit_ImSaturated = get_bitmask_def(InPar.Bit_ImSaturated,'Bit_ImSaturated');

SatPix = zeros(size(Sim));
for Iim=1:1:Nim,
    if (~InPar.SameSatLevel || Iim==1)
       
        if (ischar(InPar.SatLevel)),
             NewCellHead = cell_fitshead_getkey(Header,InPar.SatLevel,NaN);
             if (~isnan(NewCellHead{1,2})),
                 SatLevel = NewCellHead{1,2};
             else
                 error('Saturation level is not available in header');
             end
        else
            SatLevel = InPar.SatLevel;
        end
    end
    
    % count the number of saturated pixels
    FlagSaturated = Sim(Iim).(ImageField)>SatLevel;
    
    SatPix(Iim)   = numel(find(FlagSaturated));
    
    % create a new mask if needed
    if (~isfield(Sim(Iim),MaskField)),
        Sim(Iim).(MaskField) = zeros(size(Sim(Iim).(ImageField)),InPar.MaskType);
    end
    if (isempty(Sim(Iim).(MaskField))),
        Sim(Iim).(MaskField) = zeros(size(Sim(Iim).(ImageField)),InPar.MaskType);
    end
    
    Mask = maskflag_set([],InPar.MaskType,InPar.Bit_ImSaturated,FlagSaturated);
    
    switch lower(InPar.FlagOp)
         case 'or'
             Sim(Iim).(MaskField) = bitor(Sim(Iim).(MaskField),Mask);
         case 'and'
             Sim(Iim).(MaskField) = bitand(Sim(Iim).(MaskField),Mask);
         case 'onlysat'
             Sim(Iim).(MaskField) = Mask;
         otherwise
             error('Unknown FlagOp option');
    end
end

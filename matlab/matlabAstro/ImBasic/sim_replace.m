function Sim=sim_replace(Sim,varargin)
%--------------------------------------------------------------------------
% sim_replace function                                             ImBasic
% Description: Replace pixels in a give value ranges with other values
%              in a set of structure images (SIM).
% Input  : - Set of images in which to replace pixel values.
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
%            'Range'    - Matrix containing ranges to replace:
%                         [Min Max; Min Max; ...].
%                         Where each line contains a Min Max of a range.
%                         The program will look for pixel values in this
%                         range and will replace them with the appropriate
%                         value in the 'Value' input parameter.
%                         Default is [-Inf 0].
%            'Value'    - Vector of values to use as a replacments.
%                         Each element in the vector corresponds to each
%                         row in the 'Range' matrix. If scalar then
%                         use the same value for all the rows.
%                         Default is 0.
%            'ImageField'- Image field in the SIM structure.
%                        Default is 'Im'.
%            'MaskField' - Mask field in the SIM structure.
%                        Default is 'Mask'.
%            'CreateMask' - Propogate mask image to the output structure
%                      array of images {true|false}. Default is false.
%            'MaskType' - If a mask file is created (FlagOp='no') then
%                      this is the data type of the mask.
%                      Default is 'uint16'.
%            'Bit_Replace' - The bit index which is used in order to flag
%                      pixels that were replaced.
%                      Alternatively, this can be a function
%                      handle which get the the bit name
%                      (i.e., 'Bit_Divide0') and return the bit index.
%                      Default is 1.
%            'CopyHead' - Copy header from original image {'y' | 'n'}.
%                         Default is 'y'.
%            'AddHead'  - Cell array with 3 columns containing additional
%                         keywords to be add to the header.
%                         See cell_fitshead_addkey.m for header structure
%                         information. Default is empty matrix.
%            'DelDataSec' - Delete the 'DATASEC' header keyword
%                         {true|false}. Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - Structure of images with replaced pixels.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Im=rand(5,5);Im(1,1)=10;Im(3,4)=-10;
%          Sim=sim_replace(Im,'Range',[-Inf 0; 1 Inf],'Value',[-1 2]);
% Reliable: 2
%-----------------------------------------------------------------------------
FunName = mfilename;

ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

DefV.Range       = [-Inf 0];
DefV.Value       = 0;
DefV.CreateMask  = false;
DefV.MaskType    = 'uint16';
DefV.Bit_Replace = 1;
DefV.ImageField  = ImageField;
DefV.MaskField   = MaskField;
DefV.CopyHead    = 'y';
DefV.AddHead     = [];
DefV.DelDataSec  = true;

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

ImageField = InPar.ImageField;
MaskField  = InPar.MaskField;

% read images
Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

% ranges
Nr = size(InPar.Range,1);
% values
Nv = numel(InPar.Value);
if (Nv==1),
    InPar.Value = ones(Nr,1).*InPar.Value;
end

% mask bits
InPar.Bit_Replace = get_bitmask_def(InPar.Bit_Replace,'Bit_Replace');

for Iim=1:1:Nim,
    for Ir=1:1:Nr,
        if (isnan(InPar.Range(Ir,1))),
            FlagR = isnan(Sim(Iim).(ImageField));
        else
            FlagR = Sim(Iim).(ImageField)>=InPar.Range(Ir,1) & ...
                    Sim(Iim).(ImageField)<=InPar.Range(Ir,2);
        end
        Sim(Iim).(ImageField)(FlagR) = InPar.Value(Ir);
        
        % Mask image
        if (InPar.CreateMask),
            if (~isfield(Sim(Iim),MaskField)),
                Sim(Iim).(MaskField) = zeros(size(Sim(Iim).(ImageField)),InPar.MaskType);
            else
                if (isempty(Sim(Iim).(MaskField))),
                    Sim(Iim).(MaskField) = zeros(size(Sim(Iim).(ImageField)),InPar.MaskType);
                end
            end
            % update mask
            Sim(Iim).(MaskField)(FlagR) = bitset(Sim(Iim).(MaskField)(FlagR),InPar.Bit_Replace);
            
        end
    end
    
    
    %--- Update header ---
    if (~isfield(Sim(Iim),HeaderField)),
        Sim(Iim).(HeaderField) = [];
    end
    Sim(Iim) = sim_update_head(Sim(Iim),'CopyHeader',InPar.CopyHead,...
                                        'AddHead',InPar.AddHead,...
                                        'DelDataSec',InPar.DelDataSec,...
                                        'Comments',{sprintf('Created by %s.m written by Eran Ofek',FunName)});
                                    
                                    
    
end

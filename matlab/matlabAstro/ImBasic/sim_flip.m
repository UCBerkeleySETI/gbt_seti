function Sim=sim_flip(Sim,varargin)
%--------------------------------------------------------------------------
% sim_flip function                                                ImBasic
% Description: Flip or transpose a set of structure images (SIM).
% Input  : - Set of images to transpose or flip.
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
%            'Op'      - Flip operation function handle
%                        {@transpose|@ctranspose|@fliplr|@flipud}.
%                        Default is @transpose;
%            'FlipIm'  - Flip image field (ImageField) {true|false}.
%                        Default is true.
%            'FlipMask'- Flip mask image field (MaskField) {true|false}.
%                        Default is true.
%            'FlipErrIm'- Flip error image field (ErrImField) {true|false}.
%                        Default is true.
%            'FlipBack' - Flip background image field (BackImField) {true|false}.
%                        Default is true.
%            'ImageField'- Image field in the SIM structure.
%                        Default is 'Im'.
%            'CopyHead' - Copy header from original image {'y' | 'n'}.
%                         Default is 'y'.
%            'AddHead'  - Cell array with 3 columns containing additional
%                         keywords to be add to the header.
%                         See cell_fitshead_addkey.m for header structure
%                         information. Default is empty matrix.
%            'DelDataSec' - Delete the 'DATASEC' header keyword
%                         {true|false}. Default is true.
%            'OutSIM' - Output is a SIM class (true) or a structure
%                      array (false). Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - Structure of flipped images.
%            Note that header information (e.g., NAXIS1/2) is not
%            modified.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=sim_flip(Sim);
% Reliable: 2
%-----------------------------------------------------------------------------
FunName = mfilename;

ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
MaskField   = 'Mask';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';

DefV.Op          = @transpose;   % {@transpose|@fliplr|@flipud}
DefV.FlipIm      = true;
DefV.FlipMask    = true;
DefV.FlipErrIm   = true;
DefV.FlipBack    = true;
DefV.ImageField  = ImageField;
DefV.CopyHead    = 'y';
DefV.AddHead     = [];
DefV.DelDataSec  = true;
DefV.OutSIM      = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

ImageField = InPar.ImageField;



Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);
if (InPar.OutSIM && ~issim(Sim)),
    Sim = struct2sim(Sim); %SIM;    % output is of SIM class
end

for Iim=1:1:Nim,
    
        
    % flip image
    if (isfield(Sim(Iim),ImageField) && InPar.FlipIm),
        Sim(Iim).(ImageField) = InPar.Op(Sim(Iim).(ImageField));
    end
    
    % flip mask
    if (isfield(Sim(Iim),MaskField) && InPar.FlipMask),
        Sim(Iim).(MaskField) = InPar.Op(Sim(Iim).(MaskField));
    end
    
    % flip error image
    if (isfield(Sim(Iim),ErrImField) && InPar.FlipErrIm),
        Sim(Iim).(ErrImField) = InPar.Op(Sim(Iim).(ErrImField));
    end
    
    % flip background image
    if (isfield(Sim(Iim),BackImField) && InPar.FlipBack),
        Sim(Iim).(BackImField) = InPar.Op(Sim(Iim).(BackImField));
    end
    
    
    %--- Update header ---
    if (~isfield(Sim(Iim),HeaderField)),
        Sim(Iim).(HeaderField) = [];
    end
    Sim(Iim) = sim_update_head(Sim(Iim),'CopyHeader',InPar.CopyHead,...
                                        'AddHead',InPar.AddHead,...
                                        'DelDataSec',InPar.DelDataSec,...
                                        'Comments',{sprintf('Created by %s.m written by Eran Ofek',FunName)},...
                                        'History',{sprintf('Flip operation : %s',char(InPar.Op))});
                                    
    
end

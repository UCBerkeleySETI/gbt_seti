function Sim=sim_ufun(Sim,varargin)
%--------------------------------------------------------------------------
% sim_ufun function                                                ImBasic
% Description: Operate a unary function on a set of structure images (SIM).
% Input  : - Set of images to operate the function.
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
%            'Op'      - Operation function handle
%                        e.g., @sin, @cos, @tan, @not,...
%                        Default is @sin.
%            'OpIm'    - Operate on image field (ImageField) {true|false}.
%                        Default is true.
%            'OpMask'  - Operate on mask image field (MaskField) {true|false}.
%                        Default is false.
%            'OpErrIm' - Operate on error image field (ErrImField) {true|false}.
%                        Default is true.
%            'OpBack'  - Operate on background image field (BackImField) {true|false}.
%                        Default is true.
%            'ImageField'- Image field in the SIM structure.
%                        Default is 'Im'.
%            'CopyHead' - Copy header from original image {'y' | 'n'}.
%                        Default is 'y'.
%            'AddHead'  - Cell array with 3 columns containing additional
%                        keywords to be add to the header.
%                        See cell_fitshead_addkey.m for header structure
%                        information. Default is empty matrix.
%            'DelDataSec' - Delete the 'DATASEC' header keyword
%                        {true|false}. Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - Structure of images on which the function was applied.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=sim_ufun(Sim);
% Reliable: 2
%-----------------------------------------------------------------------------
FunName = mfilename;

ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
MaskField   = 'Mask';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';

DefV.Op          = @sin;   
DefV.OpIm        = true;
DefV.OpMask      = false;
DefV.OpErrIm     = true;
DefV.OpBack      = true;
DefV.ImageField  = ImageField;
DefV.CopyHead    = 'y';
DefV.AddHead     = [];
DefV.DelDataSec  = true;

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

ImageField = InPar.ImageField;



Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

for Iim=1:1:Nim,
    % unary function on image
    if (isfield(Sim(Iim),ImageField) && InPar.OpIm),
        Sim(Iim).(ImageField) = InPar.Op(Sim(Iim).(ImageField));
    end
    
    % unary function on mask
    if (isfield(Sim(Iim),MaskField) && InPar.OpMask),
        Sim(Iim).(MaskField) = InPar.Op(Sim(Iim).(MaskField));
    end
    
    % unary function on error image
    if (isfield(Sim(Iim),ErrImField) && InPar.OpErrIm),
        Sim(Iim).(ErrImField) = InPar.Op(Sim(Iim).(ErrImField));
    end
    
    % unary function on background image
    if (isfield(Sim(Iim),BackImField) && InPar.OpBack),
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
                                        'History',{sprintf('Unary function operation : %s',char(InPar.Op))});
                                    
    
end

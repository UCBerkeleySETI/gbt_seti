function Sim=sim_zeropad(Sim,varargin)
%--------------------------------------------------------------------------
% sim_zeropad function                                             ImBasic
% Description: Pad SIM images with zeros.
% Input  : - SIM or FITS images or any input valid for images2sim.m.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'PadSize' - The size of the final padded image [X Y].
%                      This parameter must be provided. Default is [].
%            'PadPos'  - Where to insert the original image in the zero
%                      image.
%                      'centerc'- at the center ceil (default).
%                      'centerf'- at the center floor.
%                      'corner' - at the 1,1 corner.
%                      'shiftedcenter' - at the center, but shifted by half
%                                 the image size.
%                      or [X,Y] position of image corner.
%                      Default is 'centerc'.
%            'PadImage' - Pad image {true|false}. Default is true.
%            'PadMask' - Pad mask {true|false}. Default is true.
%            'PadBack' - Pad background image {true|false}. Default is true.
%            'PadErr'  - Pad error image {true|false}. Default is true.
%            'PadCat'  - Pad catalog coordinates. Not supported yet.
%                        Default is false.
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
%            images2sim.m, image2sim.m
% Output : - Padded SIM images.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Feb 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=sim_zeropad(rand(10,10),'PadSize',[20 20]);
% Reliable: 2
%--------------------------------------------------------------------------
FunName = mfilename;


ImageField      = 'Im';
HeaderField     = 'Header';
%FileField       = 'ImageFileName';
MaskField       = 'Mask';
BackImField     = 'BackIm';
ErrImField      = 'ErrIm';
CatField        = 'Cat';
%CatColField     = 'Col';
%CatColCellField = 'ColCell';


DefV.PadSize          = [];         % [X Y]
DefV.PadPos           = 'centerc';   % 'corner' or position [x y]
DefV.PadImage         = true;
DefV.PadMask          = true;
DefV.PadBack          = true;
DefV.PadErr           = true;
DefV.PadCat           = false;
DefV.CopyHead         = 'y';
DefV.AddHead          = [];
DefV.DelDataSec   = true;
DefV.OutSIM       = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (isempty(InPar.PadSize)),
    error('PadSize option must be provided');
end

Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);
if (InPar.OutSIM && ~issim(Sim)),
    Sim = struct2sim(Sim); %SIM;    % output is of SIM class
end

for Iim=1:1:Nim,
    
    ImSize = size(Sim(Iim).(ImageField));  % [Y X]
    
    % padding position
    if (ischar(InPar.PadPos)),
        switch lower(InPar.PadPos)
            case 'centerf'
                J1 = floor((InPar.PadSize(1) - ImSize(2)).*0.5);
                I1 = floor((InPar.PadSize(2) - ImSize(1)).*0.5);
                Ji = (J1:1:J1+ImSize(2)-1);
                Ii = (I1:1:I1+ImSize(1)-1);
            case 'centerc'
                J1 = ceil((InPar.PadSize(1) - ImSize(2)).*0.5);
                I1 = ceil((InPar.PadSize(2) - ImSize(1)).*0.5);
                Ji = (J1:1:J1+ImSize(2)-1);
                Ii = (I1:1:I1+ImSize(1)-1);
            case 'corner'
                Ii = (1:1:ImSize(1));
                Ji = (1:1:ImSize(2));
            case 'shiftedcenter'
                J1 = floor((InPar.PadSize(1) - ImSize(2)).*0.5) - floor(ImSize(2).*0.5);
                I1 = floor((InPar.PadSize(2) - ImSize(1)).*0.5) - floor(ImSize(1).*0.5);
                Ji = (J1:1:J1+ImSize(2)-1);
                Ii = (I1:1:I1+ImSize(1)-1);
            otherwise
                error('Unknown PadPos option');
        end
    else
        Ii = (InPar.PadPos(2):1:InPar.PadPos(2)+ImSize(1)-1);
        Ji = (InPar.PadPos(1):1:InPar.PadPos(1)+ImSize(2)-1);
    end
                
    Zeros  = zeros(InPar.PadSize(2),InPar.PadSize(1));            
    if (InPar.PadImage && isfield_notempty(Sim(Iim),ImageField)),
        Tmp = Zeros;
        Tmp(Ii,Ji) = Sim(Iim).(ImageField);
        Sim(Iim).(ImageField) = Tmp;
    end
    
    if (InPar.PadMask && isfield_notempty(Sim(Iim),MaskField)),
        Tmp = zeros(InPar.PadSize(2),InPar.PadSize,'like',Sim(Iim).(MaskField));
        Tmp(Ii,Ji) = Sim(Iim).(MaskField);
        Sim(Iim).(MaskField) = Tmp;
    end
    
    if (InPar.PadBack && isfield_notempty(Sim(Iim),BackImField)),
        Tmp = Zeros;
        Tmp(Ii,Ji) = Sim(Iim).(BackImField);
        Sim(Iim).(BackImField) = Tmp;
    end
        
    if (InPar.PadErr && isfield_notempty(Sim(Iim),ErrImField)),
        Tmp = Zeros;
        Tmp(Ii,Ji) = Sim(Iim).(ErrImField);
        Sim(Iim).(ErrImField) = Tmp;
    end
    
    if (InPar.PadCat && isfield_notempty(Sim(Iim),CatField)),
        warning('Cat field is not yet updated');
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

function Sim=sim_class(Sim,OutClass,varargin)
%--------------------------------------------------------------------------
% sim_class function                                               ImBasic
% Description: Convert SIM image class to another class.
% Input  : - SIM or FITS files. See images2sim.m for options.
%          - Output class (e.g., @double, @single, @unit16).
%            Default is @double.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'ConvertImage'  - Convert ImageField. Default is true.
%            'ConvertMask'   - Convert MaskField. Default is false.
%            'ConvertBack'   - Convert BackImField. Default is true.
%            'ConvertErr'    - Convert ErrImField. Default is true.
%            'ConvertWeight' - Convert WeightImField. Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            image2sim.m, images2sim.m
% Output : - SIM in which the images are converted to the requested class.
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    May 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=sim_class(Sim);
% Reliable: 2
%--------------------------------------------------------------------------

Def.OutClass = @double;
if (nargin==1),
    OutClass   = Def.OutClass;
end
if (isempty(OutClass)),
    OutClass   = Def.OutClass;
end

ImageField     = 'Im';
%HeaderField    = 'Header';
%FileField      = 'ImageFileName';
MaskField       = 'Mask';
BackImField     = 'BackIm';
ErrImField      = 'ErrIm';
WeightImField   = 'WeightIm';
%CatField        = 'Cat';
%CatColField     = 'Col';
%CatColCellField = 'ColCell';



DefV.ConvertImage      = true;
DefV.ConvertMask       = false;
DefV.ConvertBack       = true;
DefV.ConvertErr        = true;
DefV.ConvertWeight     = true;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

% for each image
for Iim=1:1:Nim,
    % convert to OutClass
    if (InPar.ConvertImage && isfield_notempty(Sim,ImageField)),
        Sim(Iim).(ImageField) = OutClass(Sim(Iim).(ImageField));
    end
    if (InPar.ConvertMask && isfield_notempty(Sim,MaskField)),
        Sim(Iim).(MaskField) = OutClass(Sim(Iim).(MaskField));
    end
    if (InPar.ConvertBack && isfield_notempty(Sim,BackImField)),
        Sim(Iim).(BackImField) = OutClass(Sim(Iim).(BackImField));
    end
    if (InPar.ConvertErr && isfield_notempty(Sim,ErrImField)),
        Sim(Iim).(ErrImField) = OutClass(Sim(Iim).(ErrImField));
    end
    if (InPar.ConvertWeight && isfield_notempty(Sim,WeightImField)),
        Sim(Iim).(WeightImField) = OutClass(Sim(Iim).(WeightImField));
    end
end

    
    
    
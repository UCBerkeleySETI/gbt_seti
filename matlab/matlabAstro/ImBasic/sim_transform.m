function Sim=sim_transform(Sim,varargin)
%--------------------------------------------------------------------------
% sim_transform function                                           ImBasic
% Description: Aplay a spatial transformation to a list of SIM images.
% Input  : - Structure array images or any of the input supported by
%            images2sim.m.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'Trans' - Spatial transformation. Must be specified.
%                      This can be either a single transformation, or
%                      a structure array of transformations per image.
%                      The transformation is any valid type acceptable
%                      by imtransform.m or imwarp.m. See for example:
%                      maketform.m, affine2d.m,...
%            'OutSize' - Output image size [Y,X]. Must be specified.
%            'InterpIm' - Interpolation method. Either a resampler
%                      structure (see resampler.m), or a cell array to
%                      pass to resampler.m - to work with imtransform.m.
%                      For imwarp this must be a string.
%                      Default is 'blinear'.
%            'FillVal' - Value to fill outside image bounds.
%                      Default is 0.
%            'TransMethod' - program to use in image transformation
%                      {'imwarp'|'imtransform'}. Default is 'imwarp'.
%            'DeleteCat' - Delete the content of the Cat field in the SIM
%                      after the alignment {true|false}. Default is true.
%            'TransIm' - Transform image {true|false}.
%                      Default is true.
%            'TransMask' - Transform mask image (using nearest
%                      interpolation). {true|false}. Default is true.
%            'TransBack' - Transform background image {true|false}.
%                      Default is true.
%            'TransErr' - Transform error image {true|false}.
%                      Default is true.
%            'TransWeight' - Transform weight image {true|false}.
%                      Default is true.
%            'OutSIM' - Force output to be a SIM array {true|false}.
%                      Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs: images2sim.m, image2sim.m
% Output : - SIM structure array with transformed images.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [~,T]=coo_trans([1,1],true,'Shift',[0 1],'PolyX',[5 2 1 500 500 1000 1000])
%          Fun   = @(XY,T) coo_trans(XY,true,T.tdata);
%          Trans = maketform('custom', 2, 2, [], Fun, T);
%          Sim=sim_transform(Sim,'Trans',Trans,'OutSize',[4096 2048],'TransMethod','imtransform');
% Reliable: 2
%--------------------------------------------------------------------------


ImageField     = 'Im';
%HeaderField    = 'Header';
%FileField      = 'ImageFileName';
MaskField       = 'Mask';
BackImField     = 'BackIm';
ErrImField      = 'ErrIm';
WeightImField   = 'WeightIm';
CatField        = 'Cat';
CatColField     = 'Col';
CatColCellField = 'ColCell';


DefV.Trans            = [];
DefV.OutSize          = [];
DefV.InterpIm         = 'bilinear';  % or resampler structure or cell to pass to: makeresampler.m
DefV.FillVal          = 0;
DefV.TransMethod      = 'imwarp';   % {'imwarp'|'imtransform'}
DefV.DeleteCat        = true;
DefV.TransIm          = true;
DefV.TransMask        = true;
DefV.TransBack        = true;
DefV.TransErr         = true;
DefV.TransWeight      = true;
DefV.OutSIM           = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (isempty(InPar.Trans)),
    error('Trans must provided');
end
Ntrans = numel(InPar.Trans);

if (isempty(InPar.OutSize)),
    error('OutSize must provide');
end

% read images
Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);
if (InPar.OutSIM && ~issim(Sim)),
    Sim = struct2sim(Sim); %SIM;    % output is of SIM class
end

% interpolation kernel
if (iscell(InPar.InterpIm)),
    InterpIm = makeresampler(InPar.InterpIm{:});
else
    % char (interpolation method) or structure (resampler)
    InterpIm = InPar.InterpIm;
end


switch lower(InPar.TransMethod)
    case 'imtransform'
        warning('imtransform option wasnt tested yet');
        TrPars   = {'XData',[1 InPar.OutSize(2)],'YData',[1 InPar.OutSize(1)]};
        TrFun    = @imtransform;      
    case 'imwarp'
        Ref2d  = imref2d(InPar.OutSize);
        TrPars = {'OutputView',Ref2d,'FillValues',InPar.FillVal};
        TrFun  = @imwarp;    
    otherwise
        error('Unknown TransMethod option');
end
    
for Iim=1:1:Nim,
    Itrans = min(Ntrans,Iim);  % index of transformation to use
    
    if (InPar.TransIm),
        Sim(Iim).(ImageField) = TrFun(Sim(Iim).(ImageField),InPar.Trans(Itrans),InterpIm,TrPars{:});
    end
    
    if (InPar.TransMask && isfield_notempty(Sim(Iim),MaskField)),
        Sim(Iim).(MaskField) = TrFun(Sim(Iim).(MaskField),InPar.Trans(Itrans),'nearest',TrPars{:});
    end
    if (InPar.TransBack && isfield_notempty(Sim(Iim),BackImField)),
        Sim(Iim).(BackImField) = TrFun(Sim(Iim).(BackImField),InPar.Trans(Itrans),InterpIm,TrPars{:});
    end
    if (InPar.TransErr && isfield_notempty(Sim(Iim),ErrImField)),
        Sim(Iim).(ErrImField) = TrFun(Sim(Iim).(ErrImField),InPar.Trans(Itrans),InterpIm,TrPars{:});
    end
    if (InPar.TransWeight && isfield_notempty(Sim(Iim),WeightImField)),
        Sim(Iim).(WeightImField) = TrFun(Sim(Iim).(WeightImField),InPar.Trans(Itrans),InterpIm,TrPars{:});
    end
    
    % delete the Cat field content
    if (InPar.DeleteCat),
        Sim(Iim).(CatField)        = [];
        Sim(Iim).(CatColField)     = [];
        Sim(Iim).(CatColCellField) = [];
    end
end
    

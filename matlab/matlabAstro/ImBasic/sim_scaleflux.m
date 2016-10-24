function [Sim,Scale]=sim_scaleflux(Sim,varargin)
%--------------------------------------------------------------------------
% sim_scaleflux function                                           ImBasic
% Description: Scale the flux of SIM images.
% Input  : - Input images can be FITS, SIM or other types of
%            images. For input options see images2sim.m.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'BackMethod' - Subtract background prior to scaling.
%                           Default is none. See sim_background.m
%                           for options.
%            'ScaleMethod'- Scale flux method:
%                      'mode_fit' - divide each image by its
%                                   fitted mode. Default.
%                      'none'     - do not scale images.
%                      'median'   - divide each image by its median.
%                      'mean'     - divide each image by its mean.
%                      'std'      - divide each image by its std.
%                      'var'      - divide each image by its variance.
%                      'const'    - multiply each image by a constant.
%            'ScaleConst' - Constant to use (per image) if Scale=const.
%                      Default is 1.
%            'ScaleImage' - Scale the image field of the SIM.
%                      Default is true.
%            'ScaleBack'  - Scale the background field of the SIM.
%                      Default is false.
%            'ScaleErr'   - Scale the error field of the SIM.
%                      Default is false.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m, image2sim.m, sim_background.m, mode_fit.m
% Output : - SIM of scaled images.
%          - Vector of scales by which each image was divided.
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Apr 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Sim,Scale]=sim_scaleflux('*.fits');
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
%HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';


DefV.BackMethod        = 'none';
DefV.ScaleMethod       = 'mode_fit';
DefV.ScaleConst        = 1;
DefV.ScaleImage        = true;
DefV.ScaleBack         = false;
DefV.ScaleErr          = false;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% read images
Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

InPar.ScaleConst = InPar.ScaleConst.*ones(Nim,1);

% subtract background
switch lower(InPar.BackMethod)
    case 'none'
        % do nothing
    otherwise
        Sim = sim_background(Sim,varargin{:},'BackMethod',InPar.BackMethod);
end

% scale images flux
Scale = zeros(Nim,1);
for Iim=1:1:Nim,
    switch lower(InPar.ScaleMethod)
        case 'mode_fit'
            Scale(Iim) = 1./mode_fit(Sim(Iim).(ImageField));
        case 'none'
            Scale(Iim) = 1;
        case 'median'
            Scale(Iim) = 1./nanmedian(Sim(Iim).(ImageField)(:));
        case 'mean'
            Scale(Iim) = 1./nanmean(Sim(Iim).(ImageField)(:));
        case 'std'
            Scale(Iim) = 1./nanstd(Sim(Iim).(ImageField)(:));
        case 'var'
            Scale(Iim) = 1./nanvar(Sim(Iim).(ImageField)(:));
        case {'const','constant'}
            Scale(Iim) = InPar.ScaleConst(Iim);
        otherwise
            error('Unknown ScaleMethod option');
    end
    
    if (InPar.ScaleImage),
        Sim(Iim).(ImageField) = Sim(Iim).(ImageField).* Scale(Iim);
    end
    if (InPar.ScaleBack),
        Sim(Iim).(BackImField) = Sim(Iim).(BackImField).* Scale(Iim);
    end
    if (InPar.ScaleErr),
        Sim(Iim).(ErrImField) = Sim(Iim).(ErrImField).* Scale(Iim);
    end
    
    
end

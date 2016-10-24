function SubImage=sim_get_stamp(Sim,X,Y,varargin)
%--------------------------------------------------------------------------
% sim_get_stamp function                                           ImBasic
% Description: Get image stamps around locations from a set of images.
% Input  : - SIM image name, see images2sim.m for additional options.
%          - X coordinates of center of stamp.
%            This is either a scalar value (to use in all SIMs) or
%            a vector (one element per SIM).
%          - Y coordinates of center of stamp.
%            This is either a scalar value (to use in all SIMs) or
%            a vector (one element per SIM).
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'CooType'   - Coordinates type {'image','J2000'}.
%                          Default is 'image'.
%            'ImageField' - Image field in the SIM. Default is 'Im'.
%            'XHW'        - X half width of stamp in pixels.
%            'YHW'        - Y half width of stamp in pixels.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs: images2sim.m, image2sim.m
% Output : - Structure array of stamp images containing the following
%            fields:
%            .Im      - Stamp image
%            .Offset  - Offset of stamp coordinates relative to original
%                       image.
%            .ErrCL   - 1,2,3 sigma percentiles of stamp.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Nov 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: SubImage=sim_get_stamp(Sim,X,Y);
% Reliable: 2
%--------------------------------------------------------------------------

ImageField  = 'Im';
%HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';


IRAD = pi./180;
%RAD = 180./pi;

DefV.CooType    = 'image';  % {'image','J2000'}
DefV.ImageField = ImageField;
DefV.XHW        = 20;
DefV.YHW        = 20;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


Sim  = images2sim(Sim,varargin{:});
Nsim = numel(Sim);

switch lower(InPar.CooType)
    case 'j2000'
        % convert J2000 Eq. coordinates to X/Y using WCS
        error('unsupported option');
        %[X,Y] = sky2xy_tan(Sim,X.*IRAD,Y.*IRAD);
    case 'image'
        % do nothing
    otherwise
        error('Unknown CooType option');
end

for Isim=1:1:Nsim,
    [SubImage(Isim).(ImageField),SubImage(Isim).Offset] = cut_image(Sim(Isim).(InPar.ImageField),[round(X) round(Y) InPar.XHW InPar.YHW],'Center');
    % gray levels properties
    SubImage(Isim).ErrCL = err_cl(SubImage(Isim).(ImageField)(:));
end
    
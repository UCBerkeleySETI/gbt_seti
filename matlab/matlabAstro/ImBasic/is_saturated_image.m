function SatPix=is_saturated_image(Sim,varargin)
%--------------------------------------------------------------------------
% is_saturated_image function                                      ImBasic
% Description: Count how many saturated pixels are in each image.
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
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - An array with the number of saturated pixels in each image.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: SatPix=is_saturated_image('*.fits');
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

DefV.SatLevel     = 60000;
DefV.SameSatLevel = false;

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

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
     
    SatPix(Iim) = numel(find(Sim(Iim).(ImageField)>SatLevel));
end

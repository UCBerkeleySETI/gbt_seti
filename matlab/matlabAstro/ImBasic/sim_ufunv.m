function Val=sim_ufunv(Sim,varargin)
%--------------------------------------------------------------------------
% sim_ufunv function                                               ImBasic
% Description: Operate a unary function that operate on all the elements
%              of an image and return a scalar (e.g., @mean). The function
%              are applied to a set of structure images (SIM).
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
%                        e.g., @mean, @median, @sum,...
%                        Default is @sum.
%                        To use functions like @nanstd, use @std and
%                        set NaN to true.
%            'NaN'     - {true|false} - Ignore NaN. Default is true.
%                        If true than this function ignore the NaNs,
%                        e.g., mean will act like nanmean.
%            'Par'     - Extra parameter to pass to the function.
%                        Default is {};
%            'ImageField'- Field in the SIM structure on which to operate
%                        the function. Default is 'Im'.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - An array in which each element contains the scalar value 
%            returned by operating the function on the structure array
%            image.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Val=sim_ufunv(Sim,'Op',@nanmedian);
% Reliable: 2
%-----------------------------------------------------------------------------
FunName = mfilename;

ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
MaskField   = 'Mask';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';

DefV.Op          = @sum;   
DefV.NaN         = true;
DefV.Par         = {};
DefV.ImageField  = ImageField;

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

ImageField = InPar.ImageField;



Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);
Val = zeros(size(Sim)).*NaN;

for Iim=1:1:Nim,
    % unary function on image
    if (isfield(Sim(Iim),ImageField)),
        if (InPar.NaN),
            Val(Iim) = InPar.Op( Sim(Iim).(ImageField)(~isnan(Sim(Iim).(ImageField))),InPar.Par{:} );
        else
            Val(Iim) = InPar.Op(Sim(Iim).(ImageField)(:),InPar.Par{:});
        end
    end        
end

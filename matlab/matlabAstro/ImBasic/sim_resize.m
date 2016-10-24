function Sim=sim_resize(Sim,varargin)
%--------------------------------------------------------------------------
% sim_resize function                                              ImBasic
% Description: Resize a set of structure array of images using the
%              imresize.m function. 
% Input  : - Set of images to resize (increase or decrease by magnify
%            or demagnify).
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
%            'Scale'   - Returns an image that is Scale times the
%                        size of the input image (see imresize.m).
%                        Default is 0.5.
%            'TargetSize'- [Nrow, Ncol]. Resizes the image so that it has
%                        the specified number of rows and columns. 
%                        If Nrow or Ncol are NaNs then computes the number
%                        of rows or columns automatically in order to
%                        preserve the image aspect ratio (see imresize.m).
%                        Default is empty. If empty will use the 'Scale'
%                        parameter. If TargetSize is given then this
%                        will override the 'Scale' parameter.
%            'ResMethod'- Interpolation method or interepolation kernel
%                        Options are: {'nearest','bilinear','bicubic',
%                        'box','triangle','cubic','lanczos2','lanczos3'}
%                        or  two-element cell array of the form {f,w},
%                        where f is the function handle for a custom
%                        interpolation kernel, and w is the custom
%                        kernel's width (see imresize.m).
%                        Default is 'cubic'.
%            'ResIm'   - Resize image field (ImageField) {true|false}.
%                        Default is true.
%            'ResMask' - Resize mask image field (MaskField) {true|false}.
%                        Default is true. The mask field will always
%                        be resized using the 'nearest' method.
%            'ResErrIm'- Resize error image field (ErrImField) {true|false}.
%                        Default is true.
%            'ResBack' - Resize background image field (BackImField) {true|false}.
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
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - Structure of resized images.
%            Note that header information (e.g., NAXIS1/2) is not
%            modified.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=sim_resize(Sim,'Scale',0.7);
% Reliable: 2
%-----------------------------------------------------------------------------
FunName = mfilename;



ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
MaskField   = 'Mask';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';

DefV.Scale       = 0.5;
DefV.TargetSize  = [];
DefV.ResMethod   = 'cubic';
DefV.ResIm       = true;
DefV.ResMask     = true;
DefV.ResErrIm    = true;
DefV.ResBack     = true;
DefV.ImageField  = ImageField;
DefV.CopyHead    = 'y';
DefV.AddHead     = [];
DefV.DelDataSec  = true;

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

if (isempty(InPar.TargetSize)),
    ReSize = InPar.Scale;
else
    ReSize = InPar.TargetSize;
end


ImageField = InPar.ImageField;

Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

for Iim=1:1:Nim,
    
        
    % resize image
    if (isfield(Sim(Iim),ImageField) && InPar.ResIm),
        Sim(Iim).(ImageField) = imresize(Sim(Iim).(ImageField),ReSize,InPar.ResMethod);
    end
    
    % flip mask
    if (isfield(Sim(Iim),MaskField) && InPar.ResMask),
        Sim(Iim).(MaskField) = imresize(Sim(Iim).(MaskField),ReSize,'nearest');
    end
    
    % flip error image
    if (isfield(Sim(Iim),ErrImField) && InPar.ResErrIm),
        Sim(Iim).(ErrImField) = imresize(Sim(Iim).(ErrImField),ReSize,InPar.ResMethod);
    end
    
    % flip background image
    if (isfield(Sim(Iim),BackImField) && InPar.ResBack),
        Sim(Iim).(BackImField) = imresize(Sim(Iim).(BackImField),ReSize,InPar.ResMethod);
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

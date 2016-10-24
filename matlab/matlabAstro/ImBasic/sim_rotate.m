function [Sim]=sim_rotate(Sim,varargin)
%--------------------------------------------------------------------------
% sim_rotate function                                              ImBasic
% Description: Rotate a set of structure images (SIM).
% Input  : - Set of images to rotate.
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
%            'Rotation' - Rotation angle [deg] measured in the counter
%                         clockwise direction. Default is 0 deg.
%                         Rotation can be scalar or vector, if vector than
%                         each image will be rotated by the amount
%                         specified in the corresponding element.
%            'RotMethod' - Rotation interpolation method
%                         (see imrotate.m)
%                         {'nearest' | 'bilinear' | 'bicubic'}.
%                         Default is 'bilinear'.
%                         Mask image rotation will always use 'nearest'.
%            'ImageField' - Name of the structure image field that
%                         contains the main image to rotate.
%                         Default is 'Im'.
%            'RotImage' - Rotate image {true|false}. Default is true.
%            'RotMask'  - Rotate mask image {true|false}. Default is true.
%            'RotBack'  - Rotate background image {true|false}.
%                         Default is true.
%            'RotErr'   - Rotate error image {true|false}.
%                         Default is true.
%            'Crop'     - Crop rotated image to the size of the input
%                         image {true|false}, default is false.
%            'CopyHead' - Copy header from original image {'y' | 'n'}.
%                         Default is 'y'.
%            'AddHead'  - Cell array with 3 columns containing additional
%                         keywords to be add to the header.
%                         See cell_fitshead_addkey.m for header structure
%                         information. Default is empty matrix.
%            'DelDataSec' - Delete the 'DATASEC' header keyword
%                         {true|false}. Default is true.
%            'Verbose'  - Print progress messages {true|false}.
%                         Default is false.
%            'OutSIM' - Output is a SIM class (true) or a structure
%                      array (false). Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - Structure of rotated images.
% Tested : Matlab R2013a
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=sim_rotate('lred0127.fits','Rotation',45);
%          Sim=sim_rotate('lred012[6-7].fits','Rotation',[45;30],'Verbose',true);
% Reliable: 2 
%--------------------------------------------------------------------------
FunName = mfilename;

ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
MaskField   = 'Mask';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';

DefV.Rotation     = 0;
DefV.RotMethod    = 'bilinear';
DefV.ImageField   = ImageField;
DefV.RotImage     = true;
DefV.RotMask      = true;
DefV.RotBack      = true;
DefV.RotErr       = true;
DefV.Crop         = false;
DefV.CopyHead     = 'y';
DefV.AddHead      = [];
DefV.DelDataSec   = true;
DefV.Verbose      = false;
DefV.OutSIM       = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

ImageField = InPar.ImageField;


%--- read images ---
Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);
if (InPar.OutSIM && ~issim(Sim)),
    Sim = struct2sim(Sim); %SIM;    % output is of SIM class
end

% prep rotation vector
if (numel(InPar.Rotation)==1),
    InPar.Rotation = InPar.Rotation.*ones(1,Nim);
end

% prep crop
if (InPar.Crop),
    BBox = 'crop';
else
    BBox = 'loose';
end

% verify the status of image fields and requested rotations
if (~isfield(Sim,ImageField)),
    InPar.RotImage = false;
end
if (~isfield(Sim,MaskField)),
    InPar.RotMask = false;
end
if (~isfield(Sim,BackImField)),
    InPar.RotBack = false;
end
if (~isfield(Sim,ErrImField)),
    InPar.RotErr = false;
end
  
%--- Go over images and rotate ---
for Iim=1:1:Nim,
    %--- rotate image ---  
    if (InPar.Verbose),
        fprintf('Rotate image number %d by %f deg\n',Iim,InPar.Rotation(Iim));
    end
        
    
    if (InPar.RotImage),
        Sim(Iim).(ImageField) = imrotate(Sim(Iim).(ImageField),InPar.Rotation(Iim),InPar.RotMethod,BBox);
    end
    if (InPar.RotMask),
        Sim(Iim).(MaskField) = imrotate(Sim(Iim).(MaskField),InPar.Rotation(Iim),'nearest',BBox);
    end
    if (InPar.RotBack),
        Sim(Iim).(BackImField) = imrotate(Sim(Iim).(BackImField),InPar.Rotation(Iim),InPar.RotMethod,BBox);
    end
    if (InPar.RotErr),
        Sim(Iim).(ErrImField) = imrotate(Sim(Iim).(ErrImField),InPar.Rotation(Iim),InPar.RotMethod,BBox);
    end

    
    
    %--- Update header ---
    if (~isfield(Sim(Iim),HeaderField)),
        Sim(Iim).(HeaderField) = [];
    end
    Sim(Iim) = sim_update_head(Sim(Iim),'CopyHeader',InPar.CopyHead,...
                                        'AddHead',InPar.AddHead,...
                                        'DelDataSec',InPar.DelDataSec,...
                                        'Comments',{sprintf('Created by %s.m written by Eran Ofek',FunName)},...
                                        'History',{sprintf('Rotation : %f deg',InPar.Rotation(Iim)),...
                                                   sprintf('Rotation interpolation method: %s',InPar.RotMethod)});
         
end

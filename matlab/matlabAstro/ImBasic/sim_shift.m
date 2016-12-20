function [Sim]=sim_shift(Sim,varargin)
%--------------------------------------------------------------------------
% sim_shift function                                               ImBasic
% Description: Shift in X/Y coordinates a set of structure images (SIM). 
% Input  : - Set of images to shift.
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
%            'Shift' - A two column matrix of [ShiftX, ShiftY] to shift
%                      each of the images. If a single row is provided
%                      then all the images will be shifted by the same
%                      amount.
%            'ShiftImage' - Shift the image {true|false}. Default is true.
%            'ShiftMask' - Shift the mask image {true|false}.
%                      Default is true.
%            'ShiftBack' - Shift the background image {true|false}.
%                      Default is true.
%            'ShiftErr' - Shift the error image {true|false}.
%                      Default is true.
%            'SameSize' - Set the 'MaxShiftX' to max(ShiftX),
%                      'MaxShiftY' to max(ShiftY),
%                      'MinShiftX' to min(ShiftX),
%                      and 'MinShiftY' to min(ShiftY).
%                      In this case, if all the input images have the same
%                      size than also the output images will have the
%                      same size.
%            'MaxShiftX' - Control the size of the output image.
%                      Default is empty. If empty use Shift(1).
%                      This is usefull if several images with different
%                      shift are required to have the same size.
%                      In this case use max(ShiftX).
%            'MaxShiftY' - Like 'MaxShiftX', but for the Y-axis.
%            'MinShiftX' - Control the size of the output image.
%                      Default is empty. If empty then 1.
%                      This is usefull if several images with different
%                      shift are required to have the same size.
%                      In this case use min(ShiftX).
%            'MinShiftY' - Like 'MinShiftX', but for the Y-axis.
%            'TInterpolant' - Transformation interpolant.
%                      See makeresampler.m for options.
%                      Default is 'cubic'.
%                      Mask image rotation will always use 'nearest'.
%            'TPadMethod' - Transformation padding method.
%                      See makeresampler.m for options.
%                      Default is 'bound'.
%            'FillValues' - Value to use in order to fill missing data
%                      points. Default is NaN.
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
%            images2sim.m, image_shift.m
% Output : - Structure of shifted images.
% See also: image_shift.m
% Tested : Matlab R2013a
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Sim]=sim_shift('lred012[6-7].fits','Shift',[100 100;200 10]);
% Reliable: 2
%--------------------------------------------------------------------------
FunName = mfilename;


ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
MaskField   = 'Mask';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';

DefV.Shift        = [0 0];
DefV.TInterpolant = 'cubic';
DefV.ImageField   = ImageField;
DefV.ShiftImage   = true;
DefV.ShiftMask    = true;
DefV.ShiftBack    = true;
DefV.ShiftErr     = true;
DefV.TPadMethod   = 'bound';
DefV.FillValues   = NaN;
DefV.SameSize     = true;
DefV.MaxShiftX    = [];
DefV.MaxShiftY    = [];
DefV.MinShiftX    = [];
DefV.MinShiftY    = [];
DefV.CopyHead     = 'y';
DefV.AddHead      = [];
DefV.DelDataSec   = true;
DefV.Verbose      = false;
DefV.OutSIM      = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

ImageField = InPar.ImageField;


%--- read images ---
Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);
if (InPar.OutSIM && ~issim(Sim)),
    Sim = struct2sim(Sim); %SIM;    % output is of SIM class
end

% prep shift array
if (size(InPar.Shift,2)==1),
    InPar.Shift = repmat(InPar.Shift, Nim,1);
end

% prep shift output size
if (InPar.SameSize),
   InPar.MaxShiftX = max(InPar.Shift(:,1));
   InPar.MaxShiftY = max(InPar.Shift(:,2));
   InPar.MinShiftX = min(InPar.Shift(:,1));
   InPar.MinShiftY = min(InPar.Shift(:,2));
end
    

% prep crop
%if (InPar.Crop),
%    BBox = 'crop';
%else
%    BBox = 'loose';
%end

% verify the status of image fields and requested shifts
if (~isfield(Sim,ImageField)),
    InPar.ShiftImage = false;
end
if (~isfield(Sim,MaskField)),
    InPar.ShiftMask = false;
end
if (~isfield(Sim,BackImField)),
    InPar.ShiftBack = false;
end
if (~isfield(Sim,ErrImField)),
    InPar.ShiftErr = false;
end
  
%--- Go over images and shift ---
for Iim=1:1:Nim,
    %--- shift image ---  
    if (InPar.Verbose),
        fprintf('Shift image number %d by X=%f Y=%f pixels\n',Iim,InPar.Shift(Iim,:));
    end
        
    
    if (InPar.ShiftImage),
        Sim(Iim).(ImageField) = image_shift(Sim(Iim).(ImageField),InPar.Shift(Iim,:),varargin{:},...
                                            'MaxShiftX',InPar.MaxShiftX,...
                                            'MaxShiftY',InPar.MaxShiftY,...
                                            'MinShiftX',InPar.MinShiftX,...
                                            'MinShiftY',InPar.MinShiftY,...
                                            'TPadMethod',InPar.TPadMethod,...
                                            'FillValues',InPar.FillValues);
        
    end
    if (InPar.ShiftMask),
        Sim(Iim).(MaskField) = image_shift(Sim(Iim).(ImageField),InPar.Shift(Iim,:),varargin{:},...
                                            'MaxShiftX',InPar.MaxShiftX,...
                                            'MaxShiftY',InPar.MaxShiftY,...
                                            'MinShiftX',InPar.MinShiftX,...
                                            'MinShiftY',InPar.MinShiftY,...
                                            'TInterpolant','nearest',...
                                            'TPadMethod',InPar.TPadMethod,...
                                            'FillValues',InPar.FillValues);
    end
    if (InPar.ShiftBack),
        Sim(Iim).(BackImField) = image_shift(Sim(Iim).(ImageField),InPar.Shift(Iim,:),varargin{:},...
                                            'MaxShiftX',InPar.MaxShiftX,...
                                            'MaxShiftY',InPar.MaxShiftY,...
                                            'MinShiftX',InPar.MinShiftX,...
                                            'MinShiftY',InPar.MinShiftY,...
                                            'TPadMethod',InPar.TPadMethod,...
                                            'FillValues',InPar.FillValues);
    end
    if (InPar.ShiftErr),
        Sim(Iim).(ErrImField) = image_shift(Sim(Iim).(ImageField),InPar.Shift(Iim,:),varargin{:},...
                                            'MaxShiftX',InPar.MaxShiftX,...
                                            'MaxShiftY',InPar.MaxShiftY,...
                                            'MinShiftX',InPar.MinShiftX,...
                                            'MinShiftY',InPar.MinShiftY,...
                                            'TPadMethod',InPar.TPadMethod,...
                                            'FillValues',InPar.FillValues);
                                        
    end


    
    %--- Update header ---
    if (~isfield(Sim(Iim),HeaderField)),
        Sim(Iim).(HeaderField) = [];
    end
    Sim(Iim) = sim_update_head(Sim(Iim),'CopyHeader',InPar.CopyHead,...
                                        'AddHead',InPar.AddHead,...
                                        'DelDataSec',InPar.DelDataSec,...
                                        'Comments',{sprintf('Created by %s.m written by Eran Ofek',FunName)},...
                                        'History',{sprintf('Shift : X=%f Y=%f pixels',InPar.Shift(Iim,:)),...
                                                   sprintf('Shift interpolation method: %s',InPar.TInterpolant)});
      
end

function Sim=cube2sim(Cube,Sim,varargin)
%--------------------------------------------------------------------------
% cube2sim function                                                ImBasic
% Description: Convert a cube to a structure array of images (SIM).
% Input  : - A 3D cube.
%          - A structure array of images in which to add the images.
%            If empty, then create a new structure array. Default is empty.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'Field' - Field name in the structure array in which to store
%                      each 2D image. Default is 'Im'.
%            'Dim'   - Dimension of the z-axis (image index) in which to
%                      break the images. Default is 3.
% Output : - A structure array of images.
% Output : - SIM structure.
% See also: image2sim.m, sim2file.m, sim2cube.m
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=cube2sim(rand(10,10,5));
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
%HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';


if (nargin==1),
    Sim = [];
end

DefV.Field = ImageField;
DefV.Dim   = 3;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


if (InPar.Dim==3),
    Dims = [1 2 3];
elseif (InPar.Dim==2),
    Dims = [1 3 2];
elseif (InPar.Dim==1),
    Dims = [2 3 1];
else
    error('Unknown Dim option');
end

Cube = permute(Cube,Dims);
Nim  = size(Cube,3);

if (isempty(Sim)),
   Sim = struct(InPar.Field,cell(Nim,1));
end

for Iim=1:1:Nim,
   Sim(Iim).(InPar.Field) = Cube(:,:,Iim);
end

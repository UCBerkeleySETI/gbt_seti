function Mat3D=create_mat3d_images(Input,DataSec,varargin)
%-----------------------------------------------------------------------------
% create_mat3d_images function                                        ImBasic
% Description: Given a list of FITS images, read them into a 3D matrix,
%              in which the third axis corresponds to the image index.
% Input  : - Set of input images.
%            This can be either:
%            A string containing FITS images name (including wild cards -
%            see create_list.m for options).
%            A cell array in which each cell contains a FITS image name.
%            A cell array in which each cell contains a 2D matrix.
%            A 3D matrix, in which the third dimension is the image index.
%            All the images should have the same size.
%          - An image data section [Xmin,Xmax,Ymin,Ymax] by which to crop
%            all images. Default is empty matrix.
%            If empty matrix then do not crop images.
%          * Arbitrary number of additional arguments to pass to
%            fitsread.m
% Output : - A 3D array in which the dimensions are y-axis, x-axis
%            and image-index.
% Tested : Matlab 7.11
%     By : Eran O. Ofek                    Mar 2011
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%-----------------------------------------------------------------------------

Def.DataSec = [];
if (nargin==1),
   DataSec  = Def.DataSec;
end

Fits2Mat = 0;
if (isstr(Input)),
   [~,InputCell] = create_list(Input,NaN);
   Fits2Mat = 1;
elseif (iscell(Input)),
   if (isstr(Input{1})),
      InputCell = Input;
      Fits2Mat  = 1;
   elseif (isnumeric(Input{1})),  
      [Ny, Nx] = size(Input{1});
      Nim      = length(Input);
      Mat3D = reshape([Input{:}],[Ny,Nx,Nim]);
      Mat3D = Mat3D([DataSec(3):DataSec(4)],[DataSec(1):DataSec(2)],:);
   else
      error('Unknwon input option');
   end
elseif (isnumeric(Input)),
   [Ny,Nx,Nim] = size(Input);
   Mat3D = Input;
else
   error('Unknwon input option');
end
clear Input;

switch Fits2Mat
 case 1
    % read fits files
    Nim       = length(InputCell);
    if (isempty(DataSec)),
       % use the entire image
       [Ny,Nx]   = size(fitsread(InputCell{1},varargin{:}));
    else
       % crop image mode
       Ny = DataSec(2)-DataSec(1)+1;
       Nx = DataSec(4)-DataSec(3)+1;
    end
    Mat3D     = zeros(Ny,Nx,Nim);
    for Iim=1:1:Nim,
       CurImage = fitsread(InputCell{Iim},varargin{:});
       if (isempty(DataSec)),
          Mat3D(:,:,Iim) = CurImage;
       else
          Mat3D(:,:,Im) = CurImage([DataSec(3):DataSec(4)],[DataSec(1):DataSec(2)]);
       end
    end
 case 0
    % do nothing
 otherwise
   error('Unknwon Fits2Mat option');
end



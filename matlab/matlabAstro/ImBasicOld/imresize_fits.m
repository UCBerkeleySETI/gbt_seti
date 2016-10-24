function [OutImageFileCell,OutputMatrixCell]=imresize_fits(ImInput,ImOutput,ReSize,varargin);
%-----------------------------------------------------------------------------
% imresize_fits function                                              ImBasic
% Description: Resize a FITS image. This is dones by interpolation or
%              convolution (using an interpolation kernel) so effectively
%              this can be used also to convolve an image.
% Input  : - List of input images (see create_list.m for details).
%            Alternatively, this can be a cell array in which each cell
%            contains the image in a matrix form, or a matrix.
%            In this case ImInput can not be an empty matrix.
%          - List of output images (see create_list.m for details).
%            If empty matrix, then set it to be equal to the Input list.
%            Default is empty matrix.
%          - New image size [[New X dimension], [New Y dimension]]
%            or alternatively a scalar (or a column vector) indicating a
%            scaling factor by which to scale the size of the image.
%            If empty matrix or [NaN NaN] than will set the new dimension
%            to be equal to the original dimension of the image.
%          * Arbitrary number of pairs of input parameters: keyword,value,...
%            Available keywords are:
%            'Method'   - Interpolation method or interpolation kernel.
%                         For detailed options see imresize.m
%                         This can be either one of the following
%                         interpolation methods {'nearest','bilinear','bicubic'}
%                         or one of the following kerenels
%                         {'box','triangle','cubic','lanczos2','lanczos3'}
%                         or {f, w}. f is a function handle for a custom
%                         interpolation kernel and w is the custom kernels width.
%                         f(x) must be zero outside the interval -w/2 <= x < w/2.
%                         Your function handle f may be called with a scalar
%                         or a vector input. 
%                         Default is 'bilinear'.
%            'Antialiasing'- A Boolean value that specifies whether to perform
%                         antialiasing when shrinking an image.
%                         Default is true - If you are using Method='nearest',
%                         it is recomended to set it to false.
%            'OutPrefix'- Add prefix before output image names,
%                         default is empty string (i.e., '').
%            'OutDir'   - Directory in which to write the output images,
%                         default is empty string (i.e., '').
%            'CopyHead' - Copy header from original image {'y' | 'n'}.
%                         Default is 'y'.
%            'AddHead'  - Cell array with 3 columns containing additional
%                         keywords to be add to the header.
%                         See cell_fitshead_addkey.m for header structure
%                         information. Default is empty matrix.
%            'DelDataSec'-Delete DATASEC keyword from image header {'y' | 'n'}.
%                         Default is 'y'.
%                         The reason for that the DATASEC keywords may
%                         caseuse problem in image display.
%            'DataType' - Output data type (see fitswrite.m for options), 
%                         default is float32.
%            'CCDSEC'   - Image sction for image to be rotated. If given
%                         then the image will be croped before rotation.
%                         This could be string containing image keyword 
%                         name (e.g., 'CCDSEC'), or a vector of 
%                         [Xmin, Xmax, Ymin, Ymax].
%             'Save'    - Save FITS image to disk {'y' | 'n'}.
%                         Default is 'y'.
% Output : - Cell array containing output image names.
%          - Cell array of matrices oroutput images.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: Out=imresize_fits('lred0015.fits','try.fits',0.8);
% Reliable: 2
%-----------------------------------------------------------------------------

Def.Output   = [];
if (nargin==1),
   Output   = Def.Output;
else
   % do nothing
end


DefV.Method      = 'bilinear';
DefV.Antialiasing = true;
DefV.OutPrefix   = '';
DefV.OutDir      = '';
DefV.DelDataSec  = 'y';
DefV.CopyHead    = 'y';
DefV.AddHead     = [];
DefV.DataType    = 'float32';
DefV.CCDSEC      = [];
DefV.Save        = 'y';

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

IsNumeric = 0;
if (isnumeric(ImInput)),
   ImInput = {ImInput};
end

if (iscell(ImInput)),
   if (isnumeric(ImInput{1})),
      % input is given in a matrix form
      IsNumeric = 1;
      if (isempty(ImOutput)),
	 error('If ImInput is numeric then must specify ImOutput');
      end
   end
end

if (IsNumeric==0),
   [~,ImInputCell] = create_list(ImInput,NaN);
else
   ImInputCell = ImInput;
end
Nim = length(ImInputCell);


if (isempty(ImOutput)),
   ImOutput = ImInput;
end
[~,ImOutputCell] = create_list(ImOutput,NaN);

if (isempty(ReSize)),
   ReSize = [NaN NaN];
end

if (size(ReSize,2)==2),
   % replace X and Y axes by [I and J]
   ReSize = ReSize(:,[2 1]);
end

if (size(ReSize,1)==1),
   % set ReSize for each image
   ReSize = ones(Nim,1)*ReSize;
end


%--- Go over all images ---
for Iim=1:1:Nim,
   %--- read Image ImInput ---
   if (IsNumeric==1),
      InputImage = ImInput{Iim};
   else
      InputImage = fitsread(ImInputCell{Iim});
   end

   %--- CCDSEC ---
   if (isempty(InPar.CCDSEC)),
      % use entire image
      % do nothing
   elseif (ischar(InPar.CCDSEC)),
      [InPar.CCDSEC] = get_ccdsec_fits({ImInputCell{Iim}},InPar.CCDSEC);
      [InputImage]   = cut_image(InputImage,InPar.CCDSEC,'boundry');
   elseif (length(InPar.CCDSEC)==4),
      [InputImage]   = cut_image(InputImage,InPar.CCDSEC,'boundry');
   else
      error('Illegal CCDSEC input');
   end

   OrigSize = size(InputImage);  % original size after CCDSEC

   %--- resize image ---
   if (sum(isnan(ReSize(Iim,:)))>0),
      % keep original size (after CCDSEC)
      if (size(ReSize,2)==2),
         ReSize(Iim,:) = OrigSize;
      elseif (size(ReSize,2)==1),
         ReSize(Iim,:) = 1;
      else
	 error('ReSize has illegal size');
      end
   end

   InputImage = imresize(InputImage,ReSize(Iim,:),InPar.Method,'Antialiasing',InPar.Antialiasing);

   NewSize = size(InputImage);  % size of new image


   OutImageFileName = sprintf('%s%s%s',InPar.OutDir,InPar.OutPrefix,ImOutputCell{Iim});
   OutImageFileCell{Iim} = OutImageFileName;

   if (IsNumeric==0),
      switch lower(InPar.CopyHead)
       case 'y'
          Info = fitsinfo(ImInputCell{Iim});
          HeaderInfo = Info.PrimaryData.Keywords;
       otherwise
          HeaderInfo = [];
      end
   else
      HeaderInfo = [];
   end

   if (IsNumeric==1),
      InputName = 'matlab Matrix format';
   else
      InputName = ImInputCell{Iim};
   end

   if (iscell(InPar.Method)),
      ReSizeMethod = 'custom';
   else
      ReSizeMethod = InPar.Method;
   end

   %--- Add to header comments regarding file creation ---
   [HeaderInfo] = cell_fitshead_addkey(HeaderInfo,...
                                       Inf,'COMMENT','','Created by imresize_fits.m written by Eran Ofek',...
                                       Inf,'HISTORY','',sprintf('ReSize Method: %s',ReSizeMethod),...
                                       Inf,'HISTORY','',sprintf('Original size: %d,%d',OrigSize([2 1])),...
    			               Inf,'HISTORY','',sprintf('New size: %d,%d',NewSize([2 1])),...
                                       Inf,'HISTORY','',sprintf('Input image name: %s',InputName));

   if (~isempty(InPar.AddHead)),
      %--- Add additional header keywords ---
      HeaderInfo = [HeaderInfo; InPar.AddHead];
   end

   switch lower(InPar.DelDataSec)
    case 'n'
        % do nothing
    case 'y'
        % delete DATASEC keyword from header
        [HeaderInfo] = cell_fitshead_delkey(HeaderInfo,'DATASEC');
    otherwise
        error('Unknown DelDataSec option');
   end

   %--- Write fits file ---
   switch lower(InPar.Save)
    case 'y'
       fitswrite_my(InputImage,OutImageFileName,HeaderInfo,InPar.DataType);
    case 'n'
       % do not save FITS image
    otherwise
       error('Unknown Save option');
   end

   if (nargout>1),
      OutputMatrixCell{Iim} = InputImage;
   end

end



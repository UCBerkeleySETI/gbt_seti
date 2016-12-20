function [OutImageFileCell,OutputMatrixCell]=imconv_fits(ImInput,ImOutput,Kernel,varargin);
%-----------------------------------------------------------------------------
% imconv_fits function                                                ImBasic
% Description: Convolve a FITS image with a kernel.
% Input  : - List of input images (see create_list.m for details).
%            Alternatively, this can be a cell array in which each cell
%            contains the image in a matrix form, or a matrix.
%            In this case ImPutput can not be an empty matrix.
%          - List of output images (see create_list.m for details).
%            If empty matrix, then set it to be equal to the Input list.
%            Default is empty matrix.
%          - Convolution kerenl. This can be (i) a matrix;
%            or (ii) a cell array in which the first cell is an
%            handle to a function that return the kernel, the second cell
%            is the semi-width of the kerenel, and the other
%            (optional) elements are arguments of the function.
%            Alternatively one can pass the additional parameters
%            using the 'AddPar' keyword (see below).
%            The functions should be called like Z=fun(X,Y,additional_par);
%            A list of built in functions:
%            @gauss_2d(X,Y,[SigmaX,SigmaY],Rho,[X0,Y0], MaxRadius,Norm)
%                  by default SigmaY=SigmaX, Rho=0, X0=0, Y0=0.
%                  MaxRadius is an optional parameter that set the kernel
%                  to zero outside the specified radius. Default is Inf.
%                  Norm is a volume normalization constant of the final matrix.
%                  If NaN then donot normalize. Default is 1.
%            @triangle_2d(X,Y,Base,[X0, Y0],Norm)
%                  A conic (triangle) convolution kernel.
%                  Base - its semi width
%                  [X0, Y0] - center, default is [0 0].
%                  Norm - Volume normalization, default is 1.
%            @box_2d(X,Y,SemiWidth,[X0 Y0],Norm)
%                  A box (square) convolution kernel.
%                  SemiWidth - The box semi width. If two elements are
%                              specified then these are the semi width
%                              in the X and Y direction, respectively.
%                  [X0, Y0] - center, default is [0 0].
%                  Norm - Volume normalization, default is 1.
%            @circ_2d(X,Y,Radius,[X0 Y0],Norm)
%                  A circle (cylinder) convolution kernel.
%                  Radius - The circle radius.
%                  [X0, Y0] - center, default is [0 0].
%                  Norm - Volume normalization, default is 1.
%            @lanczos_2d(X,Y,A,Stretch,[X0,Y0],Norm)
%                  A lanczos convolution kernel.
%                  A   - Order. Default is 2.
%                  Stretch - Stretch factor. Default is 1.
%                  [X0, Y0] - center, default is [0 0].
%                  Norm - Volume normalization, default is 1.
%          * Arbitrary number of pairs of input parameters: keyword,value,...
%            Available keywords are:
%            'Size'     - A string specifying the size of the output image.
%                         This can be one of the following strings:
%                         'same' - returns the central part of the convolution
%                                  that is the same size as the input (default).
%                         'valid'- returns only those parts of the correlation
%                                  that are computed without the zero-padded edges.
%                         'full' - returns the full convolution.
%            'AddPar'   - Cell array of additional parameters to pass to
%                         the convolution function. Default is empty cell {}.
%                         This can come instead or in addition to the
%                         additional parameter cell array (see above).
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
%                         Default is 'n'.
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
% Output  : - Cell array containing output image names.
%           - Cell array of matrices containing the output images
%             in matrix form.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: % convolve a set of images with a Gaussian with Sigma= 3 pixels
%          % and a total semi-width of 10 pixels:
%          Out=imconv_fits('ccd.0[42-56].0.fits','try.fits',{@gauss_2d,10,3});
%          % convolve a matrix with a random kernel:
%          Out=imconv_fits(rand(100,100),'try.fits',rand(10,10));
%          % convolve with a triangle:
%          Out=imconv_fits('ccd.042.0.fits','try.fits',{@triangle_2d,10,5});
%          % or alternatively:
%          Out=imconv_fits('ccd.042.0.fits','try.fits',{@triangle_2d,10},'AddPar',{5});
%          % Another example, with no parameters passed to the function:
%          Out=imconv_fits('ccd.042.0.fits','try.fits',{@lanczos_2d,10});
% Reliable: 2
%-----------------------------------------------------------------------------

Def.Output   = [];
if (nargin==1),
   Output   = Def.Output;
else
   % do nothing
end


DefV.Size        = 'same';
DefV.AddPar      = {};
DefV.OutPrefix   = '';
DefV.OutDir      = '';
DefV.DelDataSec  = 'n';
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

%--- prepare the Kernel ---
if (isnumeric(Kernel)),
   ConvKernel = Kernel;
else
   if (iscell(Kernel)),
      KernelFun   = Kernel{1};
      KernelWidth = Kernel{2};
      KernelPar   = {Kernel{3:end},InPar.AddPar{:}};
      [MatX,MatY] = meshgrid([-KernelWidth:1:KernelWidth],[-KernelWidth:1:KernelWidth]);
      ConvKernel  = feval(KernelFun,MatX,MatY,KernelPar{:});
   else
     error('Unknown Kernel type');
   end
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

   %--- convolve image ---
   InputImage = conv2(InputImage,ConvKernel,InPar.Size);

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

   KernelSize = size(ConvKernel);

   %--- Add to header comments regarding file creation ---
   [HeaderInfo] = cell_fitshead_addkey(HeaderInfo,...
                                       Inf,'COMMENT','','Created by imconv_fits.m written by Eran Ofek',...
                                       Inf,'HISTORY','',sprintf('Size Method: %s',InPar.Size),...
                                       Inf,'HISTORY','',sprintf('Original size: %d,%d',OrigSize([2 1])),...
    			               Inf,'HISTORY','',sprintf('New size: %d,%d',NewSize([2 1])),...
    			               Inf,'HISTORY','',sprintf('Convolution kernel size: %d,%d',KernelSize([2 1])),...
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
       fitswrite(InputImage,OutImageFileName,HeaderInfo,InPar.DataType);
    case 'n'
       % do not save FITS image
    otherwise
       error('Unknown Save option');
   end

   if (nargout>1),
      OutputMatrixCell{Iim} = OutputImage;
   end

end



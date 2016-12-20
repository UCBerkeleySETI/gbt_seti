function [OutImageFileCell,OutputMatrixCell,BackMatrixCell]=imsubback_fits(ImInput,ImOutput,Method,varargin)
%------------------------------------------------------------------------------
% imsubback_fits function                                              ImBasic
% Description: Subtract background from a 2-D matrix or a FITS image
%              using various nethods.
% Input  : - List of FITS images (see create_list.m for details) or list
%            of 2-D matrices from which to subtract a background.
%          - List of output images (see create_list.m for details).
%            If empty matrix, then set it to be equal to the first
%            Input list. Default is empty matrix.
%            In order to avoid saving the oupput as a FITS image, use
%            the {'Save','n'} option (see below).
%          - Method of background subtraction:
%            'mode'   - Subtract the mode of the image (default).
%            'median' - Subtract the median of the image.
%            'mean'   - Subtract the mean of the image.
%            'medfilt'- Calculate a median filter of the image and
%                       subtract it from the image. The box size of the
%                       median filter is set by the 'SubPar' argument
%                       (see below). 
%            'ordfilt'- replaces each element in the input image by the
%                       K-th element in a "box" centered on the pixel
%                       (see ordfilt2.m for details).
%                       Default for SubPar arguments are:
%                       {9,ones(3,3)} which is a 3x3 minimum filter.
%            'poly'   - Fit a 2-D polynomial surface to the image and
%                       subtract it from the image. Polynomial
%                       parameter are set bythe 'SubPar' argument
%                       (see below).
%            Alternatively this can be an handle to a function that
%            estimate the background of the image:
%               Background = Fun(Image,additional_par);
%               Where Background is a background matrix or scalar,
%               Image is the original image in matrix form and
%               additional_par can be passed using the 'AddPar'
%               argument (see below).
%          * Arbitrary number of pairs of input parameters: keyword,value,...
%            Available keywords are:
%            'MaskStar' - Remove stars from image using maskstars_fits.m
%                         before estimating the background.
%                         Options are:
%                         []     - do not mask stars (default).
%                         'sex'  - Find stars using the default settings of
%                                  SExtractor
%                                  (see run_sextractor.m for details).
%                         'perc' - Mask pixels which value is found a the
%                                  top percentile of the pixels in an image.
%                         'val'  - Mask pixels which value is above a given
%                                  value.
%                         Alternatively one can pass additional parameters
%                         to maskstars_fits.m by putting the method in a cell
%                         array along with additional parameters
%                         e.g., {'sex','DETECT_THRESH',1.5}
%            'SubPar'   - Parameters for background subtraction
%                         algorithm. If 'medfilt', then parameters
%                         are [SizeX, SizeY]. Default is [100 100].
%                         If 'poly', then parameters are
%                         [DegX, DegY, DegXY] or
%                         {[DegX, DegY, DegXY],'SigClip',SigClip,..
%                          'MaxNiter',MaxNiter,'FunType',FunType,'Method',Method}.
%                         Default is {[3 3 2],'SigClip',[3.5 3.5],...
%                                     'MaxNiter',1,'FunType','poly','Method','chol'}
%                         See polysurface_fit.m for details.
%                         If Method is a function handle than this
%                         is a cell array of additional arguments
%                         to pass to the function, default is {}.
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
%           - Cell array of matrices of output images.
%           - Cell array of matrices of background images.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                    August 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [OutFiles,OutMat]=imsubback_fits('ccd.045.0.fits','out.fits','medfilt');
%          [OutFiles,OutMat]=imsubback_fits('ccd.045.0.fits','out.fits','mode');
%          [OutFiles,OutMat]=imsubback_fits('ccd.045.0.fits','out.fits','mode','MaskStar','sex');
%          [OutFiles,OutMat]=imsubback_fits('ccd.045.0.fits','out.fits','poly','SubPar',[2 2 1],'MaskStar','sex');
% Reliable: 2
%------------------------------------------------------------------------------

Def.ImOutput   = [];
Def.Method     = 'mode';
if (nargin==1),
   ImOutput   = Def.ImOutput;
   Method     = Def.Method;
elseif (nargin==2),
   Method     = Def.Method;
else
   % do nothing
end

DefV.Save        = 'y';
DefV.MaskStar    = [];
DefV.SubPar      = {};
DefV.OutPrefix   = '';
DefV.OutDir      = '';
DefV.DelDataSec  = 'n';
DefV.CopyHead    = 'y';
DefV.AddHead     = [];
DefV.DataType    = 'float32';
DefV.CCDSEC      = [];
DefV.Save        = 'y';
InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

% set default for SubPar in special cases:
switch lower(Method)
 case 'medfilt'
    if (isnumeric(InPar.SubPar)),
       % do nothing - SubPar is already a numeric vector
    else
       InPar.SubPar = [100 100];
    end
    if (length(InPar.SubPar)==1),
       InPar.SubPar = [InPar.SubPar, InPar.SubPar];
    end
 case 'ordfilt'
    if (iscell(InPar.SubPar)),
       if (isempty(InPar.SubPar)),
 	     InPar.SubPar = {9,ones(3,3)};
       else
          % do nothing - SubPar is already a cell array format
       end
    else
       error('SubPar additional parameters for ordfilt method must be a cell array');
     end
 case 'poly'
    if (isnumeric(InPar.SubPar)),
       % put in cell array (to send to polysurface_fit.m)
       InPar.SubPar = {InPar.SubPar};
    end
 otherwise
    % do nothing
end

%--- check if input is matrix ---
IsNumeric = 0;
if (isnumeric(ImInput)),
   ImInput = {ImInput};
end

if (iscell(ImInput)),
   if (isnumeric(ImInput{1})),
      % input is given in a matrix form
      IsNumeric = 1;
      if (isempty(ImOutput)),
          switch InPar.Save
              case 'y'
	             error('If ImInput is numeric then must specify ImOutput');
              otherwise
                  % do nothing
          end
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


%--- Go over all images ---
for Iim=1:1:Nim,
   %--- read Image ImInput1 ---
   if (IsNumeric==1),
      InputImage = ImInputCell{Iim};
   else
      InputImage = fitsread(ImInputCell{Iim});
   end

   %--- CCDSEC for input image ---
   if (isempty(InPar.CCDSEC)),
      % use entire image
      % do nothing
   elseif (ischar(InPar.CCDSEC)),
      [InPar.CCDSEC] = get_ccdsec_fits(ImInputCell(Iim),InPar.CCDSEC);
      [InputImage]   = cut_image(InputImage,InPar.CCDSEC,'boundry');
   elseif (length(InPar.CCDSEC)==4),
      [InputImage]   = cut_image(InputImage,InPar.CCDSEC,'boundry');
   else
      error('Illegal CCDSEC input');
   end

   if (isempty(InPar.MaskStar)),
      % do not mask stars
      CleanedInputImage = InputImage;
   else
      if (ischar(InPar.MaskStar))
	     InPar.MaskStar = {InPar.MaskStar};   % set arguments for maskstars_fits in cell array
      end
      % mask stars:
      [~,~,Temp]=maskstars_fits(InputImage,[],InPar.MaskStar{:},'Save','n');
      CleanedInputImage = Temp{1};
      clear Temp;
   end

   %--- estimate the background level of image ---
   if (isa(Method,'function_handle')),
      % Method is function handle
      if (iscell(SubPar)),
    	 Background = feval(Method,Image,SubPar{:});
      else
	 error('When Method is a function handle, SubPar must be a cell array');
      end
   else
      % use a builtin method:
      switch lower(Method)
       case 'mode'
          InImStat = imstat_fits(CleanedInputImage);
          OutputImage = InputImage - InImStat.Mode;
       case 'median'
          InImStat = imstat_fits(CleanedInputImage);
          OutputImage = InputImage - InImStat.Median;
       case 'mean'
          InImStat = imstat_fits(CleanedInputImage);
          OutputImage = InputImage - InImStat.Mean;
       case 'medfilt'
          OutputImage = InputImage - medfilt2(CleanedInputImage,InPar.SubPar);
       case 'ordfilt'
          OutputImage = InputImage - ordfilt2(CleanedInputImage,InPar.SubPar{:});
       case 'poly'
          X = (1:1:size(InputImage,2));
          Y = (1:1:size(InputImage,1));
          % Normalize X and Y before doing the fit
          X = (X - 0.5.*X)./range(X);
          Y = (Y - 0.5.*Y)./range(Y);
          [ParStruct,Resid]=polysurface_fit(X,Y,CleanedInputImage,1,InPar.SubPar{:});

          OutputImage = InputImage - Resid;  % equivalent to modeled surface
       otherwise
          error('Unknown Method option');
      end
   end

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

   %--- Add to header comments regarding file creation ---
   [HeaderInfo] = cell_fitshead_addkey(HeaderInfo,...
                                       Inf,'COMMENT','','Created by imsubback_fits.m written by Eran Ofek',...
                                       Inf,'HISTORY','',sprintf('Background subtraction method: %s',Method),...
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
       fitswrite(OutputImage,OutImageFileName,HeaderInfo,InPar.DataType);
    case 'n'
       % do not save FITS image
    otherwise
       error('Unknown Save option');
   end

   if (nargout>1),
      OutputMatrixCell{Iim} = OutputImage;
   end

   if (nargout>2),
      BackMatrixCell{Iim} = InputImage - OutputImage;
   end

end



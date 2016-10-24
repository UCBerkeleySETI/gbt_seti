function [OutImageFileCell,OutputMatrixCell]=imfun_fits(Output,Fun,Input,varargin)
%-----------------------------------------------------------------------------
% imfun_fits function                                                 ImBasic
% Description: Applay a function to a set of images.
%              The function, may be operating on multiple images at a time.
% Input  : - List of output images (see create_list.m for details).
%          - Function handle or function name.
%          - First list of images on which the function is operating
%            (see create_list.m for details).
%            Alternatively this could be a scalar of vector (of scalars).
%          * Cell array containing an arbitrary number of additional list
%            of images on which the function is operating
%            (see create_list.m for details).
%            Alternatively this could be a scalar of vector (of scalars).
%          * Arbitrary number of pairs of input parameters: keyword,value,...
%            Available keywords are:
%            'Divide0'  - Set division by zero to specific value, default is 0.
%            'OutPrefix'- Add prefix before output image names,
%                         default is empty string (i.e., '').
%            'OutDir'   - Directory in which to write the output images,
%                         default is empty string (i.e., '').
%            'Header'   - From which input to copy header {1 | 2 | ...} or to
%                         use minimal header {0}.
%                         Default is 1.
%            'DataType' - Output data type (see fitswrite.m for options),
%                         default is float32.
%            'CCDSEC'   - Image sction for both images on which to do the
%                         arithmatic operation.
%                         This could be string containing image keyword
%                         name (e.g., 'CCDSEC'), or a vector of
%                         [Xmin, Xmax, Ymin, Ymax].
%                         The output image size will be equal to that specified
%                         in the CCDSEC keyword, so in practice the output
%                         image will be trimmed.
%                         If empty matrix (default; i.e., [])
%                         than do not use CCDSEC.
%             'Save'    - Save FITS image to disk {'y' | 'n'}.
%                         Default is 'y'.
% Output  : - Cell array containing output image names.
%           - Cell array of matrices or output images.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: OutImageFileCell=imfun_fits({'Out.fits'},@sqrt,{{'lred0015.fits'}});
%          OutImageFileCell=imfun_fits({'Out.fits'},@sqrt,{{{'lred0015.fits'}}});
%          OutImageFileCell=imfun_fits('Out.fits',@sqrt,{{{'lred0015.fits'}}});
%          % the following example: out_lred0010.fits = atan2(lred0010.fits,lred0010.fits)
%          % and so on for all the images in the list
%          OutImageFileCell=imfun_fits('lred001*.fits',@atan2,{{'lred001*.fits'},{'lred001*.fits'}},'OutPrefix','out_');         
% Reliable: 2
%-----------------------------------------------------------------------------

Divide0     = 0;
OutPrefix   = '';
OutDir      = '';
Header      = 1;
DataType    = 'float32';
CCDSEC      = [];
Save        = 'y';
Narg        = length(varargin);
for Iarg=1:2:Narg-1,
   switch lower(varargin{Iarg})
    case 'divide0'
       Divide0      = varargin{Iarg+1};
    case 'outprefix'
       OutPrefix    = varargin{Iarg+1};
    case 'outdir'
       OutDir       = varargin{Iarg+1};
    case 'header'
       Header       = varargin{Iarg+1};
    case 'datatype'
       DataType     = varargin{Iarg+1};
    case 'ccdsec'
       CCDSEC       = varargin{Iarg+1};
    otherwise
       error('Unknown keyword in input argument');
   end
end

if (~iscell(Input)),
   Input = {Input};
end

Nin = length(Input);
for Iin=1:1:Nin,
   if (isnumeric(Input{Iin}{1})),
      InputCell{Iin} = Input{Iin}{1};
   else
      [~,InputCell{Iin}] = create_list(Input{Iin}{1},NaN);
   end
end


%--- Construct a string containing Function name ---
if (ischar(Fun)),
   FunName = Fun
else
   FunName = func2str(Fun);
end

%--- List of output images ---
[~,OutputCell] = create_list(Output,NaN);
Nim = length(OutputCell);


%--- Go over all images ---
for I=1:1:Nim,
   HeaderComments = cell(1,0);  % Cell containing comments for header
   for Iin=1:1:Nin,
      if (iscell(InputCell{Iin})),
         % read image
         InImage{Iin} = fitsread(InputCell{Iin}{I});
         StrIm{Iin}   = InputCell{Iin}{I};
      else
         % assume image is a scalar
	 if (length(InputCell{Iin})==1),
            % assume that the user specified a scalar
   	    InImage{Iin} = InputCell{Iin}(1);
            StrIm{Iin}   = sprintf('%f',InputCell{Iin}(1));;
         else
            % assume that the user specified a vector of the correct length
	    InImage{Iin} = InputCell{Iin}(I);
            StrIm{Iin}   = sprintf('%f',InputCell{Iin}(I));;
         end

         if (Header==1),
            % image is a scalar...
            % in case Header is requested from the first image
            % than set it to image 0.
	    Header = 0;
         end
      end

      % Construct a cell array containing information
      % regarding the input images for the output image header
      IndComm = (Iin-1).*4;
      HeaderComments{IndComm+1} = Inf;
      HeaderComments{IndComm+2} = 'HISTORY';
      HeaderComments{IndComm+3} = '';
      HeaderComments{IndComm+4} = sprintf('InputImage number %d : %s',Iin,StrIm{Iin});

      %--- CCDSEC ---
      if (isempty(CCDSEC)),
         % use entire image
         % do nothing
      elseif (ischar(CCDSEC)),
	 [CCDSEC]=get_ccdsec_fits({InputCell1{I}},CCDSEC);
      
         [InImage{Iin}] = cut_image(InImage{Iin},CCDSEC,'boundry');
      elseif (length(CCDSEC)==4),
	 [InImage{Iin}] = cut_image(InImage{Iin},CCDSEC,'boundry');

      else
         error('Illegal CCDSEC input');
      end
   end

   % evaluate the function
   OutImage = feval(Fun,InImage{:});

   % deal with devision by zero
   Iinf = find(OutImage==Inf);
   OutImage(Iinf) = Divide0;

   OutImageFileName = sprintf('%s%s%s',OutDir,OutPrefix,OutputCell{I});
   OutImageFileCell{I} = OutImageFileName;

   %--- Prepare Header for output image ---
   switch Header
    case 0
       % use minimal header
       HeaderInfo = [];
    otherwise
       Info = fitsinfo(InputCell{Iin}{Header});
       HeaderInfo = Info.PrimaryData.Keywords;
   end

   %--- Add to header comments regarding file creation ---
   [HeaderInfo] = cell_fitshead_addkey(HeaderInfo,...
                       Inf,'COMMENT','','Created by imfun_fits.m written by Eran Ofek',...
                       Inf,'HISTORY','',sprintf('Function Name : %s',FunName),...
		       HeaderComments{:});

   %--- Write fits file ---
   switch lower(InPar.Save)
    case 'y'
       fitswrite(OutImage,OutImageFileName,HeaderInfo,DataType);
    case 'n'
       % do not save FITS image
    otherwise
       error('Unknown Save option');
   end

   if (nargout>1),
     OutputMatrixCell{Iim} = OutImage;
   end

end



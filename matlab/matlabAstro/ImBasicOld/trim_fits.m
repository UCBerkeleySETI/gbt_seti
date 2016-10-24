function OutMat=trim_fits(OutImages,InImages,CCDSEC,varargin)
%-----------------------------------------------------------------------------
% trim_fits function                                                  ImBasic
% Description: Trim a list of FITS images or 2D arrays.
% Input  : - List of output images (see create_list.m for details).
%            If an empty matrix (i.e., []), then use the InImages list.
%          - List of input images (see create_list.m for details).
%            Alternatively this can be a cell array of 2D matrices
%            or scalars. If scalars than trim operation is ignored.
%          - Image sction to trim from all images.
%            This could be string containing image keyword
%            name (e.g., 'CCDSEC'), or a vector of [Xmin, Xmax, Ymin, Ymax].
%            The output image size will be equal to that specified
%            in this parameter.
%            If empty matrix (default; i.e., []) than copy the full image.
%          * Arbitrary number of pairs of input arguments
%            ...,keyword,value,... where available keywords are: 
%            'ImType'   - Type of input images {'fits','array','mat'}.
%                         'fits' - A FITS image (default).
%                         'array' - A matrix in the matlab workspace.
%                         'mat' - A matlab mat file.
%            'CopyHead' - Copy header from original image to trimmed
%                         image {'y' | 'n}, default is 'y'.
%            'OutPrefix'- Add prefix before output image names,
%                         default is empty string (i.e., '').
%            'OutDir'   - Directory in which to write the output images,
%                         default is empty string (i.e., '').
%            'DataType' - Output data type (see fitswrite.m for options),
%                         default is float32.
%            'DATASEC'  - New data section keyword name in header 
%                         in which to write the data section of the
%                         new image which is [1:end,1:end].
%                         If given then will first delete the existing
%                         DATASEC keyword and then write the new one.
%                         Default is 'DATASEC'.
%                         If empty then don not write this keyword
%                         to header.
% Output : - Cell array containing trimmed images in matrix form.
%            This may be a very large array, so in order to save
%            memory, specify this ouput argument only if needed.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                    Jun 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: OutMat=trim_fits({'Out.fits'},{'Ex2.fits'},[1 2 1 2]);
% Reliable: 2
%-----------------------------------------------------------------------------

Def.CCDSEC    = [];
if (nargin==2),
   CCDSEC      = Def.CCDSEC;
else
   % do nothing
end

% varargin default parameters
ImType     = 'fits';
CopyHead   = 'y';
OutPrefix  = '';
OutDir     = '';
DataType   = 'float32';
DATASEC    = 'DATASEC';
Narg = length(varargin);
for Iarg=1:2:Narg-1,
   switch lower(varargin{Iarg})
    case 'outprefix'
       OutPrefix    = varargin{Iarg+1};
    case 'outdir'
       OutDir       = varargin{Iarg+1};
    case 'copyhead'
       Header       = varargin{Iarg+1};
    case 'datatype'
       DataType     = varargin{Iarg+1};
    case 'imtype'
       ImType       = varargin{Iarg+1};
    case 'datasec'
       DATASEC      = varargin{Iarg+1};
    otherwise
       error('Unknown keyword in input argument');
   end
end

switch lower(ImType)
 case 'array'
    % do nothing
    % assume InImages is a cell array of matrices
    Nim = length(InImages);
 case {'fits','mat'}
    [~,InputCell] = create_list(InImages,NaN);
    Nim = length(InputCell);
 otherwise
    error('Unknown ImType option');
end

if (~isempty(OutImages)),
   [~,OutputCell] = create_list(OutImages,NaN);
else
   OutputCell = InputCell;
end

%--- For each image ---
for Iim=1:1:Nim,
   %--- read current image ---
   switch lower(ImType)
    case 'array'
       CurImage = InImages{Iim};
    case 'fits'
       CurImage = fitsread(InputCell{Iim});
    case 'mat'
       Temp     = load(InputCell{Iim});
       if (isstruct(Temp)),
          FN = fieldnames(Temp);
          CurImage = getfield(Temp,FN);
          clear Temp;
       else
          CurImage = Temp;
          clear Temp;
       end
    otherwise
      error('Unknown ImType option');
   end

   %--- CCDSEC ---
   if (isempty(CCDSEC)),
      % use entire image
      % do nothing
   elseif (ischar(CCDSEC)),
      [CCDSEC]=get_ccdsec_fits({InputCell{I}},CCDSEC);

      [CurImage] = cut_image(CurImage,CCDSEC,'boundry');
   elseif (length(CCDSEC)==4),
      [CurImage] = cut_image(CurImage,CCDSEC,'boundry');
   else
      error('Illegal CCDSEC input');
   end

   if (nargout>0),
      OutMat{Iim} = CurImage;
   end

   %if (isempty(OutImages)),
   %   %--- Do not save output images in FITS format ---
   %else
      %--- Save output images in FITS format ---

      switch lower(CopyHead)
       case 'y'
          Head = fitsinfo(InputCell{Iim});
          HeaderInfo = Head.PrimaryData.Keywords;
       case 'n'
          HeaderInfo = [];
       otherwise
          error('Unknown CopyHead option');
      end

      %--- Write DATASEC keyword to header ---
      if (isempty(DATASEC)),
         % do not write DATASEC keyword to header
      else
         SizeCurImage = size(CurImage);
         if (isempty(HeaderInfo)),
            IndHead = 1;
         else
            % delete existing DATASEC keyword
            HeaderInfo = cell_fitshead_delkey(HeaderInfo,DATASEC);

            IndHead = size(HeaderInfo,1) + 1;
         end

         HeaderInfo{IndHead,1}   = DATASEC;
         HeaderInfo{IndHead,2}   = sprintf('[%d:%d,%d:%d]',...
                                            1,SizeCurImage(2),...
                                            1,SizeCurImage(1));
         HeaderInfo{IndHead,3}   = 'New data section written by trim_fits.m';
      end

      OutImageFileName = sprintf('%s%s%s',OutDir,OutPrefix,OutputCell{Iim});
      [Flag,HeaderInfo]=fitswrite(CurImage,OutImageFileName,...
                                  HeaderInfo, DataType);
   %end
end

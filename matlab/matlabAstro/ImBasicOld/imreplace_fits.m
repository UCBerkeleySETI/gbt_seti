function OutMat=imreplace_fits(OutImages,InImages,varargin)
%-----------------------------------------------------------------------------
% imreplace_fits function                                             ImBasic
% Description: Search for pixels with value in a given range and replace them
%              with another value.
% Input  : - List of output images (see create_list.m for details).
%            If an empty matrix (i.e., []), then will not save the
%            replaced images as FITS file (only return matrices in OutMat).
%          - List of input images (see create_list.m for details).
%            Alternatively this can be a cell array of 2D matrices.
%          * Arbitrary number of pairs of input arguments
%            ...,keyword,value,... where available keywords are: 
%            'Range'    - Matrix containing ranges to replace:
%                         [Min Max; Min Max; ...].
%                         Where each line contains a Min Max of a range.
%                         The program will look for pixel values in this
%                         range and will replace them with the appropriate
%                         value in the 'Value' input parameter.
%                         Default is [-Inf 0].
%            'Value'    - Vector of values to use as a replacments.
%                         Each element in the vector corresponds to each
%                         row in the 'Range' matrix. If scalar then
%                         use the same value for all the rows.
%                         Default is 0.
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
%            'Save'     - Save FITS image to disk {'y' | 'n'}.
%                         Default is 'y'.
% Output : - Cell array containing the new images in matrix form.
%            This may be a very large array, so in order to save
%            memory, specify this ouput argument only if needed.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                    Jun 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: OutMat=imreplace_fits('Out.fits','In.fits','Range',[-Inf 0],'Value',0);
% Reliable: 2
%-----------------------------------------------------------------------------

if (nargin<2),
   error('Illegal number of input arguments');
end

% varargin default parameters
DefV.Range      = [-Inf 0];
DefV.Value      = 0;
DefV.ImType     = 'fits';
DefV.CopyHead   = 'y';
DefV.OutPrefix  = '';
DefV.OutDir     = '';
DefV.DataType   = 'float32';
DefV.Save       = 'y';

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});


% Convert Value from Scalar to Vector
Nr = size(InPar.Range,1);
if (length(InPar.Value)==1),
   InPar.Value = InPar.Value.*ones(Nr,1);
end


switch lower(InPar.ImType)
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
end

%--- For each image ---
for Iim=1:1:Nim,
   %--- read current image ---
   switch lower(InPar.ImType)
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

   %--- Search and replace ---
   NpixRange = zeros(Nr,1);
   for Ir=1:1:Nr,
      Ind = find(CurImage>InPar.Range(Ir,1) & CurImage<InPar.Range(Ir,2));
      CurImage(Ind) = InPar.Value(Ir);
      NpixRange(Ir) = length(Ind);
   end

   if (nargout>0),
      OutMat{Iim} = CurImage;
   end

   if (isempty(OutImages)),
      %--- Do not save output images in FITS format ---
   else
      %--- Save output images in FITS format ---

      switch lower(InPar.CopyHead)
       case 'y'
          Head = fitsinfo(InputCell{Iim});
          HeaderInfo = Head.PrimaryData.Keywords;
          IndHead = size(HeaderInfo,1) + 1;

       case 'n'
          HeaderInfo = [];
          IndHead = 1;
       otherwise
          error('Unknown CopyHead option');
      end

      %--- Write Info about pixels replacement to header ---
      HeaderInfo{IndHead,1}   = 'HISTORY';
      HeaderInfo{IndHead,2}   = '';
      HeaderInfo{IndHead,3}   = 'Pixel values were replaced by imreplace_fits.m';
      for Ir=1:1:Nr,
         IndHead = IndHead + 1;
         HeaderInfo{IndHead,1}   = 'HISTORY';
         HeaderInfo{IndHead,2}   = '';
         HeaderInfo{IndHead,3}   = sprintf('Range: [%f %f] replaced by %f for %d pixels',InPar.Range(Ir,1),InPar.Range(Ir,2),InPar.Value(Ir),NpixRange(Ir));
      end
      OutImageFileName = sprintf('%s%s%s',InPar.OutDir,InPar.OutPrefix,OutputCell{Iim});

      switch lower(InPar.Save)
       case 'y'
          [Flag,HeaderInfo]=fitswrite(CurImage,OutImageFileName,...
                                      HeaderInfo, InPar.DataType);
       case 'n'
          % do not save FITS image
       otherwise
          error('Unknown Save option');
      end

   end
end

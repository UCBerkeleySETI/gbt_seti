function [Stat,InputCell,OutMat]=imstat_fits(InImages,varargin)
%----------------------------------------------------------------------------
% imstat_fits function                                               ImBasic
% Description: Given a list of FITS images or images in matrix format
%              calculate various statistics for each image.
% Input  : - List of output images (see create_list.m for details).
%            Alternatively this can be a cell array of 2D matrices,
%            or a 3D matrix in which the third dimension is image index.
%          * Arbitrary number of pairs of input arguments
%            ...,keyword,value,... where available keywords are: 
%            'ImType'   - Type of input images {'fits','array','mat'}.
%                         'fits' - A FITS image (default).
%                         'array' - A matrix in the matlab workspace.
%                         'mat' - A matlab mat file.
%            'CCDSEC'   - Image sction in which to calculate statistics.
%                         This could be string containing image keyword
%                         name (e.g., 'CCDSEC'), or a vector 
%                         of [Xmin, Xmax, Ymin, Ymax].
%                         The output image matrix size will be equal to
%                         that specified in this parameter.
%                         If empty matrix (default; i.e., []) than use
%                         the full image. Default is empty matrix.
% Output : - A vector of structures containing the statistics for
%            each image.
%            Statistic includes:
%            .Min, .Max, .Mean, .Median, .StD,
%            .NAXIS1, .NAXIS2, (self explanatory)
%            and:
%            .NumNAN     - Number of NaN pixels in image.
%            .Percentile - 1,2,3 sigma lower and upper percentiles.
%                          see err_cl.m for details.
%            .MeanErr    - Estimated error in the mean,
%                          calculated using the 68-percentile divide
%                          by sqrt of number of pixels.
%            .Mode       - Mode of the data calculated
%                          by looking for the most frquent data
%                          in a binned histogram in which the
%                          bin size is set by the MeanErr.
%          - A cell array of file names in the input images list.
%            In case ImType=array then return NaN.
%          - A cell array in which each cell contains the a matrix
%            of the 2D image.
%            This may be a very large array, so in order to save
%            memory, specify this ouput argument only if needed.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [Stat,InputCell,OutMat]=imstat_fits('lred0014.fits','CCDSEC',[100 200,100 200]);
%          Stat=imstat_fits('lred00*.fits');
%          Stat=imstat_fits({rand(100,100),rand(200,200)},'ImType','array');
%          Stat=imstat_fits(rand(100,100),'ImType','array');
% Reliable: 2
%----------------------------------------------------------------------------

% varargin default parameters
DefV.ImType     = 'fits';
DefV.CCDSEC     = [];

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});



if (iscell(InImages)),
   if (isnumeric(InImages{1})),
      % reset ImType to array because input is a matrix:
      InPar.ImType = 'array';
   end
end
if (isnumeric(InImages)),
   % 3D array
   InPar.ImType = 'array';
end


switch lower(InPar.ImType)
 case 'array'
    % assume InImages is a cell array of matrices
    if (~iscell(InImages)),
       if (ndims(InImages)==2),
          % convert to cell array
          InImages = {InImages};
          Nim = length(InImages);
       elseif (ndims(InImages)==3),
	  Nim = size(InImages,3);
          InPar.ImType = 'array3';
       else
	  error('Input matrix has unknown format');
       end 
    else
       Nim = length(InImages);
    end
    InputCell = [];
 case {'fits','mat'}
    [~,InputCell] = create_list(InImages,NaN);
    Nim = length(InputCell);
 otherwise
    error('Unknown ImType option');
end

%--- For each image ---
for Iim=1:1:Nim,
   %--- read current image ---
   switch lower(InPar.ImType)
    case 'array'
       CurImage = InImages{Iim};
    case 'array3'
       CurImage = InImages(:,:,Iim);
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
   if (isempty(InPar.CCDSEC)),
      % use entire image
      % do nothing
   elseif (ischar(InPar.CCDSEC)),
      [InPar.CCDSEC]=get_ccdsec_fits({InputCell{I}},InPar.CCDSEC);
      [CurImage] = cut_image(CurImage,InPar.CCDSEC,'boundry');
   elseif (length(InPar.CCDSEC)==4),
      [CurImage] = cut_image(CurImage,InPar.CCDSEC,'boundry');
   else
      error('Illegal CCDSEC input');
   end

   if (nargout>2),
      OutMat{Iim} = CurImage;
   end

   %--- Calculate Image statistics ---        
   Stat(Iim).NAXIS1 = size(CurImage,2);     % size of x-dimension
   Stat(Iim).NAXIS2 = size(CurImage,1);     % size of y-dimension
   Stat(Iim).Min    = minnd(CurImage);      % Minimum value
   Stat(Iim).Max    = maxnd(CurImage);      % Maximum value
   Stat(Iim).Mean   = nanmean(CurImage(:));     % Mean of image
   Stat(Iim).NumNAN = sum(isnan(CurImage)); % Number of NaNs

   Stat(Iim).Median = nanmedian(CurImage(:));   % Median of image
   Stat(Iim).StD    = nanstd(CurImage(:));      % StD of image
   % 1,2,3 sigma lower and upper percentiles:
   [Stat(Iim).Mode,Stat(Iim).Percentile,Stat(Iim).MeanErr]=mode_image(CurImage);

   %Stat(Iim).Percentile = err_cl(mat2vec(CurImage),[0.6827; 0.9545; 0.9973]);
   %% Estimate the error in the mean
   %Stat(Iim).MeanErr = 0.5.*(Stat(Iim).Percentile(1,2) - ...
   %			     Stat(Iim).Percentile(1,1))./ ...
   %                    sqrt(Stat(Iim).NAXIS1.*Stat(Iim).NAXIS2);
   %% Estimate the mode of the image
   %Factor = 1./Stat(Iim).MeanErr;
   %Stat(Iim).Mode = mode(round(Factor.*mat2vec(CurImage))./Factor);
end

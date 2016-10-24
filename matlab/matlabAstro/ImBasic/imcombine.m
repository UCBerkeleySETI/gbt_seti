function [ComIm,ZeroVec,ScaleVec,ComImNotNaN]=imcombine(InMat,varargin)
%-----------------------------------------------------------------------------
% imcombine function                                                  ImBasic
% Description: Combine set of 2D arrays into a single 2D array.
%              This function use imcombine_fits.m to combine FITS images.
% Input  : - Set of images to combine. Can be one of the following:
%            (1) A structure array of images.
%            (2) 
%            This could be:
%            A 3D array in which the 3rd dimension is the image index,
%            and the first two dimensions are position in each image.
%            A cell array of 2D matrices.
%          * Arbitrary pais of parameters: keyword, value,...
%            'RField'    - Field name in which the images are stored
%                          in the structure array. Default is 'Im'.
%            'Method'    - combining method:
%                          {'sum' | 'mean' | 'median' | 'std'},
%                          default is 'median'.
%            'Zero'      - Image offset (to add before scale),
%                          {'mean' | 'median' | 'mode' | 'constant' | 'none'},
%                          default is 'none'.
%                          Note that the zero willbe applied before the scaling.
%            'ZeroConst' - A scalar or vector of a constant offsets to be applied to
%                          each image. Default is [1].
%                          This is being used only if zero=constant.
%            'Scale'     - Image scaling {'mean' | 'median' | 'mode' | 'constant' | 'none'},
%                          default is 'none'.
%            'ScaleConst'- A scalar or vector of a constant scale to be applied to
%                          each image. Default is [1].
%                          This is being used only if scale=constant.
%            'Weight'    - Method by which to wheight the images.
%                          {'mean' | 'median' | 'mode' | 'constant' | 'images' | 'none'},
%                          default is 'none'.
%            'WeightConst'-A scalar or vector of a constant weight to be applied to
%                          each image. Default is [1].
%                          This is being used only if weight=constant.
%            'WeightFun' - Function name to apply to the weight value before weighting
%                          (e.g., @sqrt).
%                          If empty matrix (i.e., []) than donot apply a function,
%                          default is empty matrix.
%            'WeightIm'  - Set of images to use as weights.
%                          This can be a cell array of images, or a 3D cube.
%                          Default is [].
%            'Reject'    - Pixels rejection methods:
%                          'sigclip'   - std clipping.
%                          'minmax'    - minmax rejuction.
%                          'none'      - no rejuction, default.
%                          'sigclip' and 'minmax' are being applied only if
%                          Niter>1.
%            'RejPar'    - rejection parameters [low, high].
%                          for 'sigclip', [LowBoundSigma, HighBoundSigma], default is [Inf Inf].
%                          for 'minmax', [Number, Number] of pixels to reject default is [0 0].
%            'Niter'     - Number of iterations in rejection scheme. Default is 1.
%            'RepNegWith'- Replace negative pixels with a fixed value prior
%                          to combine. If empty, do nothing. Default is
%                          empty.
% Output : - Combined image.
%          - Vector containing the zero offsets per image used.
%          - Vector containing the scales per image used.
%          - 2D matrix corresponding to the combined image, in which each
%            pixel is a counter of the number of not-NaN values in all
%            the images in the position of a given pixel.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                    Jun 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [ComIm,ZeroVec,ScaleVec,ComImNotNaN]=imcombine(rand(100,100,10));
%          [ComIm,ZeroVec,ScaleVec,ComImNotNaN]=imcombine(rand(100,100,10),'Method','std');
% Reliable: 2
%-----------------------------------------------------------------------------
OneSigmaProb = 0.6827;

Narg = length(varargin);
if (Narg.*0.5~=floor(Narg.*0.5)),
   error('Illegal number of input arguments');
end

ImageField  = 'Im';

% set default values
DefV.RField      = ImageField;
DefV.Method      = 'median';
DefV.Zero        = 'none';
DefV.ZeroConst   = 1;
DefV.Scale       = 'none';
DefV.ScaleConst  = 1;
DefV.Weight      = 'none';
DefV.WeightConst = 1;
DefV.WeightFun   = [];
DefV.WeightIm    = [];
DefV.Reject      = 'none';
DefV.RejPar      = [];
DefV.Niter       = 1;
DefV.RepNegWith  = []; % do nothing
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


if (InPar.Niter<1),
   error('Niter miust be >=1');
end

if (isempty(InPar.RejPar)),
   switch lower(InPar.Reject)
    case 'sigclip'
       InPar.RejPar = [Inf Inf];
    case 'minmax'
       InPar.RejPar = [0 0];
    otherwise
       % do nothing
   end
end

%imshow(squeeze(InMat{1}),[0 30]);

if (iscell(InMat)),
   % make sure InMat is a row vector:
   if (size(InMat,2)==1),
      InMat = InMat.';
   end
   [ImSizeY, ImSizeX] = size(InMat{1});  % assume all the images have the same size;
   Nim                = length(InMat);   % number of images
   
   InMat3D = reshape(cell2mat(InMat),[ImSizeY, ImSizeX, Nim]);  % 3D cube of images
   clear InMat;

elseif (isnumeric(InMat) && ndims(InMat)==3),
   InMat3D = InMat;
   [ImSizeY, ImSizeX, Nim] = size(InMat);
   clear InMat;

elseif (isstruct(InMat)),
    InMat3D = sim2cube(InMat,'ImDim',3,'RField',InPar.RField);
    [ImSizeY, ImSizeX, Nim] = size(InMat3D);
    clear InMat;
else
   error('Input images list has an unsupported format');
end

% special prep
% what to do with negative pixels?
if (~isempty(InPar.RepNegWith)),
    InMat3D(InMat3D<0) = InPar.RepNegWith;
end


%--- Zero offset ---
switch lower(InPar.Zero)
 case {'none','no','n'}
    % do nothing
    ZeroVec = zeros(1,Nim);
 case 'mean'
    % take zero offset from mean of each image
    ZeroVec = nanmean(reshape(InMat3D,[ImSizeY.*ImSizeX, Nim]));
 case 'median'
    % take zero offset from median of each image
    ZeroVec = nanmedian(reshape(InMat3D,[ImSizeY.*ImSizeX, Nim]));
 case 'mode'
    ZeroVec = zeros(1,Nim);
    for Iim=1:1:Nim,
       Percentile = err_cl(mat2vec( queeze(InMat3D(:,:,Iim)) ),[OneSigmaProb]);
       MeanErr = 0.5.*(Percentile(1,2) - ...
		       Percentile(1,1))./ ...
	         sqrt(ImSizeX.*ImSizeY);
       Factor = 1./Stat(Iim).MeanErr;
       ZeroVec(Iim) = mode(round(Factor.*mat2vec(squeeze(InMat3D(:,:,Iim))))./Factor);
    end
 case 'constant'
    if (prod(size(InPar.ZeroConst))==1),
       ZeroVec = InPar.ZeroConst.*ones(1,Nim);
    else
       if (size(InPar.ZeroConst,1)==1),
	       ZeroVec = InPar.ZeroConst;
       else
	       ZeroVec = InPar.ZeroConst.';
       end
    end
 otherwise
    error('Unknown Zero option');
end

% apply offset
InMat3D = bsxfun(@minus,InMat3D, reshape(ZeroVec,[1 1 Nim]));

%--- Scale ---

switch lower(InPar.Scale)
 case {'none','no','n'}
    % do nothing
    ScaleVec = ones(1,Nim);
 case 'mean'
    % take zero offset from mean of each image
    ScaleVec = nanmean(reshape(InMat3D,[ImSizeY.*ImSizeX, Nim]));
 case 'median'
    % take zero offset from median of each image
    ScaleVec = nanmedian(reshape(InMat3D,[ImSizeY.*ImSizeX, Nim]));
 case 'mode'
    ScaleVec = zeros(1,Nim);
    for Iim=1:1:Nim,
       Percentile = err_cl(mat2vec( queeze(InMat3D(:,:,Iim)) ),[OneSigmaProb]);
       MeanErr = 0.5.*(Percentile(1,2) - ...
		       Percentile(1,1))./ ...
	         sqrt(ImSizeX.*ImSizeY);
       Factor = 1./Stat(Iim).MeanErr;
       ScaleVec(Iim) = mode(round(Factor.*mat2vec(squeeze(InMat3D(:,:,Iim))))./Factor);
    end
 case 'constant'
    if (prod(size(InPar.ScaleConst))==1),
       ScaleVec = InPar.ScaleConst.*ones(1,Nim);
    else
       if (size(InPar.ScaleConst,1)==1),
	        ScaleVec = InPar.ScaleConst;
       else
	        ScaleVec = InPar.ScaleConst.';
       end
    end
 otherwise
    error('Unknown Scale option');
end

ScaleVec = 1./ScaleVec;


% apply offset
InMat3D = bsxfun(@times,InMat3D, reshape(ScaleVec,[1 1 Nim]));


%--- Prep weights ---
switch lower(InPar.Weight)
 case {'none','no','n'}
    % no weight:
    WeightVec = ones(1,Nim);
 case 'mean'
    % take weight from mean of each image
    WeightVec = nanmean(reshape(InMat3D,[ImSizeY.*ImSizeX, Nim]))
 case 'median'
    % take weight from median of each image
    WeightVec = nanmedian(reshape(InMat3D,[ImSizeY.*ImSizeX, Nim]))
 case 'mode'    
    WeightVec = zeros(1,Nim);
    for Iim=1:1:Nim,
       Percentile = err_cl(mat2vec( queeze(InMat3D(:,:,Iim)) ),[OneSigmaProb]);
       MeanErr = 0.5.*(Percentile(1,2) - ...
		       Percentile(1,1))./ ...
	         sqrt(ImSizeX.*ImSizeY);
       Factor = 1./Stat(Iim).MeanErr;
       WeightVec(Iim) = mode(round(Factor.*mat2vec(squeeze(InMat3D(:,:,Iim))))./Factor);
    end
 case 'constant'
    if (prod(size(InPar.WeightConst))==1),
       WeightVec = InPar.WeightConst.*ones(1,Nim);
    else
       if (size(InPar.WeightConst,1)==1),
	       WeightVec = InPar.WeightConst;
       else
	       WeightVec = InPar.WeightConst.';
       end
    end
 case 'images'
    if (iscell(InPar.WeightIm)),
       [WImSizeY, WImSizeX] = size(InPar.WeightIm{1});  % assume all the images have the same size;
       NWim                 = length(InPar.WeightIm);   % number of images
       WeightIm3D           = reshape(cell2mat(InPar.WeightIm),[WImSizeY, WImSizeX, NWim]);  % 3D cube of images
       clear WeightIm;
    elseif (ndims(InPar.WeightIm)==3),
       WeightIm3D = InPar.WeightIm;
       clear InPar.WeightIm;
    else
       error('Input weight images list has an unsupported format');
    end
 otherwise
    error('Unknown Zero option');
end


% apply weights
switch lower(InPar.Weight)
 case 'images'
    % Weight is an image per image
    InMat3D    = InMat3D.*WeightIm3D;
    WeightNorm = nanmean(WeightIm3D,3);
 otherwise
    % weight is a scalar per image
    InMat3D    = bsxfun(@times,InMat3D, reshape(WeightVec,[1 1 Nim]));
    WeightNorm = nanmean(WeightVec);
end

%--- Combine ---
switch InPar.Reject
 case {'none','no','n'}
    % do nothing
    Inpar.Niter = 1;
 otherwise
    % do nothing
end

WRej  = ones([ImSizeY, ImSizeX, Nim]);  % 1 - use; NaN - not use (sigma clipping)

for Iiter=1:1:InPar.Niter,
   switch lower(InPar.Method)
    case 'sum'
       ComIm = nansum(InMat3D.*WRej,3);
       %imshow(squeeze(InMat3D(:,:,1)),[0 30]);
    case 'mean'
       ComIm = nanmean(InMat3D.*WRej,3);
    case 'median'
       ComIm = nanmedian(InMat3D.*WRej,3);
    case 'std'
       ComIm = nanstd(InMat3D.*WRej,[],3);
    otherwise
       error('Unknown Method option');
   end

   if (InPar.Niter>1),
      switch InPar.Reject
       case {'none','no','n'}
          % do nothing
          InPar.Niter = 1;
       case 'minmax'
          SortedInMat3D = sort(InMat3D,3);
          InMat3D       = SortedInMat3D(:,:,[1+Inpar.RejPar(1):end-Inpar.RejPar(2)]);
          WRej = 1;
       case 'sigclip'
          StdIm = std(InMat3D,[],3);
          WRej  = zeros([ImSizeY, ImSizeX, Nim]).*NaN;

          I1 = find(bsxfun(@rdivide,bsxfun(@minus,InMat3D,ComIm),StdIm)>-Inpar.RejPar(1) & ...
                    bsxfun(@rdivide,bsxfun(@minus,InMat3D,ComIm),StdIm)<InPar.RejPar(2));

%          I1    = find((InMat3D - ComIm)./StdIm>-RejPar(1) && (InMat3D - ComIm)./StdIm<RejPar(2));
          WRej(I1) = 1;
       otherwise
          error('Unknown Reject option');
      end
   end
end

ComIm = ComIm./WeightNorm;

if (nargout>3),
   ComImNotNaN = nansum(~isnan(InMat3D),3);
end


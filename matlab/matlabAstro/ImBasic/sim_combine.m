function [ComIm,ZeroVec,ScaleVec,ComImNotNaN]=sim_combine(Sim,varargin)
%--------------------------------------------------------------------------
% sim_combine function                                             ImBasic
% Description: Combine set of 2D arrays into a single 2D array.
%              This function use imcombine_fits.m to combine FITS images.
% Input  : - Set of images to combine. Can be one of the following:
%            (1) Cell array of image names in string format.
%            (2) String containing wild cards (see create_list.m for
%                option). E.g., 'lred00[15-28].fits' or 'lred001*.fits'.
%            (3) Structure array of images (SIM).
%                The image should be stored in the 'Im' field.
%                This may contains also mask image (in the 'Mask' field),
%                and an error image (in the 'ErrIm' field).
%            (4) Cell array of matrices.
%            (5) A file contains a list of image (e.g., '@list').
%            (6) A cube of images in which the the image index is in the
%                third dimension.
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
%            'OutType'   - The type of output format {'sim'|'struct'|'mat'}.
%                          Default is 'sim'.
%            'Zero'      - Image offset (to subtract before scale),
%                          {'mean' | 'median' | 'mode' | 'constant' | 'none'|'mode_fit'},
%                          default is 'none'.
%                          Note that the zero willbe applied before the scaling.
%            'ZeroConst' - A scalar or vector of a constant offsets to be applied to
%                          each image. Default is [1].
%                          This is being used only if zero=constant.
%            'Scale'     - Image scaling {'mean' | 'median' | 'mode' | 'constant' | 'none'|'mode_fit'},
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
%            'CopyHead'  - Copy image header from the first image in stack
%                          (if exist) {true|false}. Default is true.
%                          Relevant only for SIM or struct output.
%            'AddHead'   - A 3 column cell array of additional header
%                          key, val, comments to add to header.
%                          Default is {}.
% Output : - Combined image.
%          - Vector containing the zero offsets per image used.
%          - Vector containing the scales per image used.
%          - 2D matrix corresponding to the combined image, in which each
%            pixel is a counter of the number of not-NaN values in all
%            the images in the position of a given pixel.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [ComIm,ZeroVec,ScaleVec,ComImNotNaN]=sim_combine(rand(100,100,10));
%          [ComIm,ZeroVec,ScaleVec,ComImNotNaN]=sim_combine(rand(100,100,10),'Method','std');
% Reliable: 2
%--------------------------------------------------------------------------
OneSigmaProb = 0.6827;

Narg = length(varargin);
if (Narg.*0.5~=floor(Narg.*0.5)),
   error('Illegal number of input arguments');
end

ImageField     = 'Im';
HeaderField    = 'Header';
WeightImField  = 'WeightIm';

% set default values
DefV.RField      = ImageField;
DefV.Method      = 'median';
DefV.OutType     = 'sim';
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
DefV.CopyHead    = true;
DefV.AddHead     = {};

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

if (isnumeric(Sim) && ndims(Sim)==3),
    % already a cube
    InMat3D = Sim;
else
   Sim     = images2sim(Sim,varargin{:});
   InMat3D = sim2cube(Sim,varargin{:},'ImDim',3);
end
% save Sim(1) header
if (isfield_notempty(Sim(1),HeaderField)),
    Header  = Sim(1).(HeaderField);
else
    Header  = {};
end
clear Sim;

[ImSizeY, ImSizeX, Nim] = size(InMat3D);


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
 case 'mode_fit'
     ZeroVec = zeros(1,Nim);
     for Iim=1:1:Nim,
        ZeroVec(Iim) = mode_fit(squeeze(InMat3D(:,:,Iim)),varargin{:});
     end
 case 'mode'
    ZeroVec = zeros(1,Nim);
    for Iim=1:1:Nim,
       Percentile = err_cl(mat2vec( squeeze(InMat3D(:,:,Iim)) ),[OneSigmaProb]);
       MeanErr = 0.5.*(Percentile(1,2) - ...
		       Percentile(1,1))./ ...
	           sqrt(ImSizeX.*ImSizeY);
       %Factor = 1./Stat(Iim).MeanErr;
       Factor = 1./MeanErr;
       ZeroVec(Iim) = mode(round(Factor.*mat2vec(squeeze(InMat3D(:,:,Iim))))./Factor);
    end
 case 'constant'
    if (numel(InPar.ZeroConst)==1),
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
  case 'mode_fit'
     ScaleVec = zeros(1,Nim);
     for Iim=1:1:Nim,
        ScaleVec(Iim) = mode_fit(squeeze(InMat3D(:,:,Iim)),varargin{:});
     end   
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


% apply scale
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
    if (numel(InPar.WeightConst)==1),
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

if (nargout>3 || strcmpi(InPar.OutType,'sim') || strcmpi(InPar.OutType,'struct')),
   ComImNotNaN = nansum(~isnan(InMat3D),3);
end

% save as SIM
switch lower(InPar.OutType)
    case {'sim','struct'}
        ComSim.(ImageField)      = ComIm;
        clear ComIm;
        ComIm = ComSim;
        % construct image header
        if (InPar.CopyHead),
            ComIm.(HeaderField)     = Header;
        end
        if (~isempty(InPar.AddHead)),
            if (~isfield(ComIm,HeaderField)),
                ComIm.(HeaderField) = {};
            end
            Tmp = InPar.AddHead.';
            ComIm.(HeaderField)     = cell_fitshead_update(ComIm.(HeaderField),Tmp{:});
        end
        
        ComIm.(WeightImField)   = ComImNotNaN;
        
        if (strcmpi(InPar.OutType,'sim')),
            ComIm = struct2sim(ComIm);
        end
    case 'mat'
        % do nothing
    otherwise
        error('Unknown OutType option');
end
        
    




function [OutImageFileCell,FlatOutput,SuperFlatStat]=flatcreate_fits(FlatInput,FlatOutput,varargin)
%-----------------------------------------------------------------------------
% flatset_fits function                                               ImBasic
% Description: Create a flat field from a single filter set of images.
%              See also flat_fits.m; flatset.fits.m; optimal_flatnorm_fits.m
%              This function is obsolote: Use ImBasic toolbox.
% Input  : - List of input images from which to create flat
%            See create_list.m for details.
%            Default is '*.fits'. If empty matrix then use default.
%            All images should be of the same filter.
%          - String containing the name of the output super flat FITS
%            image. Default is 'SuperFlat.fits'.
%            If empty use default.
%          * Arbitrary number of pairs of input parameters: keyword,value,...
%            Available keywords are:
%            'Combine'  - Combining method using: {'mean'|'median'}.
%                         Default is 'median'.
%            'Mask'     - Indicating if the program masks stars in the flat images
%                         before creating the super flat {'none','mask'}.
%                         Default is 'none'.
%            'CheckFilt'- String or cell array of strings containing name of
%                         Filter header keywords.
%                         If specified then the program will check that
%                         all the images are of the same filter.
%                         If not (or empty) all the images are of the same filter
%                         then the program will abort. Default is empty
%                         matrix.
%            'DataType' - Output data type (see fitswrite.m for options), 
%                         default is float32.
%            'StD'      - String containing the name of an optional
%                         FITS image of the StD of all the flat images.
%                         This may be useful to identify bad pixels.
%                         If empty matrix than the StD image will not
%                         be calculated. Default is empty matrix.
%                         The StD is calculated over the median normalized
%                         3D image.
%            'Sum'      - String containing the name of an optional
%                         FITS image of the sum of all the flat images.
%                         This may be useful for S/N estimate.
%                         If empty matrix than the Sum image will not
%                         be calculated. Default is empty matrix.
%            'Nn'       - String containing the name of an optional
%                         FITS image of the number of non NaN values in each
%                         pixel.
%            'PreScale' - Scaling method to apply to each image prior to
%                         combining the images: {'mean'|'median'|'mode'|'optimal'|'none'}.
%                         Default is 'median'.
%            'NormFlat' - Function by which to normalize final flat image:
%                         {'mean' | 'median' | 'mode' | 'none'}, default is 'median'.
%            'RemNeg'   - Replace negative or zero pixels in the images by NaN {'y'|'n'},
%                         default is 'y'.
%            'Divide0'  - Set division by zero to specific value, default is 0.
%            'Reject'   - Pixels rejection methods in bias construction:
%                         'sigclip'   - std clipping.
%                         'minmax'    - minmax rejuction.
%                         'none'      - no rejuction, default.
%                         'sigclip' and 'minmax' are being applied only if
%                         Niter>1.
%            'RejPar'   - rejection parameters [low, high].
%                         for 'sigclip', [LowBoundSigma, HighBoundSigma], default is [Inf Inf].
%                         for 'minmax', [Number, Number] of pixels to reject default is [0 0].
%            'Niter'    - Number of iterations in rejection scheme. Default is 1.
%            'CheckSat' - Check if input flat images are saturated {'y' | 'n'}.
%                         Default is 'y'.
%            'SatLevel' - Saturation level [adu].
%                         If CheckStat='y' then program will discard images
%                         in which the fraction of saturated pixels is larger
%                         than 'SatFrac'. Default is 30000.
%                         See description in identify_flat_fits.m 
%            'SatFrac'  - Maximum allowed fraction of saturated pixels.
%                         This is used only if CheckSat='y'. Default is 0.02.
%                         See description in identify_flat_fits.m 
%            'SatKey'   - A string or cell array of strings containing
%                         header keyword which stors the saturation level
%                         of the CCD.
%                         See default and description in identify_flat_fits.m 
%            'SatNaN'   - Replace saturated pixels with NaNs {'y' | 'n'},
%                         default is 'y'.
%            'DETECT_THRESH'  - Detection threshold to pass to SExtractor.
%                         See default and description in maskstars_fits.m
%            'DETECT_MINAREA' - Dectection minimum area to pass to SEx.
%                         See default and description in maskstars_fits.m
%            'Verbose'  - Print progress report {'y' | 'n'}.
%                         Default is 'y'.
% Output  : - Cell array containing output image names.
%             This list may be different in length than the input images list.
%             For example, if the flat images are identified automatically,
%             the program will not divide the flat images by the super flat image.
%           - String containing the name of the super flat FITS image.
%             If empty matrix than SuperFlat image was not constructed
%             due to lack of flat images.
%             In this case the StD and NonLinear images will also not be generated.
%           - Structure containing statistics regarding the flat image.
%             This includes all the statistical properties returned by imstat_fits.m.
%             If empty matrix than SuperFlat image was not constructed.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: 
% Reliable: 2
%-----------------------------------------------------------------------------
MaskStars_Prefix = 'maskstar_';
OutImageFileCell = [];
SuperFlatStat    = [];

Def.FlatInput  = '*.fits';
Def.FlatOutput = 'SuperFlat.fits';
if (nargin==0),
   FlatInput  = Def.FlatInput;
   FlatOutput = Def.FlatOutput;
elseif (nargin==1),
   FlatOutput = Def.FlatOutput;
else
   % do nothing
end

if (isempty(FlatOutput)),
   FlatOutput  = Def.FlatOutput;
end


% deal with varargin:
DefV.Combine     = 'median';
DefV.Mask        = 'none';  
DefV.CheckFilt   = []; %{'FILTER', 'FILTER1'};
DefV.CopyHead    = 'y';
DefV.DataType    = 'float32';
DefV.StD         = [];
DefV.Sum         = [];
DefV.Nn          = [];
DefV.NonLinear   = [];
DefV.PreScale    = 'median';
DefV.NormFlat    = 'median';
DefV.RemNeg      = 'y';
DefV.Divide0     = 0;
DefV.Reject      = 'none';
DefV.RejPar      = [];
DefV.Niter       = 1;
% parameters for maskstars_fits.m
DefV.DETECT_THRESH  = [];
DefV.DETECT_MINAREA = [];
% parameters for identify_flat_fits.m
DefV.CheckSat    = 'n';
DefV.SatLevel    = 30000;
DefV.SatFrac     = 0.02;
DefV.SatKey      = [];
DefV.SatNaN      = 'y';
DefV.Verbose     = 'y';

InPar = set_varargin_keyval(DefV,'n','def',varargin{:});

if (~iscell(InPar.CheckFilt)),
   InPar.CheckFilt = {InPar.CheckFilt};
end


% Generate list from FlatInput:
[~,FlatInputCell] = create_list(FlatInput,NaN);
NflatOrig = length(FlatInputCell);

%--- remove saturated flats images from list ---
switch lower(InPar.CheckSat)
 case 'n'
    % do nothing
 case 'y'
    % Only check for saturation and remove satrurated flats from list
    [FlatInputCell] = identify_flat_fits(FlatInputCell,varargin{:},'OnlyCheck','y'); 
 otherwise
    erroe('Unknown CheckSat option');
end

Nflat = length(FlatInputCell);

%--- Check that all the images were taken using the same filter ---
if (isempty(InPar.CheckFilt{1})),
   % do not check for filter consistency
else
   % check for filter consistency
   [Groups,GroupStr,GroupInd,ImListCell] = keyword_grouping_fits([FlatInputCell(:); ImInputCell(:)],...
								 InPar.CheckFilt);
   if (length(GroupStr)>1),
      error('Images were taken using more then one filter, use flat_fits.m instead');
   end
end



if (Nflat==0),
   FlatOutput = [];
   fprintf('No images from which flat can be constructed were found\n');
else
   % FlatInputCell contains a list of all images from which to construct a flat
   %--- mask stars ---
   switch lower(InPar.Mask)
    case 'none'
       % do nothing
       % use images as is
    case 'mask'
       % mask stars before creating the flat

       [FlatInputCell] = maskstars_fits(FlatInputCell,[],'sex',...
				     'OutPrefix',MaskStars_Prefix,...
      				     'DETECT_THRESH',InPar.DETECT_THRESH,...
      				     'DETECT_MINAREA',InPar.DETECT_MINAREA);

    otherwise
       error('Unknown Method option');
   end


   %--- Read all images into memory ---
   Mat3D = create_mat3d_images(FlatInputCell,[]);

   %--- replace all negative or zero pixels with NaNs ---
   switch lower(InPar.RemNeg)
    case 'y'
       Mat3D(find(Mat3D<=0)) = NaN;
    case 'n'
       % do nothing
    otherwise
       error('Unknown RemNeg option');
   end


   %--- Replace saturated pixels with NaNs ---
   switch lower(InPar.SatNaN)
    case 'y'
       Mat3D(find(Mat3D>InPar.SatLevel)) = NaN;
    case 'n'
       % do nothing
    otherwise
       error('Unknown SatNaN option');
   end
   
   % construct Flat image
   switch lower(InPar.Verbose)
    case 'y'
       fprintf('Constructing Flat image: %s from %d images\n',FlatOutput,length(FlatInputCell));
    otherwise
       % do nothing
   end

   %--- find Sum Image ---
   if (isempty(InPar.Sum)),
      % do nothing
   else
      fitswrite(nansum(Mat3D,3),InPar.Sum);
   end

   % find pre scale normalizations:
   switch lower(InPar.PreScale)
    case 'optimal'
       % and return normalize images
       [Norm,~,Nnn,~,Mat3D]=optimal_flatnorm_fits(Mat3D);
    otherwise
       StatPre = imstat_fits(Mat3D);
       switch lower(InPar.PreScale)
        case 'mean'
           Norm = [StatPre(:).Mean];
        case 'median'
           Norm = [StatPre(:).Median];
        case 'mode'
           Norm = [StatPre(:).Mode];
        case 'none'
   	   Norm = ones(1,length(FlatInputCell));
        otherwise
   	   error('Unknown PreScale option');
        end
        % normalize images
        [Ny,Nx,Nim] = size(Mat3D);
        Mat3D = reshape(bsxfun(@times,reshape(Mat3D,[Ny.*Nx Nim]),1./Norm),[Ny Nx Nim]);
   end
   fprintf('Pre flat normalizations: ');
   for Inorm=1:1:length(Norm),
      fprintf('%f  ',Norm(Inorm));
   end
   fprintf('\n');



%Mat2D = reshape(Mat3D,[Ny.*Nx Nim]);
%nanmedian(bsxfun(@times,Mat2D,1./Norm),1)'
%squeeze(Mat3D(500,600,:))
%1./Norm
%   Mat3D = reshape(bsxfun(@times,reshape(Mat3D,[Ny.*Nx Nim]),1./Norm),[Ny Nx Nim]);
%squeeze(Mat3D(500,600,:))

   % combine the final flat image
   [ComIm,ZeroVec,ScaleVec,ComImNotNaN]=imcombine(Mat3D,...
					       'Method',InPar.Combine,...
					       'Zero','none',...
					       'Scale','none',...
					       'Weight','none',...
					       'Reject',InPar.Reject,...
					       'RejPar',InPar.RejPar,...
					       'Niter',InPar.Niter);


   % delete makestar_*
   delete(sprintf('%s*',MaskStars_Prefix));

   %--- Write StD image ---
   if (isempty(InPar.StD)),
      % do not calculate StD image
   else
      StD = std(Mat3D./nanmedian(Mat3D(:)),[],3);
      fitswrite(StD,InPar.StD);
      % get total StD:
      %AA=Mat3D./nanmedian(Mat3D(:));
      %nanstd(AA(:))
   end

   %--- Write Nn image ---
   if (isempty(InPar.Nn)),
      % do not calculate Nn image
   else
      Nnn = sum(~isnan(Mat3D),3);
      fitswrite(StD,InPar.Nn);
   end

   % calculate unnormalized Flat statistics

   switch lower(InPar.NormFlat)
    case 'mean'
       %Norm = Stat.Mean;
       Norm = nanmean(ComIm(:));
    case 'median'
       %Norm = Stat.Median;
       Norm = nanmedian(ComIm(:));
    case 'mode'
       Norm = mode_image(ComIm);
    case 'none'
       Norm = 1;
    otherwise
       error('Unknown NormFlat option');
   end

   % normalize the flat image
   ComIm = ComIm./Norm;
   fitswrite(ComIm,FlatOutput);
   %imarith_fits(FlatOutput,FlatOutput,'/',Norm);

   %--- calculate Flat field statistics ---
   [SuperFlatStat]=imstat_fits(ComIm);

   %--- calculate NonLinearity images ---
   if (isempty(InPar.NonLinear)),
      % do nothinh
   else
      error('NonLinear option not exist yet');
   end

end




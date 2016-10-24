function [OutImageFileCell,FlatOutput,SuperFlatStat]=flatset_fits(ImInput,FlatInput,ImOutput,FlatOutput,varargin)
%-----------------------------------------------------------------------------
% flatset_fits function                                               ImBasic
% Description: Given a set of images from which to construct a flat field.
%              Correct another set of images of the same size.
%              This is a simpleer version of flat_fits.m which works
%              on a predefined lists. All the images in these lists
%              should be taken using the same filter (see also flat_fits.m).
%              This function is obsolote: Use ImBasic toolbox.
% Input  : - List of input images (see create_list.m for details).
%            Default is '*.fits'. If empty matrix then
%            use default.
%          - List of flat images (see create_list.m for details).
%            Default is the ImInput list. If empty matrix then
%            use default.
%          - List of flat divided output images
%            (see create_list.m for details).
%            This list should have the same length as the list given by the
%            first input argument (ImInput).
%            If empty matrix than will use ImInput as the output list.
%            If empty matrix and if 'OutPrefix' is not specified than
%            will set 'OutPrefix' to 'f_'.
%          - String containing the name of the output super flat FITS
%            image. Default is 'SuperFlat.fits'.
%            If empty use default.
%          * Arbitrary number of pairs of input parameters: keyword,value,...
%            Available keywords are:
%            'Combine'  - Combining method using: {'mean'|'median'}.
%                         Default is 'median'.
%            'Method'   - Indicating if the program masks stars in the
%                         flat images before creating the super flat
%                         {'none','mask'}. Default is 'none'.
%            'OutPrefix'- Add prefix before output image names,
%                         default is 'f_'.
%            'OutDir'   - Directory in which to write the output images,
%                         default is empty string (i.e., '').
%            'CheckFilt'- String or cell array of strings containing name of
%                         Filter header keywords.
%                         If specified then the program will check that
%                         all the images are of the same filter.
%                         If not (or empty) all the images are of the same
%                         filter then the program will abort. Default is empty
%                         matrix.
%            'CopyHead' - Copy header from the original images to the
%                         flat subtracted images {'y' | 'n'}, default is 'y'.
%            'DataType' - Output data type (see fitswrite.m for options), 
%                         default is float32.
%            'StD'      - String containing the name of an optional
%                         FITS image of the StD of all the flat images.
%                         This may be useful to identify bad pixels.
%                         If empty matrix than the StD image will not
%                         be calculated. Default is empty matrix.
%            'Sum'      - String containing the name of an optional
%                         FITS image of the sum of all the flat images.
%                         This may be useful for S/N estimate.
%                         If empty matrix than the Sum image will not
%                         be calculated. Default is empty matrix.
%            'NonLinear'- Not operational yet.
%                         String containing the name of an optional
%                         FITS image which indicate the amount of
%                         non-linearity in the response of each pixel.
%                         .... define...
%                         If empty matrix than the non-linearity image will not
%                         be calculated. Default is empty matrix.
%            'PreScale' - Scaling method to apply to each image prior to
%                         combining the images:
%                         {'mean'|'median'|'mode'|'optimal'}.
%                         Default is 'median'.
%            'NormFlat' - Function by which to normalize final flat image:
%                         {'mean' | 'median' | 'mode'}, default is 'median'.
%            'Divide0'  - Set division by zero to specific value, default is 0.
%            'Reject'   - Pixels rejection methods in flat construction:
%                         'sigclip'   - std clipping.
%                         'minmax'    - minmax rejuction.
%                         'none'      - no rejuction, default.
%                         'sigclip' and 'minmax' are being applied only if
%                         Niter>1.
%            'RejPar'   - rejection parameters [low, high].
%                         for 'sigclip', [LowBoundSigma, HighBoundSigma],
%                         default is [Inf Inf].
%                         for 'minmax', [Number, Number] of pixels to reject
%                         default is [0 0].
%            'Niter'    - Number of iterations in rejection scheme.
%                         Default is 1.
%            'FlatCor'  - Divide all the images by the super flat image
%                         {'y' | 'n'}, default is 'y'.
%            'CheckSat' - Check if Flat image is saturated {'y' | 'n'}.
%                         Default is 'n'. The default here is not the same as
%                         in flat_fits.m (which already checks for saturation
%                         by default).
%            'SatLevel' - Saturation level [adu].
%                         If CheckStat='y' then program will discard images
%                         in which the fraction of saturated pixels is larger
%                         than 'SatFrac'.
%                         See default and description in identify_flat_fits.m 
%            'SatFrac'  - Maximum allowed fraction of saturated pixels.
%                         This is used only if CheckSat='y'.
%                         See default and description in identify_flat_fits.m 
%            'SatKey'   - A string or cell array of strings containing
%                         header keyword which store the saturation level
%                         of the CCD.
%                         See default and description in identify_flat_fits.m 
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

Def.OutPrefix  = 'f_';
Def.ImInput    = '*.fits';
Def.FlatInput  = [];
Def.ImOutput   = [];
Def.FlatOutput = 'SuperFlat.fits';
if (nargin==0),
   ImInput    = Def.ImInput;
   FlatInput  = Def.FlatInput;
   ImOutput   = Def.ImOutput;
   FlatOutput = Def.FlatOutput;
elseif (nargin==1),
   FlatInput  = Def.FlatInput;
   ImOutput   = Def.ImOutput;
   FlatOutput = Def.FlatOutput;
elseif (nargin==2),
   ImOutput   = Def.ImOutput;
   FlatOutput = Def.FlatOutput;
elseif (nargin==3),
   FlatOutput = Def.FlatOutput;
else
   % do nothing
end

if (isempty(ImInput)),
   ImInput = Def.ImInput;
end

if (isempty(FlatInput)),
   FlatInput = ImInput;
end

if (isempty(ImOutput)),
   ImOutput = ImInput;
end

if (isempty(FlatOutput)),
   FlatOutput  = Def.FlatOutput;
end


% deal with varargin:
DefV.Combine     = 'median';
DefV.Method      = 'none';  
DefV.OutPrefix   = Def.OutPrefix;
DefV.OutDir      = '';
DefV.CheckFilt   = []; %{'FILTER', 'FILTER1'};
DefV.CopyHead    = 'y';
DefV.DataType    = 'float32';
DefV.StD         = [];
DefV.Sum         = [];
DefV.NonLinear   = [];
DefV.PreScale    = 'median';
DefV.NormFlat    = 'median';
DefV.Divide0     = 0;
DefV.Reject      = 'none';
DefV.RejPar      = [];
DefV.Niter       = 1;
DefV.FlatCor     = 'y';
% parameters for maskstars_fits.m
DefV.DETECT_THRESH  = [];
DefV.DETECT_MINAREA = [];
% parameters for identify_flat_fits.m
DefV.CheckSat    = 'n';
DefV.SatLevel    = [];
DefV.SatFrac     = [];
DefV.SatKey      = [];
DefV.Verbose     = 'y';

InPar = set_varargin_keyval(DefV,'n','def',varargin{:});

% if ImOutput = [] and OutPrefix not specified than set OutPrefix to 'f_'
if (isempty(ImOutput) && isempty(find(strcmpi(varargin,'OutPrefix')==1,1))),
  InPar.OutPrefix = Def.OutPrefix;
end

if (~iscell(InPar.CheckFilt)),
   InPar.CheckFilt = {InPar.CheckFilt};
end


% Generate list from ImInput:
[~,ImInputCell] = create_list(ImInput,NaN);

% Generate list from FlatInput:
[~,FlatInputCell] = create_list(FlatInput,NaN);
NflatOrig = length(FlatInputCell);

%--- remove saturated flats ---
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
else

   % FlatInputCell contains a list of all images from which to construct a flat
   %--- mask stars ---
   switch lower(InPar.Method)
    case 'none'
       % do nothing
       % use images as is
    case 'mask'
       % mask stars before creating the flat

       [FlatInputCell] = maskstars_fits(FlatInputCell,[],'sex',...
                                     'OutDir',InPar.OutDir,...
				     'OutPrefix',MaskStars_Prefix,...
      				     'DETECT_THRESH',InPar.DETECT_THRESH,...
      				     'DETECT_MINAREA',InPar.DETECT_MINAREA);

    otherwise
       error('Unknown Method option');
   end
   
   % construct Flat image
   switch lower(InPar.Verbose)
    case 'y'
       fprintf('Constructing Flat image: %s from %d images\n',FlatOutput,length(FlatInputCell));
    otherwise
       % do nothing
   end

   switch lower(InPar.PreScale)
    case 'optimal'
       [Norm,~,Nnn,~,Mat3D]=optimal_flatnorm_fits(FlatInputCell);
       [ComIm,ZeroVec,ScaleVec,ComImNotNaN]=imcombine(Mat3D,...
					       'Method',InPar.Combine,...
					       'Zero','none',...
					       'Scale','none',...
					       'Weight','none',...
					       'Reject',InPar.Reject,...
					       'RejPar',InPar.RejPar,...
					       'Niter',InPar.Niter);
      %--- Write StD image ---
      if (isempty(InPar.StD)),
         % do not calculate StD image
      else
 	 StD = std(Mat3D,[],3);
         fitswrite(StD,InPar.StD);
      end
    otherwise
       [ComIm,ZeroVec,ScaleVec]=imcombine_fits([],FlatInputCell,[], ...
				        'Method',InPar.Combine,...
                                        'Zero','none',...
    				        'Scale',InPar.PreScale,...
                                        'Weight','none',...
                                        'Reject',InPar.Reject,...
                                        'RejPar',InPar.RejPar,...
                                        'Niter',InPar.Niter);

       %--- Calculate StD image ---
       if (isempty(InPar.StD)),
          % do not calculate StD image
       else
          % calculate StD image
          imcombine_fits(InPar.StD,FlatInputCell,[],...
                     'Scale',InPar.NormFlat,...
		     'Zero','none',...
		     'Weight','none',...
		     'Method','StD');

       end
   end

   % delete makestar_*
   delete(sprintf('%s*',MaskStars_Prefix));

   % calculate unnormalized Flat statistics
   %[Stat,InputCell,OutMat]=imstat_fits(FlatOutput);

   switch lower(InPar.NormFlat)
    case 'mean'
       %Norm = Stat.Mean;
       Norm = mean(ComIm(:));
    case 'median'
       %Norm = Stat.Median;
       Norm = median(ComIm(:));
    case 'mode'
       %Norm = Stat.Mode;
       error('not supported');
    otherwise
       error('Unknown NormFlat option');
   end

   % normalize the flat image
   ComIm = ComIm./Norm;
   fitswrite(ComIm,FlatOutput);
   %imarith_fits(FlatOutput,FlatOutput,'/',Norm);

   %--- calculate Flat field statistics ---
   [SuperFlatStat,InputCell,OutMat]=imstat_fits(FlatOutput);



   %--- Calculate Sum image ---
   if (isempty(InPar.Sum)),
      % do not calculate Sum image
   else
      % calculate Sum image

      imcombine_fits(InPar.Sum,FlatInputCell,[],...
                     'Scale','none',...
		     'Zero','none',...
		     'Weight','none',...
		     'Method','sum');

   end

   %--- calculate NonLinearity images ---
   if (isempty(InPar.NonLinear)),
      % do nothinh
   else
      error('NonLinear option not exist yet');
   end

   %--- Correct all the images by the super flat ---
   switch lower(InPar.FlatCor)
    case 'n'
       % do nothing - do not correct images
    case 'y'
       % divide ImInput by FlatOutput

       switch lower(InPar.CopyHead)
        case 'y'
           Header = 1;
        case 'n'
	   Header = 0;
        otherwise
	   error('Unknown CopyHead option');
       end

       % construct additional header information:
       IndHead = 1;
       AddHead{IndHead,1}    = 'HISTORY';
       AddHead{IndHead,2}    = '';
       AddHead{IndHead,3}    = 'Flat corrected image generated by flatset_fits.m';
       IndHead = IndHead + 1;
       AddHead{IndHead,1}    = 'HISTORY';
       AddHead{IndHead,2}    = '';
       AddHead{IndHead,3}    = sprintf('Flat image used: %s',FlatOutput);

       OutImageFileCell=imarith_fits(ImOutput,ImInput,'/',FlatOutput,...
				     'OutPrefix',InPar.OutPrefix,...
				     'OutDir',InPar.OutDir,...
				     'Divide0',InPar.Divide0,...
				     'DataType',InPar.DataType,...
				     'Header',Header,...
                                     'AddHead',AddHead);
    otherwise
       error('Unknown FlatCor option');
   end
end




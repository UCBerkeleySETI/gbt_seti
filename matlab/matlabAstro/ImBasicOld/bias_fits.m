function [OutImageFileCell,BiasOutput,SuperBiasStat]=bias_fits(ImInput,BiasInput,ImOutput,BiasOutput,varargin)
%-----------------------------------------------------------------------------
% bias_fits function                                                  ImBasic
% Description: Given a set of images with the same size,
%              look for the bias images, create 
%              a super bias image and subtract it from all the non-bias
%              images.
%              This function is obsolote: Use ImBasic toolbox.
% Input  : - List of input images (see create_list.m for details).
%            In case that the second input argument (BiasInput) is empty
%            than this is treated as a list of all images, including
%            science, flat and bias images.
%            In this case the program will try to identify the bias
%            images using various methods (described in identify_bias_fits.m).
%            The super bias will be subtracted from all the non-bias images.
%            Default is '*.fits'.
%            If empty matrix than use default.
%          - List of bias images (see create_list.m for details).
%            If empty list than the program will try to identify the bias
%            images using various methods.
%            If given then the program will assume these are all the bias
%            images and subtract the super bias for all the images in the
%            list given in the first input argument (ImInput).
%            Default is empty matrix.
%          - List of bias subtracted output images
%            (see create_list.m for details).
%            This list should have the same length as the list given by the
%            first input argument (ImInput).
%            If empty matrix than will use ImInput as the output list.
%            If empty matrix and if 'OutPrefix' is not specified than
%            will set 'OutPrefix' to 'b_'.
%          - String containing the name of the output super bias FITS
%            image. Default is 'SuperBias.fits'.
%            If empty use default.
%          * Arbitrary number of pairs of input parameters: keyword,value,...
%            Available keywords are:
%            'BiasSec'  - A string containing an overscan bias section
%                         keyword name in the FITS header.
%                         Alternatively this can be a vector
%                         containing [Xmin Xmax Ymin Ymax].
%                         If empty matrix than do not use overscan.
%                         Default is empty matrix.
%                         If given than will use get the median of the
%                         bias section to derive a float bias value.
%                         For each image this value will be added to
%                         a median subtracted version of the SuperBias.
%                         If SuperBias image can not be constructed
%                         because bias images are not present then
%                         will use only the overscan region (line by line).
%                         If a cell array of strings or a matrix
%                         of n by 4 then assume that there are multiple
%                         bias sections (i.e., multiple amplifiers).
%                         In this case the user need to specify the data
%                         section corresponding to each bias section
%                         in the 'DataSec' input keyword
%            'DataSec'  - A data section keyword name, or cell array of 
%                         keyword names, in the header
%                         or a matrix cintaining (in each line)
%                         [Xmin Xmax Ymin Ymax] of the data section
%                         corresponding to each one of the bias sections.
%                         This is used only if the 'BiasSec' is given.
%            'OutPrefix'- Add prefix before output image names,
%                         default is 'b_'.
%            'OutDir'   - Directory in which to write the output images,
%                         default is empty string (i.e., '').
%            'CopyHead' - Copy header from the original images to the
%                         bias subtracted images {'y' | 'n'}, default is 'y'.
%            'DataType' - Output data type (see fitswrite.m for options), 
%                         default is float32.
%            'StD'      - String containing the name of an optional
%                         FITS image of the StD of all the bias images.
%                         This may be useful to identify bad pixels.
%                         If empty matrix than the StD image will not
%                         be calculated. Default is empty matrix.
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
%            'SubBias'  - Subtract the bias image from all other images {'y' | 'n'},
%                         default is 'y'.
%            'IdBias'   - A cell array of additional parameters to pass to
%                         identify_bias_fits.m. Default is empty matrix.
%                         See identify_bias_fits.m. for details.
% Output  : - Cell array containing output image names.
%             This list may be different in length than the input images list.
%             For example, if the bias images are identified automatically,
%             the program will not subtract the super bias from those images.
%           - String containing the name of the super bias FITS image.
%             If empty matrix than SuperBias image was not constructed
%             due to lack of bias images.
%             In this case the StD image will also not be generated.
%             Note that even in case that overscan floating value is used then
%             the super bias image returned is NOT median subtracted.
%             If only overscan is used then SuperBias is not calculated.
%           - Structure containing statistics regarding the bias image.
%             This includes all the statistical properties returned by imstat_fits.m.
%             If empty matrix than SuperBias image was not constructed.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [OutImFCell,BiasOutput,SuperBiasStat]=bias_fits('*.fits',[],[],BiasOutput,'OutPrefix','b_','StD','BiasStd.fits');
% Reliable: 2
%-----------------------------------------------------------------------------

Def.OutPrefix  = 'b_';
Def.ImInput    = '*.fits';
Def.BiasInput  = [];
Def.ImOutput   = [];
Def.BiasOutput = 'SuperBias.fits';
if (nargin==0),
   ImInput    = Def.ImInput;
   BiasInput  = Def.BiasInput;
   ImOutput   = Def.ImOutput;
   BiasOutput = Def.BiasOutput;
elseif (nargin==1),
   BiasInput  = Def.BiasInput;
   ImOutput   = Def.ImOutput;
   BiasOutput = Def.BiasOutput;
elseif (nargin==2),
   ImOutput   = Def.ImOutput;
   BiasOutput = Def.BiasOutput;
elseif (nargin==3),
   BiasOutput = Def.BiasOutput;
else
   % do nothing
end

if (isempty(ImInput))
   ImInput = Def.ImInput;
end

% if ImOutput = [] and OutPrefix not specified than set OutPrefix to 'b_'
if (isempty(ImOutput) && isempty(find(strcmpi(varargin,'OutPrefix')==1,1))),
   OutPrefix = Def.OutPrefix;
end

if (isempty(ImOutput)),
   ImOutput = ImInput;
end

if (isempty(BiasOutput)),
   BiasOutput  = Def.BiasOutput;
end


% deal with varargin:
DefV.OutPrefix   = Def.OutPrefix;
DefV.OutDir      = '';
DefV.BiasSec     = [];
DefV.DataSec     = [];
DefV.CopyHead    = 'y';
DefV.DataType    = 'float32';
DefV.StD         = [];
DefV.Reject      = 'none';
DefV.RejPar      = [];
DefV.Niter       = 1;
DefV.SubBias     = 'y';
DefV.IdBias      = {};
InPar = set_varargin_keyval(DefV,'y','use',varargin{:});


% Generate list from ImInput:
[~,ImInputCell] = create_list(ImInput,NaN);

if (isempty(BiasInput)),
   %--- Attempt to look for the bias images automatically ---

  % identify_bias_fits.m will attempt to identify bias images
  % among images of different types.
  [BiasInputCell,OutImageFileCell] = identify_bias_fits(ImInputCell,InPar.IdBias{:});


else
   % Assume all the input images are bias images
   [~,BiasInputCell] = create_list(BiasInput,NaN);
end

Nbias = length(BiasInputCell);

% Generate list from OutInput:
[~,ImOutputCell] = create_list(ImOutput,NaN);

%--- Construct a super bias image ---
if (Nbias==0),
   BiasOutput = [];
else
   [ComIm,ZeroVec,ScaleVec]=imcombine_fits(BiasOutput,BiasInputCell,[], ...
                                        'Method','median',...
                                        'Zero','none',...
				                        'Scale','none',...
                                        'Weight','none',...
                                        'Reject',InPar.Reject,...
                                        'RejPar',InPar.RejPar,...
                                        'Niter',InPar.Niter);
end

%--- Calculate StD image ---
if (Nbias==0),
   % Can't construct StD image since no bias images
else
   if (~isempty(InPar.StD)),
      [StdIm]     = imcombine_fits(InPar.StD,BiasInputCell,[], ...
                                'Method','std',...
                                'Zero','none',...
			                	'Scale','none',...
                                'Weight','none',...
                                'Reject',InPar.Reject,...
                                'RejPar',InPar.RejPar,...
                                'Niter',InPar.Niter);
   end
end


%--- Calculate super bias statistics ---
if (nargout>2 & Nbias>0),
   [SuperBiasStat] = imstat_fits(ComIm,'ImType','array');
else
   SuperBiasStat = [];
end

%--- Use overscan bias ---
if (isempty(InPar.BiasSec)),
   % do not use overscan - BiasSec is not specified
else
   % BiasSec is specified
   % use overscan bias

   % do nothing - this is handeled only if bias is subtracted from images.
end

%--- Subtract bias image from all the other images ---
switch lower(InPar.SubBias)
 case 'y'
    switch lower(InPar.CopyHead)
     case 'y'
        Header = 1;   % will copy header from original image
     case 'n'
        Header = 0;   % will not copy header from original image
     otherwise
        error('Unknwon CopyHead option');
    end
    
    % construct additional header information:
    IndHead = 1;
    AddHead{IndHead,1}    = 'HISTORY';
    AddHead{IndHead,2}    = '';
    AddHead{IndHead,3}    = 'Bias Subtracted image generated by bias_fits.m';
    IndHead = IndHead + 1;
    AddHead{IndHead,1}    = 'HISTORY';
    AddHead{IndHead,2}    = '';
    AddHead{IndHead,3}    = sprintf('Bias image used: %s',BiasOutput);

    if (Nbias>0 & isempty(InPar.BiasSec)),
       % use only SuperBias image    
       OutImageFileCell = imarith_fits1(ImOutputCell,ImInputCell,'-',BiasOutput,...
                                    'DataType',InPar.DataType,...
                                    'OutPrefix',InPar.OutPrefix,...
                                    'OutDir',InPar.OutDir,...
                                    'Header',Header,...
                                    'AddHead',AddHead);
    elseif (Nbias>0 & ~isempty(InPar.BiasSec)),
       % use both SuperBias and overscan region

       error('Bias overscan option is not available yet');       

%       subtract_overscanfloat_fits - given list of images and a super bias
%          subtract a median from the super bias and for add the median floating overscan bias
%          value of each image - than subtract from each image its corresponding bias image.
%       OutImageFileCell = subtract_overscanfloat_fits(ImOutputCell,ImInputCell,SuperBias,InPar.BiasSec,InPar.DataSec);

    elseif (Nbias==0 & ~isempty(InPar.BiasSec)),
       % use only overscan region

       error('Bias overscan option is not available yet');

%       subtract_overscanline_fits - subtract the bias section line by line from each FITS image
%          and deal with multiply amplifiers:
%       OutImageFileCell = subtract_overscanline_fits(ImOutputCell,ImInputCell,InPar.BiasSec,InPar.DataSec);


    else
       % can not subtract bias image from images - no bias information
       warning('No Bias information');
    end

 case 'n'
    % do not subtract bias image
 otherwise
    error('Unknown SubBias option');
end



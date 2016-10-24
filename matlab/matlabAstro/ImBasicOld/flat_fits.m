function [OutImageFileCell,FlatOutputFilter,SuperFlatStat,GroupIm,GroupFlat]=flat_fits(ImInput,FlatInput,ImOutput,FlatOutput,varargin);
%-----------------------------------------------------------------------------
% flat_fits function                                                  ImBasic
% Description: Given a set of images with the same size,
%              construct a flat-field image for each filter
%              and divide all the images of a given filter by the
%              corresponding normalized flat field image.
%              The script can look for twilight or dome flat images or
%              alternatively construct a super flat image
%              (see superflat_fits.m for details). Note that superflat_fits.m
%              can remove stars from the images prior to creation of the
%              flat image.
%              This function is obsolote: Use ImBasic toolbox.
% Input  : - List of input images (see create_list.m for details).
%            In case that the second input argument (FlatInput) is empty
%            than this is treated as a list of all images, including
%            science, flat and bias images.
%            In this case the program will try to identify the flat
%            images using various methods (described in identify_flat_fits.m).
%            All the images will be divided by the normalized super flat
%            image. Default is '*.fits'.
%            If empty matrix than use default.
%          - List of flat images (see create_list.m for details).
%            If empty list and FlatType='flat' then the program will
%            try to identify the flat images using various methods.
%            If given then the program will assume these are all the flat.
%            If empty list and FlatType='super' then the program will
%            use all the images to construct the super flat.
%            Default is empty matrix.
%          - List of flat divided output images
%            (see create_list.m for details).
%            This list should have the same length as the list given by the
%            first input argument (ImInput).
%            If empty matrix than will use ImInput as the output list.
%            If empty matrix and if 'OutPrefix' is not specified than
%            will set 'OutPrefix' to 'f_'.
%          - String containing the name of the output super flat FITS
%            image. Default is 'SuperFlat_<Filter>.fits'.
%            Where <Filter> is the filer name taken from the FITS header.
%            If empty use default.
%          * Arbitrary number of pairs of input parameters: keyword,value,...
%            Available keywords are:
%            'FlatType' - {'super', 'flat'}, default is 'flat'.
%                         'super' will cinsurct super flat field using
%                         all the images while 'flat' will use only images
%                         of type 'flat'.
%                         If the FlatType='super' option is used than all the images
%                         will be divided by the super flat.
%                         If FlatType='flat' option is used than all the images
%                         except the individual flat images will be divided by
%                         the super flat.
%            'Method'   - Indicating if the program masks stars in the flat images
%                         before creating the super flat {'norm','mask'}.
%                         If 'FlatType'='flat' then default is 'norm'.
%                         If 'FlatType'='super' then default is 'mask'.
%            'OutPrefix'- Add prefix before output image names,
%                         default is 'f_'.
%            'OutDir'   - Directory in which to write the output images,
%                         default is empty string (i.e., '').
%            'Filter'   - String or cell array of strings containing name of
%                         Filter header keyword based on which to
%                         identify images taken with different filters.
%                         If more than one keyword is specified than the program
%                         will use the first keyword that appear in the header.
%                         Default is {'FILTER', 'FILTER1'}.
%                         If empty matrix (i.e., []), then assume all images
%                         where taken in the same filter.
%            'CopyHead' - Copy header from the original images to the
%                         bias subtracted images {'y' | 'n'}, default is 'y'.
%            'DataType' - Output data type (see fitswrite_my.m for options), 
%                         default is float32.
%            'StD'      - String containing the name of an optional
%                         FITS image of the StD of all the flat images.
%                         This may be useful to identify bad pixels.
%                         If empty matrix than the StD image will not
%                         be calculated. Default is empty matrix.
%                         If string contains %s then it will be replaced
%                         by the filter name.
%            'Sum'      - String containing the name of an optional
%                         FITS image of the Sum of all the flat images.
%                         If empty matrix than the Sum image will not
%                         be calculated. Default is empty matrix.
%                         If string contain %s then it will be replaced
%                         by the filter name.
%            'NonLinear'- Not exist yet.
%                         String containing the name of an optional
%                         FITS image which indicate the amount of
%                         non-linearity in the response of each pixel.
%                         .... define...
%                         If empty matrix than the non-linearity image will not
%                         be calculated. Default is empty matrix.
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
%            'FlatCor'  - Divide all the images by the super flat image {'y' | 'n'},
%                         default is 'y'.
%            'ImTypeKey'- Image type keywords in header to be used by identify_flat_fits.m
%                         (e.g., {'IMTYPE','IMAGETYP','TYPE'}).
%                         See default and description in identify_flat_fits.m 
%            'ImTypeVal'- Image type value to search for,
%                         (e.g., {'flat','flatfield','ff','twflat','twilight'}).
%                         See default and description in identify_flat_fits.m 
%            'ByName'   - If not empty matrix then will attempt to look 
%                         for flat images by looking for a specific string,
%                         given following this keyword, in the image name.
%                         See default and description in identify_flat_fits.m 
%            'NotFlat'  - A cell array of strings containing image type
%                         values which are not flat images.
%                         See default and description in identify_flat_fits.m 
%            'OnlyCheck'- {'y' | 'n'}, If 'y' then will
%                         skip the flat image search and will assume
%                         that all the images in the input are flat images
%                         and check for saturation (if CheckSat='y').
%                         See default and description in identify_flat_fits.m 
%            'CheckSat' - Check if Flat image is saturated {'y' | 'n'}.
%                         See default and description in identify_flat_fits.m 
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
%            'Verbose'  - Print processing progress {'y' | 'n'}.
%                         Default is 'y'.
% Output  : - Cell array in which each cell correspond to one Filter
%             group and containing a cell array of output image names.
%             This list may be different in length than the input images list.
%             For example, if the flat images are identified automatically,
%             the program will not divide the flat images by the super flat image.
%           - Cell array of string containing the name of the super
%             flat FITS image. each cell corresponds to a filter.
%             If empty matrix than SuperFlat image was not constructed
%             due to lack of flat images.
%             In this case the StD and NonLinear images will also not be generated.
%           - Cell array of structure containing statistics
%             regarding the flat image. each cell corresponding to a
%             different filter.
%             This includes all the statistical properties returned by imstat_fits.m.
%             If empty matrix than SuperFlat image was not constructed.
%           - Group structure for the images
%             (see keyword_grouping1_fits.m for details).
%           - Group structure for the flats
%             (see keyword_grouping1_fits.m for details).
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: 
% Reliable: 2
%-----------------------------------------------------------------------------

Def.OutPrefix  = 'f_';
Def.ImInput    = '*.fits';
Def.FlatInput  = [];
Def.ImOutput   = [];
Def.FlatOutput = 'SuperFlat%s.fits';
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

if (isempty(ImInput))
   ImInput = Def.ImInput;
end

if (isempty(ImOutput)),
   ImOutput = ImInput;
end

if (isempty(FlatOutput)),
   FlatOutput  = Def.FlatOutput;
end


% deal with varargin:
DefV.FlatType    = 'flat';
DefV.Method      = [];  % will set based on the final value of 'FlatType'
DefV.OutPrefix   = Def.OutPrefix;
DefV.OutDir      = '';
DefV.Filter      = {'FILTER', 'FILTER1'};
DefV.CopyHead    = 'y';
DefV.DataType    = 'float32';
DefV.StD         = [];
DefV.Sum         = [];
DefV.NonLinear   = [];
DefV.Divide0     = 0;
DefV.Reject      = 'none';
DefV.RejPar      = [];
DefV.Niter       = 1;
DefV.FlatCor     = 'y';
% parameters for maskstars_fits.m
DefV.DETECT_THRESH  = [];
DefV.DETECT_MINAREA = [];
DefV.Verbose     = 'y';
% parameters for identify_flat_fits.m
DefV.ImTypeKey   = [];
DefV.ImTypeVal   = [];
DefV.ByName      = [];
DefV.NotFlat     = [];
DefV.OnlyCheck   = [];
DefV.CheckSat    = [];
DefV.SatLevel    = [];
DefV.SatFrac     = [];
DefV.SatKey      = [];

InPar = set_varargin_keyval(DefV,'n','def',varargin{:});

% if ImOutput = [] and OutPrefix not specified than set OutPrefix to 'f_'
if (isempty(ImOutput) && isempty(find(strcmpi(varargin,'OutPrefix')==1,1))),
   InPar.OutPrefix = Def.OutPrefix;
end

% If 'FlatType'='flat' then default is 'norm'.
% If 'FlatType'='super' then default is 'mask'.
if (isempty(InPar.Method)),
   switch lower(InPar.FlatType)
    case 'flat'
       InPar.Method = 'norm';
    case 'super'
       InPar.Method = 'mask';
    otherwise
       error('Unknown option for FlatType keyword');
   end
end

% set InPar.Method back into varargin
Imet = find(strcmpi(varargin(:),'method'));
if (~isempty(Imet)),
   varargin{Imet+1} = InPar.Method;
end

if (~iscell(InPar.Filter)),
   InPar.Filter = {InPar.Filter};
end

% Generate list from ImInput:
[~,ImInputCell] = create_list(ImInput,NaN);

% Identify flat images
if (isempty(FlatInput)),
   %--- Attempt to look for the flat images automatically ---

   switch lower(InPar.FlatType)
    case 'flat'
       % If FlatInput empty list and FlatType='flat' then the program will
       % try to identify the flat images using various methods.
       % also regenerate ImInputCell (without the flats)
       [ImFlatCell,ImInputCell] = identify_flat_fits(ImInputCell,varargin{:});
      
    case 'super'
       % If FlatInput empty list and FlatType='super' then the program will
       % use all the images to construct the super flat.
       ImFlatCell = ImInputCell;
      
    otherwise
       error('Unknown FlatType option');
   end
else
   % If FlatInput given then the program will assume these are all the flat.
   % Generate list of flats
   [~,ImFlatCell] = create_list(FlatInput,NaN);
end


%--- Group by filters ---
% Need to group both ImInputCell and FlatInput

if (isempty(InPar.Filter)),
   GroupIm(1).List  = ImInputCell;
   GroupIm(1).Group = '';
   GroupIm(1).Ind   = [1:1:length(ImInputCell)];

   GroupFlat(1).List  = ImInputCell;
   GroupFlat(1).Group = '';
   GroupFlat(1).Ind   = [1:1:length(ImInputCell)];
   
else
   [GroupIm,GroupStr,GroupInd]           = keyword_grouping1_fits(ImInputCell,InPar.Filter);
   [GroupFlat,GroupFlatStr,GroupFlatInd] = keyword_grouping1_fits(ImFlatCell,InPar.Filter);
end

Ngr = length(GroupIm);

%--- For each filter group ---
for Igr=1:1:Ngr,
   % check that the images groups have corresponding flat groups
   IgF = find(strcmp({GroupFlat(:).Group}, GroupIm(Igr).Group)==1);

   if (isempty(IgF)),
      fprintf(sprintf('Images filter group (%s) does not have corresponding Flat filter group\n',GroupIm(Igr).Group));
      fprintf(sprintf('%s-band images will not corrected for flat\n',GroupIm(Igr).Group));
   else
      % GroupIm(Igr).List   - list of images
      % GroupIm(Igr).Group  - Filter name
      % GroupFlat(IgF).List - list of flats
      FlatOutputFilter{Igr} = sprintf(FlatOutput,GroupIm(Igr).Group);

      % Add filter name to StD name
      if (~isempty(InPar.StD)),
         Istd = find(strcmpi(varargin,'StD')==1);
         varargin{Istd+1} = sprintf(sprintf('%s',InPar.StD),GroupIm(Igr).Group);
      end

      % Add filter name to Sum name
      if (~isempty(InPar.Sum)),
         Isum = find(strcmpi(varargin,'Sum')==1);
         varargin{Isum+1} = sprintf(sprintf('%s',InPar.Sum),GroupIm(Igr).Group);
      end

      if (isempty(ImOutput)),
         ImOutputCellFilter    = [];
      else
         % Generate list from FlatInput:
         [~,ImOutputCell] = create_list(ImOutput,NaN);
         ImOutputCellFilter = ImOutputCell(GroupIm(Igr).Ind);
      end

      switch lower(InPar.Verbose)
       case 'y'
         fprintf('flat_fits.m processing filter: %s\n',GroupIm(Igr).Group);
         fprintf('            Number of images: %d\n',length(GroupIm(Igr).List));
         fprintf('            Number of flats: %d\n',length(GroupFlat(IgF).List));
         fprintf('            Flat image name: %s\n',FlatOutputFilter{Igr});

       otherwise
          % do nothing
      end

      %--- construct Flat and correct per each filter ---
      [OutImageFileCell{Igr},FlatOutputFilter{Igr},SuperFlatStat{Igr}] = flatset_fits(GroupIm(Igr).List,...
                                                                        GroupFlat(IgF).List,...
                                                                        ImOutputCellFilter,...
		   						        FlatOutputFilter{Igr},...
                                                                        varargin{:});
   end
end



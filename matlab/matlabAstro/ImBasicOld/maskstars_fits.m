function [OutImCell,Stat,OutMat]=maskstars_fits(ImInput,ImOutput,Method,varargin);
%-----------------------------------------------------------------------------
% maskstars_fits function                                             ImBasic
% Description: Given a list of FITS images look for stars in these images
%              or pixels with high value above background and replace these
%              pixels by NaN or another value.
% Input  : - List of input images (see create_list.m for details).
%            Default is '*.fits'. If empty matrix then use default.
%          - List of output images (see create_list.m for details).
%            If empty matrix than use the input list.
%            Default is empty matrix.
%          - Maksing method - options are:
%            'sex'  - Find stars using the default settings of SExtractor
%                     (see run_sextractor.m for details). Default.
%            'perc' - Mask pixels which value is found a the top percentile
%                     of the pixels in an image.
%            'val'  - Mask pixels which value is above a given value.
%          * Arbitrary number of pairs of optional input arguments.
%            ...,keyword,value,... options are:
%            'DETECT_THRESH'  - Detection threshold to pass to SExtractor.
%                               Default is 1.5.
%            'DETECT_MINAREA' - Dectection minimum area to pass to SEx.
%                               Default is 5.
%            'Perc'           - Vercentile level to use with Method='perc'.
%                               Default is 0.05;
%            'Val'            - Value level to use with Method='val'.
%                               Default is 30000;
%            'OutPrefix'      - Add prefix before output image names,
%                               default is empty string (i.e., '').
%            'OutDir'         - Directory in which to write the output images,
%                               default is empty string (i.e., '').
%            'CopyHead'       - Copy header from original image {'y' | 'n'}, 
%                               default is 'y'.
%            'Replace'        - Value by which to replace pixels which
%                               are above the threshold. Default is NaN.
%            'DataType'       - Output data type (see fitswrite.m for options),
%                               default is float32.
%            'Save'           - Save FITS image to disk {'y' | 'n'}.
%                               Default is 'y'.
% Output : - Cell array containing list of masked output image file names.
%          - Vector of structures of optional statistics on
%            each output image. Structure fields are:
%            .Val   - Threshold value
%                     (if not used, will list the default value).
%            .Perc  - Percentile threshold
%                     (if not used, will list the default value).
%            .AreaFrac - Fraction of area which was replaced.
%            See imstat_fits.m for deatils.
%          - Cell array of matrices containing output images.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%-----------------------------------------------------------------------------

Def.ImInput                = '*.fits';
Def.ImOutput               = [];
Def.Method                 = 'sex';
Def.varargin.DETECT_THRESH = 1.5;
Def.varargin.DETECT_MINAREA= 5;
Def.varargin.Perc          = 0.05;
Def.varargin.Val          = 30000;
Def.varargin.OutPrefix    = '';
Def.varargin.OutDir       = '';
Def.varargin.CopyHead     = 'y';
Def.varargin.Replace      = NaN;
Def.varargin.DataType     = 'float32';
Def.varargin.Save         = 'y';
if (nargin==0),
   ImInput     = Def.ImInput;
   ImOutput    = Def.ImOutput;
   Method      = Def.Method;
elseif (nargin==1),
   ImOutput    = Def.ImOutput;
   Method      = Def.Method;
elseif (nargin==2),
   Method      = Def.Method;
else
   % do nothing
end

DefV.DETECT_THRESH  = Def.varargin.DETECT_THRESH;
DefV.DETECT_MINAREA = Def.varargin.DETECT_MINAREA;
DefV.Perc           = Def.varargin.Perc;
DefV.Val            = Def.varargin.Val;
DefV.OutPrefix      = Def.varargin.OutPrefix;
DefV.OutDir         = Def.varargin.OutDir;
DefV.CopyHead       = Def.varargin.CopyHead;
DefV.Replace        = Def.varargin.Replace;
DefV.DataType       = Def.varargin.DataType;
DefV.Save           = Def.varargin.Save;

InPar = set_varargin_keyval(DefV,'y','def',varargin{:});


[~,ImInputCell]  = create_list(ImInput,NaN);
if (isempty(ImOutput)),
   ImOutput = ImInput;
end
[~,ImOutputCell] = create_list(ImOutput,NaN);
if (isnumeric(ImInputCell{1}))
   IsNumeric = 1;
else
   IsNumeric = 0;
end

Nim = length(ImInputCell);

for Iim=1:1:Nim,
   % for each input image

   switch IsNumeric
    case 0
       CurImage = fitsread(ImInputCell{Iim});
    case 1
       CurImage = ImInputCell{Iim};
       % Need to re-write the FITS image so SExtractor can read it...
       ImInputCell{Iim} = tempname;
       fitswrite(CurImage,ImInputCell{Iim});

    otherwise
       error('Unknwon IsNumeric option');
   end

   switch lower(Method)
    case 'sex'
       % Mask pixels containing stars

       % Runs SExtractor and output the segmentation image
       TmpSegIm = sprintf('%s.fits',tempname);
       [CatSex,OutParam] = run_sextractor(ImInputCell{Iim},...
                               [],[],[],[],...
			       'DETECT_THRESH',sprintf('%f',InPar.DETECT_THRESH),...
			       'DETECT_MINAREA',sprintf('%d',InPar.DETECT_MINAREA),...
			       'CHECKIMAGE_TYPE','SEGMENTATION',...
			       'CHECKIMAGE_NAME',TmpSegIm);
       SegIm = fitsread(TmpSegIm);   % read segmentation image
       delete(TmpSegIm);             % delete segmentation FITS file

       % find all star pixels in segmentation image
       I           = find(SegIm>0);
       CurImage(I) = InPar.Replace;

    case 'perc'
       % Mak pixels which value is found above a give percentile
       % of the pixels in an image.
       % find the Perc percentile:
       InPar.Val        = prctile(mat2vec(CurImage),(1-InPar.Perc).*100);
       % replace all values above Val by NaNs
       I = find(CurImage>InPar.Val);
       CurImage(I) = InPar.Replace;

    case 'val'
       % Mask pixels which value is above a given value.

       % replace all values above Val by NaNs
       I = find(CurImage>InPar.Val);
       CurImage(I) = InPar.Replace;

    otherwise
       error('Unknown Method option');
   end

   %--- store statistics ---
   if (nargout>1),

     if (isnan(InPar.Replace)),
        AreaFrac = length(find(isnan(CurImage)))./prod(size(CurImage));
     else
        AreaFrac = length(find(CurImage==InPar.Replace))./prod(size(CurImage));
     end
     Stat(Iim).Val   = InPar.Val;
     Stat(Iim).Perc  = InPar.Perc;
     Stat(Iim).AreaFrac = AreaFrac;
   end

   %--- Write Ouput Image ---
   switch lower(InPar.CopyHead)
    case 'y'
       Head = fitsinfo(ImInputCell{Iim});
       HeaderInfo = Head.PrimaryData.Keywords;
       IndHead = size(HeaderInfo,1) + 1;
    case 'n'
       HeaderInfo = [];
       IndHead = 1;
    otherwise
       error('Unknown CopyHead option');
   end

   %--- Write Info about masking process to header ---
   HeaderInfo{IndHead,1}   = 'HISTORY';
   HeaderInfo{IndHead,2}   = '';
   HeaderInfo{IndHead,3}   = 'Pixel values were repaced by maskstars_fits.m';
   IndHead = IndHead + 1;
   HeaderInfo{IndHead,1}   = 'HISTORY';
   HeaderInfo{IndHead,2}   = '';
   HeaderInfo{IndHead,3}   = sprintf('maskstars_fits.m Method: %s',Method);
   IndHead = IndHead + 1;
   HeaderInfo{IndHead,1}   = 'HISTORY';
   HeaderInfo{IndHead,2}   = '';
   HeaderInfo{IndHead,3}   = sprintf('maskstars_fits.m Replace value: %f',InPar.Replace);

   OutImageFileName = sprintf('%s%s%s',InPar.OutDir,InPar.OutPrefix,ImOutputCell{Iim});
   OutImCell{Iim} = OutImageFileName;

   switch lower(InPar.Save)
    case 'y'
       fitswrite(CurImage, OutImageFileName, HeaderInfo, InPar.DataType);
    case 'n'
       % do not save FITS image
    otherwise
      error('Unknown Save option');
   end

   if (nargout>2),
      OutMat{Iim} = CurImage;
   end

   switch IsNumeric
    case 1
       % delete temporary FITS image
       delete(ImInputCell{Iim});
    otherwise
       % do nothing
   end
end


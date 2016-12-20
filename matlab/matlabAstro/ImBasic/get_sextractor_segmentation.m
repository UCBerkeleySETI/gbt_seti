function [ImOutputCell,OutMat,CatSex,OutParam,StructParam]=get_sextractor_segmentation(ImInput,ImOutput,varargin)
%------------------------------------------------------------------------------
% get_sextractor_segmentation function                                 ImBasic
% Description: Given a list of FITS images create segmentation FITS images
%              for each one of them using SExtractor.
% Input  : - List of input images (see create_list.m for details).
%            Default is '*.fits'. If empty matrix then use default.
%          - List of output images (see create_list.m for details).
%            If empty matrix than use the input list.
%            Default is empty matrix.
%          * Arbitrary number of pairs of optional input arguments.
%            ...,keyword,value,... options are:
%            'DETECT_THRESH'  - Detection threshold to pass to SExtractor.
%                               Default is 1.5.
%            'DETECT_MINAREA' - Dectection minimum area to pass to SEx.
%                               Default is 5.
%            'ANALYSIS_THRESH'- Analysis threshold to pass to SExtractor.
%                               Default is 1.5.
%            'DEBLEND_NTHRESH'- Number of delending, default is 4.
%            'DEBLEND_MINCONT'- Deblending parameter, default is 0.005.
%            'OutPrefix'      - Add prefix before output image names,
%                               default is empty string (i.e., '').
%            'OutDir'         - Directory in which to write the output images,
%                               default is empty string (i.e., '').
%            'Save'           - Save FITS image to disk {'y' | 'n'}.
%                               Default is 'y'.
% Output : - Cell array containing list of segmentation image names.
%          - Cell array of matrices containing output images.
%          - Output SExtractor catalog (see run_sextractor for details).
%          - Cell array of SExtractor output parameters.
%          - Structure in which the field names are the output parameters
%            followed by their column index.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                    Sep 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%------------------------------------------------------------------------------

Def.ImInput              = '*.fits';
Def.ImOutput             = [];
DefV.DETECT_THRESH       = 1.5;
DefV.ANALYSIS_THRESH     = 1.5;
DefV.DETECT_MINAREA      = 5;
DefV.DEBLEND_NTHRESH     = 4;
DefV.DEBLEND_MINCONT     = 0.005;
DefV.OutPrefix           = '';
DefV.OutDir              = '';
DefV.Save                = 'y';
if (nargin==0),
   ImInput     = Def.ImInput;
   ImOutput    = Def.ImOutput;
elseif (nargin==1),
   ImOutput    = Def.ImOutput;
else
   % do nothing
end

InPar = set_varargin_keyval(DefV,'y','def',varargin{:});


[~,ImInputCell]  = create_list(ImInput,NaN);
if (isempty(ImOutput))
   ImOutput = ImInput;
end
[~,ImOutputCell] = create_list(ImOutput,NaN);

Nim = length(ImInputCell);

for Iim=1:1:Nim,
   % for each input image

   OutputImageFullName = sprintf('.%s%s%s%s%s',filesep,InPar.OutDir,filesep,InPar.OutPrefix,ImOutputCell{Iim});

   % Runs SExtractor and output the segmentation image
   TmpSegIm = sprintf('%s.fits',tempname);
   [CatSex,OutParam,StructParam] = run_sextractor(ImInputCell{Iim},...
                            [],[],[],[],...
                         'DETECT_THRESH',sprintf('%f',InPar.DETECT_THRESH),...
                         'ANALYSIS_THRESH',sprintf('%f',InPar.ANALYSIS_THRESH),...
			 'DETECT_MINAREA',sprintf('%d',InPar.DETECT_MINAREA),...
             'DEBLEND_NTHRESH',sprintf('%d',InPar.DEBLEND_NTHRESH),...
             'DEBLEND_MINCONT',sprintf('%6.4f',InPar.DEBLEND_MINCONT),...
			 'CHECKIMAGE_TYPE','SEGMENTATION',...
			 'CHECKIMAGE_NAME',OutputImageFullName);

   if (nargout>1),
      OutMat{Iim} = fitsread(OutputImageFullName);
   end

   switch lower(InPar.Save)
    case 'y'
       % do nothing
    case 'n'
       % delete segmentation image
       delete(OutputImageFullName);
    otherwise
       error('Unknown Save option');
   end
end


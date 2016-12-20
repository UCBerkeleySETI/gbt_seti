function [ComIm,ZeroVec,ScaleVec,ComImNotNaN]=imcombine_fits(OutImageFileName,InImages,Keys,varargin);
%-----------------------------------------------------------------------------
% imcombine function                                                  ImBasic
% Description: Combine FITS images.
%              This function use imcombine.m.
%              This function is obsolote: Use ImBasic toolbox.
% Input  : - String containing the name of the output FITS image.
%            If empty matrix then do not save FITS file.
%          - List of input FITS images (see create_list.m for details).
%          - A Cell array containg the name of the following keywirds in
%            the header of the input imags:
%            {ExpusreTime, JD}. If empty then keyword is not available.
%            The program uses thes keywords to calculate the total 
%            exposure time and the effective JD of observation.
%            Default is {'EXPTIME','JD'}.
%          * Arbitrary pais of parameters: keyword, value,...
%            'Method'    - combining method:
%                          {'sum' | 'mean' | 'median' | 'std'},
%                          default is 'median'.
%            'Zero'      - Image offset (to add before scale),
%                          {'mean' | 'median' | 'mode' | 'constant' | 'none'},
%                          default is 'none'.
%                          Note that the zero willbe applied before the
%                          scaling.
%            'ZeroConst' - A scalar or vector of a constant offsets to be
%                          applied to each image. Default is [1].
%                          This is being used only if zero=constant.
%            'Scale'     - Image scaling
%                          {'mean' | 'median' | 'mode' | 'constant' | 'none'},
%                          default is 'none'.
%            'ScaleConst'- A scalar or vector of a constant scale to be
%                          applied to each image. Default is [1].
%                          This is being used only if scale=constant.
%            'Weight'    - Method by which to wheight the images.
%                          {'mean' | 'median' | 'mode' | 'constant' | 'images' | 'none'},
%                          default is 'none'.
%            'WeightConst'-A scalar or vector of a constant weight to be
%                          applied to each image. Default is [1].
%                          This is being used only if weight=constant.
%            'WeightFun' - Function name to apply to the weight value
%                          before weighting (e.g., @sqrt).
%                          If empty matrix (i.e., []) than donot apply a
%                          function. Default is empty matrix.
%            'WeightIm'  - List of FITS images to use as weights
%                          (see create_list.m for details).
%            'Reject'    - Pixels rejection methods:
%                          'sigclip'   - std clipping.
%                          'minmax'    - minmax rejuction.
%                          'none'      - no rejuction, default.
%                          'sigclip' and 'minmax' are being applied only if
%                          Niter>1.
%            'RejPar'    - rejection parameters [low, high].
%                          for 'sigclip', [LowBoundSigma, HighBoundSigma].
%                          Default is [Inf Inf].
%                          for 'minmax', [Number, Number] of pixels to
%                          reject default is [0 0].
%            'Niter'     - Number of iterations in rejection scheme.
%                          Default is 1.
%            'RepNegWith'- Replace negative pixels with a fixed value prior
%                          to combine. If empty, do nothing. Default is
%                          empty.
%            'CCDSEC'    - CCD section [xmin xmax ymin ymax] to combine.
%                          If empty matrix then will use entire image.
%                          Default is empty matrix.
%            'CopyHead'  - Copy header from first image to combine
%                          {'y' | 'n'}, default is 'y'.
%            'Save'      - Save output FITS image {'y' | 'n'}.
%                          Default is 'y'.
% Output : - 2D matrix of combined image.
%          - Vector containing the zero offsets per image used.
%          - Vector containing the scales per image used.
%          - 2D matrix corresponding to the combined image, in which each
%            pixel his a counter of the number of not-NaN values in all
%            the images in the position of a given pixel.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                    Jun 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/                                        
% Example: [ComIm,ZeroVec,ScaleVec]=imcombine_fits('Out.fits','lred00[15-16].fits');
% Reliable: 2
%-----------------------------------------------------------------------------

Def.Keys = {'EXPTIME','EXPOSURE','JD'};
if (nargin==2),
   Keys = Def.Keys;
end

DefV.CopyHead = 'y';
DefV.CCDSEC   = [];

Ich = find(strcmpi(varargin,'CopyHead')==1);
if (isempty(Ich)),
    CopyHead = DefV.CopyHead;
else
    CopyHead = varargin{Ich+1};
end

Ich = find(strcmpi(varargin,'CCDSEC')==1);
if (isempty(Ich)),
    CCDSEC = DefV.CCDSEC;
else
    CCDSEC = varargin{Ich+1};
end

Iwi = find(strcmpi(varargin,'WeightIm')==1);
if (isempty(Iwi)),
   % No WeightIm input
else
   % Read WeightIm FITS images
   [~,CellWeightIm] = create_list(varargin{Iwi+1},NaN);

   % Read images
   Nwim = length(CellWeightIm);
   WeightIm = cell(Nwim,1);
   for Iwim=1:1:Nwim,
      WeightIm{Iwim} = fitsread(CellWeightIm{Iwim});
   end
   % Add WeightIm to varargin for imcombine.m
   varargin{Iwi+1} = WeightIm;
end

%--- Read Input Images ---
[~,CellInImages] = create_list(InImages,NaN);

% Read images
Nim = length(CellInImages);
H.ExpTime = zeros(Nim,1);
H.JD      = zeros(Nim,1);
InMat     = cell(1,Nim);
for Iim=1:1:Nim,
   InMat{1,Iim} = fitsread(CellInImages{Iim});
   if (isempty(CCDSEC)),
       % do nothing
   else
       InMat{1,Iim} = InMat{1,Iim}(CCDSEC(3):CCDSEC(4),CCDSEC(1):CCDSEC(2));
   end
   
   %--- Read keywords to combine ---
   KeywordVal = get_fits_keyword(CellInImages{Iim},Keys);

   for Ind=1:1:length(Keys),
      if (isempty(KeywordVal{Ind})),
         H.(Keys{Ind})(Iim) = NaN;
      else
         if (ischar(KeywordVal{Ind})),
	        H.(Keys{Ind})(Iim) = str2double(KeywordVal{Ind});
         else
	        H.(Keys{Ind})(Iim) = KeywordVal{Ind};
         end
      end
   end
end

if (nargout>3),
   
   [ComIm,ZeroVec,ScaleVec,ComImNotNaN] = imcombine(InMat,varargin{:});
else
   [ComIm,ZeroVec,ScaleVec] = imcombine(InMat,varargin{:});
end

%---Construct header ---
HeaderInfo = [];

%--- Add to header comments regarding file creation ---
MinJD = min(H.JD);
MaxJD = max(H.JD);
MidJD = 0.5.*(MinJD + MaxJD);
TotalExpTime = nansum(H.ExpTime);
WeightedJD = sum(H.JD.*H.ExpTime)./sum(H.ExpTime);

[HeaderInfo] = cell_fitshead_addkey(HeaderInfo,...
                                    Inf,'COMMENT','','Created by imcombine_fits.m written by Eran Ofek',...
                                    Inf,'HISTORY','',sprintf('Number of combined images: %d',Nim),...
                                    Inf,'MINJD',sprintf('%20.5f',MinJD),sprintf('Minimum JD of images calculated by imcombine.m'),...
                                    Inf,'MAXJD',sprintf('%20.5f',MaxJD),sprintf('Maximum JD of images calculated by imcombine.m'),...
                                    Inf,'MIDJD',sprintf('%20.5f',MidJD),sprintf('Mid JD calculated by imcombine.m'),...
                                    Inf,'WEIGHTJD',sprintf('%20.5f',WeightedJD),sprintf('Weighted by ExpTime JD calculated by imcombine.m'),...
                                    Inf,'TEXPTIME',sprintf('%20.5f',TotalExpTime),sprintf('Total ExpTime calculated by imcombine.m'));

switch lower(CopyHead)
    case 'y'
       OldHead = fitsinfo(CellInImages{1});
       HeaderInfo = [HeaderInfo; OldHead.PrimaryData.Keywords];
    otherwise
        % do nothing
end
%--- Write FITS image ---
Isave = find(strcmpi(varargin,'Save'));
if (isempty(Isave)),
   Save = 'y';
else
   Save = varargin{Isave+1};
end
switch lower(Save)
 case 'n'
    OutImageFileName = [];
 otherwise
    % do nothing
end

if (isempty(OutImageFileName)),
   % do not save FITS file
else
   fitswrite_my(ComIm,OutImageFileName,HeaderInfo);
end

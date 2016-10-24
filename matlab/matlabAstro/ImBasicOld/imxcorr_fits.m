function [OutImageFileCell,Shift,OutputMatrixCell]=imxcorr_fits(ImInput1,ImInput2,ImOutput,varargin);
%------------------------------------------------------------------------------
% imxcorr_fits function                                                ImBasic
% Description: Cross correlate two FITS images using FFT and find the
%              optimal linear shift between the two images.
%              Optionally, the program subtract the background from the
%              images before the cross-correlation step.
% Input  : - First list of input images (see create_list.m for details).
%            Alternatively, this can be a cell array in which each cell
%            contains the image in a matrix form, or a matrix.
%            In this case ImOutput can not be an empty matrix.
%          - Second list of reference input images (or a single image).
%            Alternatively, this can be a cell array in which each cell
%            contains the image in a matrix form, or a matrix.
%            These image/s will be used as reference.
%            (see create_list.m for details).
%            If single image is specified then it will be cross correlated
%            against every image in the first input list.
%            If multiple images are given then it should have the same
%            length as the first input list and each image will be cross
%            correlated against the corresponding image in the first
%            input list.
%            If empty matrix, then calculate the auto-correlation.
%            Default is empty matrix.
%          - List of output images (see create_list.m for details).
%            If empty matrix, then set it to be equal to the first
%            Input list. Default is empty matrix.
%            'Norm' - Normalizes the correlation according to one of
%                     the following options:
%                     'N'   - scales the raw cross-correlation by 1/N,
%                             where N is the number of elements in the first
%                             input matrix.
%                     '1'   - scales the raw cross-correlation so the
%                             correlation at lag zero is 1.
%                     'none'- no scaling (this is the default).
%            'SubBack'  - Subtract background before cross-correlation
%                         (see imsubback_fits.m for details).
%                         Options are: {@Fun | 'mode' | 'median' | 'mean' |
%                                       'medfilt' | 'ordfilt' | 'poly'}.
%                         Default is 'mode'.
%            'SubPar'   - Parameters for background subtraction
%                         algorithm.
%                         (see imsubback_fits.m for details).
%                         Example: {'MaskStar','sex','SubPar',{[3 3 2],'MaxNiter',3}}
%                         Default is {}.
%            'Conv1'    - Convolution parameters for the first image.
%                         If {}, then first image will not be convolved with
%                         a kernel before cross-correlation. Default is {}.
%                         This is a cell array containing all the parameters
%                         to be passed to imconv_fits.m.
%                         For example: {{@gauss_2d, 10, 3}}, or equivalently:
%                         {{@gauss_2d,10},'AddPar',{3}}. 
%            'Conv2'    - Same as Conv1, but for the second image.
%            'OutPrefix'- Add prefix before output image names,
%                         default is empty string (i.e., '').
%            'OutDir'   - Directory in which to write the output images,
%                         default is empty string (i.e., '').
%            'CopyHead' - Copy header from original image {'y' | 'n'}.
%                         Default is 'y'.
%            'AddHead'  - Cell array with 3 columns containing additional
%                         keywords to be add to the header.
%                         See cell_fitshead_addkey.m for header structure
%                         information. Default is empty matrix.
%            'DelDataSec'-Delete DATASEC keyword from image header {'y' | 'n'}.
%                         Default is 'n'.
%                         The reason for that the DATASEC keywords may
%                         caseuse problem in image display.
%            'DataType' - Output data type (see fitswrite.m for options), 
%                         default is float32.
%            'CCDSEC'   - Image sction for image to be rotated. If given
%                         then the image will be croped before rotation.
%                         This could be string containing image keyword 
%                         name (e.g., 'CCDSEC'), or a vector of 
%                         [Xmin, Xmax, Ymin, Ymax].
%             'Save'    - Save FITS image to disk {'y' | 'n'}.
%                         Default is 'y'.
% Output  : - Cell array containing output image names.
%           - Vector (per image) of structure containing the best shift
%             estimate between images. These are the shifts that need
%             to bes subtracted from the reference (second) image in order
%             to get to the reference frame of the first image.
%             The following fields are available:
%             .ShiftX
%             .ShiftY
%             .PeakCorr
%             .BestShiftX
%             .BestShiftY
%           - Cell array of matrices or output images.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: % cross correlate two random images:
%          [Out,Shift]=imxcorr_fits(rand(100,100),rand(100,100),'out.fits');
%          % example of shifted images:
%          ShiftX = 30.4;  ShiftY = 229.3;
%          Im1 = fitsread('ccd.144.0.fits');
%          Size = size(Im1);  X=[1:1:Size(2)];  Y=[1:1:Size(1)];
%          Im2 = interp2(X,Y,Im1,X+ShiftX,Y'+ShiftY);
%          I=find(isnan(Im2));  Im2(I)=0;
%          S1 = imstat_fits('ccd.144.0.fits');
%          Im1 = Im1 - S1.Mode;   Im2 = Im2 - S1.Mode;
%          [Out,Shift]=imxcorr_fits(Im1,Im2,'out.fits');
%          [Out,Shift]=imxcorr_fits(Im1,Im2,'out.fits','SubBack','mode','SubPar',{'MaskStar','sex'});
% Reliable: 2
%------------------------------------------------------------------------------

Def.ImInput2 = [];
Def.Output   = [];
if (nargin==1),
   ImInput2 = Def.ImInput2;
   Output   = Def.Output;
elseif (nargin==2),
   Output   = Def.Output;
else
   % do nothing
end

DefV.SubBack     = 'mode';
DefV.SubPar      = {};
DefV.Conv1       = {};
DefV.Conv2       = {};
DefV.Norm        = 'none';
DefV.OutPrefix   = '';
DefV.OutDir      = '';
DefV.DelDataSec  = 'n';
DefV.CopyHead    = 'y';
DefV.AddHead     = [];
DefV.DataType    = 'float32';
DefV.CCDSEC      = [];
DefV.Save        = 'y';

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

IsNumeric1 = 0;
if (isnumeric(ImInput1)),
   ImInput1 = {ImInput1};
end

if (iscell(ImInput1)),
   if (isnumeric(ImInput1{1})),
      % input is given in a matrix form
      IsNumeric1 = 1;
      if (isempty(ImOutput)),
	 error('If ImInput is numeric then must specify ImOutput');
      end
   end
end

if (isempty(ImInput2)),
   % auto correlation  - set to ImInput1:
   ImInput2   = ImInput1;
   IsNumeric2 = IsNumeric1;
else
   IsNumeric2 = 0;
   if (isnumeric(ImInput2)),
      ImInput2 = {ImInput2};
   end
   if (iscell(ImInput2)),
      if (isnumeric(ImInput2{1})),
         % input is given in a matrix form
         IsNumeric2 = 1;
      end
   end
end

if (IsNumeric1==0),
   [~,ImInputCell1] = create_list(ImInput1,NaN);
else
   ImInputCell1 = ImInput1;
end
Nim1 = length(ImInputCell1);

if (IsNumeric2==0),
   [~,ImInputCell2] = create_list(ImInput2,NaN);
else
   ImInputCell2 = ImInput2;
end
Nim2 = length(ImInputCell2);

if (isempty(ImOutput)),
   ImOutput = ImInput1;
end
[~,ImOutputCell] = create_list(ImOutput,NaN);


%--- Go over all images ---
for Iim1=1:1:Nim1,
   %--- read Image ImInput1 ---
   if (IsNumeric1==1),
      InputImage1 = ImInputCell1{Iim1};
   else
      InputImage1 = fitsread(ImInputCell1{Iim1});
   end

   %--- read Image ImInput2 ---
   if (Nim2==1),
      Iim2 = 1;     % second list conatins single image
   else
      Iim2 = Iim1;  % second list contains multiple images
   end

   if (IsNumeric2==1),
      InputImage2 = ImInputCell2{Iim2};
   else
      InputImage2 = fitsread(ImInputCell2{Iim2});
   end

   %--- CCDSEC for 1st image ---
   if (isempty(InPar.CCDSEC)),
      % use entire image
      % do nothing
   elseif (ischar(InPar.CCDSEC)),
      [InPar.CCDSEC] = get_ccdsec_fits({ImInputCell1{Iim1}},InPar.CCDSEC);
      [InputImage1]   = cut_image(InputImage1,InPar.CCDSEC,'boundry');
   elseif (length(InPar.CCDSEC)==4),
      [InputImage1]   = cut_image(InputImage1,InPar.CCDSEC,'boundry');
   else
      error('Illegal CCDSEC input');
   end

   %--- CCDSEC for 2nd image ---
   if (isempty(InPar.CCDSEC)),
      % use entire image
      % do nothing
   elseif (ischar(InPar.CCDSEC)),
      [InPar.CCDSEC] = get_ccdsec_fits({ImInputCell2{Iim2}},InPar.CCDSEC);
      [InputImage2]   = cut_image(InputImage2,InPar.CCDSEC,'boundry');
   elseif (length(InPar.CCDSEC)==4),
      [InputImage2]   = cut_image(InputImage2,InPar.CCDSEC,'boundry');
   else
      error('Illegal CCDSEC input');
   end

   OrigSize1 = size(InputImage1);  % original size after CCDSEC
   OrigSize2 = size(InputImage2);  % original size after CCDSEC

   %--- Remove background from images ---
   [~,InputImage1]=imsubback_fits(InputImage1,'tmp',InPar.SubBack,InPar.SubPar{:},'Save','n');
   [~,InputImage2]=imsubback_fits(InputImage2,'tmp',InPar.SubBack,InPar.SubPar{:},'Save','n');

   %--- Convolve images with kernel before cross-correlation ---
   if (isempty(InPar.Conv1)),
      % do nothing
   else
      % convolve 1st image
      [~,InputImage1]=imconv_fits(InputImage1,'tmp',InPar.Conv1{:},'Save','n');
   end

   if (isempty(InPar.Conv2)),
      % do nothing
   else
      % convolve 2nd image
      [~,InputImage2]=imconv_fits(InputImage2,'tmp',InPar.Conv2{:},'Save','n');
   end

   % convert cell back to matrix...
   if (iscell(InputImage1))
      InputImage1 = InputImage1{1};
   end
   if (iscell(InputImage2))
      InputImage2 = InputImage2{1};
   end

   %--- cross correlate images ---
   % very slow - using matlab xcorr2 function:
   %OutputImage = xcorr2(InputImage1,InputImage2);

   OutputImage = real(ifft2(fft2(InputImage1).*conj(fft2(InputImage2))));
%   OutputImage = fftshift(OutputImage);

   %--- Normalization ---
   switch lower(InPar.Norm)
    case 'none'
       % do nothing
    case '1'
       OutputImage = OutputImage./maxnd(OutputImage);
    case 'N'
       OutputImage = OutputImage./numel(OutputImage);
    otherwise
       error('Unknown Norm option');
   end

   NewSize = size(OutputImage);  % size of new image

   OutImageFileName = sprintf('%s%s%s',InPar.OutDir,InPar.OutPrefix,ImOutputCell{Iim1});
   OutImageFileCell{Iim1} = OutImageFileName;

   if (IsNumeric1==0),
      switch lower(InPar.CopyHead)
       case 'y'
          Info = fitsinfo(ImInputCell1{Iim1});
          HeaderInfo = Info.PrimaryData.Keywords;
       otherwise
          HeaderInfo = [];
      end
   else
      HeaderInfo = [];
   end

   if (IsNumeric1==1),
      InputName1 = 'matlab Matrix format';
   else
      InputName1 = ImInputCell1{Iim1};
   end

   if (IsNumeric2==1),
      InputName2 = 'matlab Matrix format';
   else
      InputName2 = ImInputCell2{Iim2};
   end

   %--- Add to header comments regarding file creation ---
   [HeaderInfo] = cell_fitshead_addkey(HeaderInfo,...
                                       Inf,'COMMENT','','Created by imconv_fits.m written by Eran Ofek',...
                                       Inf,'HISTORY','',sprintf('Original size input1: %d,%d',OrigSize1([2 1])),...
                                       Inf,'HISTORY','',sprintf('Original size input2: %d,%d',OrigSize2([2 1])),...
    			               Inf,'HISTORY','',sprintf('New size: %d,%d',NewSize([2 1])),...
    			               Inf,'HISTORY','',sprintf('XCorr normalization: %s',InPar.Norm),...
                                       Inf,'HISTORY','',sprintf('Input1 image name: %s',InputName1),...
                                       Inf,'HISTORY','',sprintf('Input2 image name: %s',InputName2));

   if (~isempty(InPar.AddHead)),
      %--- Add additional header keywords ---
      HeaderInfo = [HeaderInfo; InPar.AddHead];
   end

   switch lower(InPar.DelDataSec)
    case 'n'
        % do nothing
    case 'y'
        % delete DATASEC keyword from header
        [HeaderInfo] = cell_fitshead_delkey(HeaderInfo,'DATASEC');
    otherwise
        error('Unknown DelDataSec option');
   end

   %--- Write fits file ---
   switch lower(InPar.Save)
    case 'y'
       fitswrite(OutputImage,OutImageFileName,HeaderInfo,InPar.DataType);
    case 'n'
       % do not save FITS image
    otherwise
       error('Unknown Save option');
   end

   if (nargout>2),
      OutputMatrixCell{Iim1} = OutputImage;
   end

   %--- Shift data ---
   [Max,MaxInd] = maxnd(OutputImage);
   SizeOut      = size(OutputImage);
   %Shift(Iim1).ShiftX = MaxInd(2)-(SizeOut(2)).*0.5;
   %Shift(Iim1).ShiftY = MaxInd(1)-(SizeOut(1)).*0.5;
   if (MaxInd(2)>SizeOut(2).*0.5),
      Shift(Iim1).ShiftX = MaxInd(2) - SizeOut(2) - 1;
   else
      Shift(Iim1).ShiftX = MaxInd(2);
   end
   if (MaxInd(1)>SizeOut(1).*0.5),
      Shift(Iim1).ShiftY = MaxInd(1) - SizeOut(1) - 1;
   else
      Shift(Iim1).ShiftY = MaxInd(1);
   end
   
   Xr = [-1:0.1:1];
   Yr = [-1:0.1:1].';

   Local = interp2(OutputImage,MaxInd(2)+Xr,MaxInd(1)+Yr,'spline');
   [Max,MaxInd] = maxnd(Local);
   Shift(Iim1).BestShiftX  = Shift(Iim1).ShiftX + Xr(MaxInd(2));
   Shift(Iim1).BestShiftY  = Shift(Iim1).ShiftY + Yr(MaxInd(1));

   % fit a 2-D Gaussian

   % TO BE IMPLEMENTED

end



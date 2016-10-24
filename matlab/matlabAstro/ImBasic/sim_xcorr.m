function [Shift,OutSim]=sim_xcorr(varargin)
%--------------------------------------------------------------------------
% sim_xcorr function                                               ImBasic
% Description: Cross correlate two sets of images. The input can be any
%              type of image, but the output is always a structure array.
%              This program work on all images in memory and therefore
%              require sufficient memory.
% Input  : * Arbitrary number of pairs of input parameters: key,val,...
%            Available keywords are:
%            'In1'   - First operand.
%                      Multiple images in one of the following forms:
%                      (1) A structure array (SIM).
%                      (2) A string containing wild cards (see create_list.m).
%                      (3) A cell array of matrices (or scalars).
%                      (4) A file name that contains a list of files
%                          (e.g., '@list', see create_list.m)
%                      (5) A cell array of file names.
%                      This parameter must be provided.
%            'In2'   - Second operand. Like 'In1'.
%            'Norm' - Normalizes the correlation according to one of
%                     the following options:
%                     'N'   - scales the raw cross-correlation by 1/N,
%                             where N is the number of elements in the first
%                             input matrix.
%                     '1'   - scales the raw cross-correlation so the
%                             correlation at lag zero is 1.
%                     'none'- no scaling (this is the default).
%            'ImageField' - Field name in the SIM which contains the
%                      image on which to operate. Default is 'Im'.
%            'Shift'  - fftshift output image {tru|false}.
%                      Default is false.
%            'CCDSEC1' - CCD section in the first operand on which
%                      to operate. This can be either a vector of
%                      [xmin, xmax, ymin, ymax] or a header keyword name
%                      which contains the data section.
%                      If empty then will use the full image.
%                      Default is empty.
%            'CCDSEC2' - Like 'CCDSEC1' but for the second operand.
%            'HeadNum' - From which input operand to copy header {1 | 2}
%                        or to use minimal header {0}.
%                        Default is 1.
%            'AddHead' - Cell array with 3 columns containing additional
%                      keywords to be add to the header.
%                      See cell_fitshead_addkey.m for header structure
%                      information. Default is empty matrix.
%            'Verbose' - Print progress messages {true|false}.
%                      Default is false.
%            'OutSIM' - Output is a SIM class (true) or a structure
%                      array (false). Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by:
%            images2sim.m, sim_background.m, sim_trim.m
% Output  : - A structure array in which each element contains information
%             about the highest peak in the corresponding cross-correlation
%             image. The available fields are:
%             .ShiftX
%             .ShiftY
%             .PeakPeak
%             .BestShiftX
%             .BestShiftY
%           - Structure array of images (SIM) in which the cross-correlation
%             output images are stored.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Sh,O]=sim_xcorr('In1','lred0127.fits','In2','lred0127.fits');
% Reliable: 2
%-----------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';



%DefV.Output      = [];
DefV.In1         = [];  % must supply
DefV.In2         = [];  % must supply
DefV.Norm        = 'none';
DefV.Shift       = false;
DefV.ImageField  = ImageField;
DefV.CCDSEC1     = [];   % must match to CCDSEC2
DefV.CCDSEC2     = [];
DefV.HeadNum     = 1;
DefV.AddHead     = [];
DefV.CreateErr   = false;
DefV.Verbose     = false;
% images2sim.m parameters
DefV.R2Field     = ImageField;
DefV.R2Name      = FileField;
DefV.ReadHead    = true;
DefV.ImType      = 'FITS';
DefV.FitsPars    = {};
DefV.ImreadPars  = {};
DefV.OutSIM      = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

ImageField = InPar.ImageField;

% check validity of input
if (isempty(InPar.In1)),
    error('In1 must be supplied');
end
if (isempty(InPar.In2)),
    error('In2 must be supplied');
end

% read images
Sim1 = images2sim(InPar.In1,varargin{:});
InPar = rmfield(InPar,'In1');
Sim2 = images2sim(InPar.In2,varargin{:});
InPar = rmfield(InPar,'In2');


Nim1 = numel(Sim1);
Nim2 = numel(Sim2);


% trim images
Sim1 = sim_trim(Sim1,'TrimSec',InPar.CCDSEC1);
Sim2 = sim_trim(Sim2,'TrimSec',InPar.CCDSEC2);

% subtract background
Sim1 = sim_background(Sim1,varargin{:}, 'StoreBack',false,'SubBack',true);
Sim2 = sim_background(Sim2,varargin{:}, 'StoreBack',false,'SubBack',true);


%--- Perform operation on all images ---
Nim    = max(Nim1,Nim2);
Shift  = struct('ShiftX',cell(Nim,1),'ShiftY',cell(Nim,1),...
                'BestShiftX',cell(Nim,1),'BestShiftY',cell(Nim,1));
            
if (InPar.OutSIM),
    OutSim = simdef(Nim); %SIM;    % output is of SIM class
else
    OutSim = struct(ImageField,cell(Nim,1));  % output is structure array
end

            
for Iim=1:1:Nim,
    I1 = min(Iim,Nim1);
    I2 = min(Iim,Nim2);
    
    if (InPar.Verbose),
        fprintf('Run operation on structured images %d and %d\n',I1,I2);
    end
    
    
    %--- cross correlate images ---
    % very slow - using matlab xcorr2 function:
    OutSim(Iim).(ImageField) = xcorr2(Sim1(I1).(ImageField),Sim2(I2).(ImageField));

     
    %--- Normalization ---
    switch lower(InPar.Norm)
        case {'no','none'}
           % do nothing
        case '1'
           OutSim(Iim).(ImageField) = OutSim(Iim).(ImageField)./maxnd(OutSim(Iim).(ImageField));
        case 'n'
           OutSim(Iim).(ImageField) = OutSim(Iim).(ImageField)./numel(OutSim(Iim).(ImageField));
        otherwise
           error('Unknown Norm option');
    end

    %--- Prepare Header for output image ---
    switch InPar.HeadNum
       case 0
          % use minimal header
          HeaderInfo = [];
       case 1
           if (isfield(Sim1(I1),HeaderField)),
               HeaderInfo = Sim1(I1).(HeaderField);
           else
               HeaderInfo = cell(0,3);
           end
       case 2
           if (isfield(Sim2(I2),HeaderField)),
               HeaderInfo = Sim2(I2).(HeaderField);
           else
               HeaderInfo = cell(0,3);
           end
       otherwise
          error('Unknown Header option');
    end
    
    %--- Add to header comments regarding file creation ---
    if (~isfield(Sim1(I1),FileField)),
        Sim1(I1).(FileField) = [];
    end
    if (~isfield(Sim2(I2),FileField)),
        Sim2(I2).(FileField) = [];
    end
    [HeaderInfo] = cell_fitshead_addkey(HeaderInfo,...
                                        Inf,'COMMENT','','Created by sim_xcorr.m written by Eran Ofek',...
                                        Inf,'HISTORY','',sprintf('InImage1=%s',Sim1(I1).(FileField)),...
                                        Inf,'HISTORY','',sprintf('InImage2=%s',Sim2(I2).(FileField)));
    if (~isempty(InPar.AddHead)),
        %--- Add additional header keywords ---
        HeaderInfo = [HeaderInfo; InPar.AddHead];
    end   
    % fix header
    HeaderInfo = cell_fitshead_fix(HeaderInfo);
    
    OutSim(Iim).(HeaderField) = HeaderInfo;
      
    %--- Shift data ---
   [~,MaxInd]   = maxnd(OutSim(Iim).(ImageField));
   SizeOut      = size(OutSim(Iim).(ImageField));
   %Shift(Iim1).ShiftX = MaxInd(2)-(SizeOut(2)).*0.5;
   %Shift(Iim1).ShiftY = MaxInd(1)-(SizeOut(1)).*0.5;
   if (MaxInd(2)>SizeOut(2).*0.5),
      Shift(Iim).ShiftX = MaxInd(2) - SizeOut(2) - 1;
   else
      Shift(Iim).ShiftX = MaxInd(2);
   end
   if (MaxInd(1)>SizeOut(1).*0.5),
      Shift(Iim).ShiftY = MaxInd(1) - SizeOut(1) - 1;
   else
      Shift(Iim).ShiftY = MaxInd(1);
   end
   
   Xr = (-1:0.1:1);
   Yr = (-1:0.1:1).';

   Local = interp2(OutSim(Iim).(ImageField),MaxInd(2)+Xr,MaxInd(1)+Yr,'spline');
   [Max,MaxInd] = maxnd(Local);
   Shift(Iim).BestShiftX  = Shift(Iim).ShiftX + Xr(MaxInd(2));
   Shift(Iim).BestShiftY  = Shift(Iim).ShiftY + Yr(MaxInd(1));
   Shift(Iim).BestPeak    = Max;
   
   % fit a 2-D Gaussian

   % TO BE IMPLEMENTED

   % fft shift
    
   if (InPar.Shift),
        OutSim(Iim).(ImageField) = fftshift(fftshift( OutSim(Iim).(ImageField), 1), 2);
    end
   
end

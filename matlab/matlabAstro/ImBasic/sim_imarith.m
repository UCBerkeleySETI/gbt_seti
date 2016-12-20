function OutSim=sim_imarith(varargin)
%--------------------------------------------------------------------------
% sim_imarith function                                             ImBasic
% Description: Perform a binary arithmatic operations between two sets of
%              images and or constants. The input can be any type of
%              image, but the output is always a structure array.
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
%                      This parameter must be provided.
%            'Op'    - Operator {'+','-','.*','./','*','/',...
%                      '<','>','>=','<=','~=','==','&','|'},
%                      or a binary function function handle
%                      (e.g., @plus, @atan2).
%                      This parameter must be provided.
%            'Divide0'- A scalar value indicating what to do with
%                      zero devision (i.e., Inf). Default is 0.
%            'OutSIM' - Output is a SIM class (true) or a structure
%                      array (false). Default is true.
%            'ImageField' - Field name in the SIM which contains the
%                      image on which to operate. Default is 'Im'.
%            'MaskField' - Field name in the SIM which contains the
%                      image mask. Default is 'Mask'.
%            'CCDSEC1' - CCD section in the first operand on which
%                      to operate. This can be either a vector of
%                      [xmin, xmax, ymin, ymax] or a header keyword name
%                      which contains the data section.
%                      If empty then will use the full image.
%                      Default is empty.
%            'CCDSEC2' - Like 'CCDSEC1' but for the second operand.
%            'OutCCDSEC' - Cut the final image according to this
%                      CCD section (options like 'CCDSEC1').
%            'CreateMask' - Propogate mask image to the output structure
%                      array of images {true|false}. Default is true.
%            'MaskType' - If a mask file is created (FlagOp='no') then
%                      this is the data type of the mask.
%                      Default is 'uint16'.
%            'FlagOp' - Bit binary operation between the masks of the two
%                      input operands {'or'|'and'|'no'}. Default is 'or'.
%            'Bit_Divide0' - The bit index which to flag a devision by
%                      zero. Alternatively, this can be a function
%                      handle which get the the bit name
%                      (i.e., 'Bit_Divide0') and return the bit index.
%                      Default is @def_bitmask_specpipeline.
%            'HeadNum' - From which input operand to copy header {1 | 2}
%                        or to use minimal header {0}.
%                        Default is 1.
%            'CopyCat' - From which input operand to copy the catalog if 
%                      exist {1 | 2}, or do not copy (0). Default is 1.
%            'AddHead' - Cell array with 3 columns containing additional
%                      keywords to be add to the header.
%                      See cell_fitshead_addkey.m for header structure
%                      information. Default is empty matrix.
%            'CreateErr' - Create error image {true|false}.
%                      Default is false.
%            'Verbose' - Print progress messages {true|false}.
%                      Default is false.
%            --- images2sim.m parameters
%            'R2Field'- Name of the structure array field in which to
%                       store the image. Default is 'Im'.
%            'R2Name' - Name of the structure array field in which to
%                       store the image name. Default is 'ImageFileName'.
%            'ReadHead'- Read header information {true|false}.
%                       Default is true.
%            'ImType' - Image type. One of the following:
%                       'FITS'   - fits image (default).
%                       'imread' - Will use imread to read a file.
%                       'mat'    - A matlab file containing matrix,
%                                  or structure array.
%            'FitsPars'- Cell array of additonal parameters to pass to
%                        fitsread.m. Default is {}.
%            'ImreadPars' - Cell array of additonal parameters to pass to
%                        imread.m. Default is {}.
% Output  : - Structure array of images (SIM) in which the output images
%             are stored.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example:
% OutSim=sim_imarith('In1','blue0031.fits','In2','blue0032.fits','Op','+')
% OutSim=sim_imarith('In1','blue0031.fits','In2','blue001*.fits','Op','+','CCDSEC2',[1 10 1 10],'CCDSEC1',[1 10 1 10]);
% Reliable: 2
%-----------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
FileField   = 'ImageFileName';
MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';
CatField        = 'Cat';
CatColField     = 'Col';
CatColCellField = 'ColCell';

%DefV.Output      = [];
DefV.In1         = [];  % must supply
DefV.In2         = [];  % must supply
DefV.Op          = [];  % must supply
DefV.Divide0     = 0;
DefV.OutSIM      = true;
DefV.ImageField  = ImageField;
DefV.MaskField   = MaskField;
DefV.CCDSEC1     = [];   % must match to CCDSEC2
DefV.CCDSEC2     = [];
DefV.OutCCDSEC   = [];   % In the output system
DefV.CreateMask  = true; % propgate both images mask to OutSim
DefV.MaskType    = 'uint16';  % used if FlagOp='no'
DefV.FlagOp      = 'or';
DefV.Bit_Divide0 = @def_bitmask_specpipeline;
DefV.HeadNum     = 1;
DefV.CopyCat     = 1;
DefV.AddHead     = [];
DefV.CreateErr   = false;
DefV.Verbose     = false;
% images2sim.m parameters
DefV.R2Field    = ImageField;
DefV.R2Name     = FileField;
DefV.ReadHead   = true;
DefV.ImType     = 'FITS';
DefV.FitsPars   = {};
DefV.ImreadPars = {};

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% if operator is not division the set CreateMask to false
if (ischar(InPar.Op)),
    if (~strcmp(InPar.Op,'/'))
        InPar.CreateMask = false;
    end
else
    if (~strcmp(func2str(InPar.Op),'rdivide')),
        InPar.CreateMask = false;
    end
end

       
% check validity of input
if (isempty(InPar.In1)),
    error('In1 must be supplied');
end
if (isempty(InPar.In2)),
    error('In2 must be supplied');
end
if (isempty(InPar.Op)),
    error('Operator must be supplied');
end

% read images
Sim1 = images2sim(InPar.In1,varargin{:});
InPar = rmfield(InPar,'In1');
Sim2 = images2sim(InPar.In2,varargin{:});
InPar = rmfield(InPar,'In2');


Nim1 = numel(Sim1);
Nim2 = numel(Sim2);

% set bit mask index
InPar.Bit_Divide0 = get_bitmask_def(InPar.Bit_Divide0,'Bit_Divide0');

%--- Perform operation on all images ---
Nim    = max(Nim1,Nim2);

if (InPar.OutSIM),
    OutSim = simdef(Nim); %SIM;    % output is of SIM class
else
    OutSim = struct(InPar.ImageField,cell(Nim,1));  % output is structure array
end

for Iim=1:1:Nim,
    
    I1 = min(Iim,Nim1);
    I2 = min(Iim,Nim2);
    
    if (InPar.Verbose),
        fprintf('Run operation on structured images %d and %d\n',I1,I2);
    end
    
    % CCDSEC on input files
    CCDSEC1 = sim_ccdsec(Sim1(I1),InPar.CCDSEC1);
    CCDSEC2 = sim_ccdsec(Sim2(I2),InPar.CCDSEC2);
%     [CCDSEC1]=get_ccdsec_head(Sim1(I1).(HeaderField),InPar.CCDSEC1);
%     [CCDSEC2]=get_ccdsec_head(Sim2(I2).(HeaderField),InPar.CCDSEC2);
%     if (any(isnan(CCDSEC1)) | isempty(CCDSEC1)),
%         Size1   = size(Sim1(I1).(ImageField));
%         CCDSEC1 = [1 Size1(2) 1 Size1(1)];
%     end
%     if (any(isnan(CCDSEC2)) | isempty(CCDSEC2)),
%         Size2   = size(Sim2(I2).(ImageField));
%         CCDSEC2 = [1 Size2(2) 1 Size2(1)];
%     end  
   
    %--- run operation ---
    
    if (ischar(InPar.Op)),
        eval(sprintf('OutSim(Iim).(InPar.ImageField) = Sim1(I1).(InPar.ImageField)(CCDSEC1(3):CCDSEC1(4),CCDSEC1(1):CCDSEC1(2)) %s Sim2(I2).(InPar.ImageField)(CCDSEC2(3):CCDSEC2(4),CCDSEC2(1):CCDSEC2(2));',InPar.Op));
    else
        OutSim(Iim).(InPar.ImageField) = InPar.Op(Sim1(I1).(InPar.ImageField)(CCDSEC1(3):CCDSEC1(4),CCDSEC1(1):CCDSEC1(2)), Sim2(I2).(InPar.ImageField)(CCDSEC2(3):CCDSEC2(4),CCDSEC2(1):CCDSEC2(2)));
    end
    
    
    % cut output acccording to: OutCCDSEC
    if (~isempty(InPar.OutCCDSEC)),
       OutSim(Iim).(InPar.ImageField) = OutSim(Iim).(InPar.ImageField)(InPar.OutCCDSEC(3):InPar.OutCCDSEC(4),InPar.OutCCDSEC(1):InPar.OutCCDSEC(2));
    end
    
    if (InPar.CreateErr),
       error('CreateErr is not supported yet');
    end
    
    %--- special cases ---
    % deal with devision by zero
    
    switch char(InPar.Op)
        case {'/','./','rdivide','mrdivide'}
            % deal with devision by zero
            FlagInf = OutSim(Iim).(InPar.ImageField)==Inf;
            OutSim(Iim).(InPar.ImageField)(OutSim(Iim).(InPar.ImageField)==Inf) = InPar.Divide0;
        otherwise
            FlagInf = false(size(OutSim(Iim).(InPar.ImageField)));
            % do nothing
    end
        
    
    %--- set Flags ---
    if (InPar.CreateMask),
        if (~isfield(Sim1(I1),InPar.MaskField)),
            Mask1 = zeros(size(Sim1(I1).(InPar.ImageField)(CCDSEC1(3):CCDSEC1(4),CCDSEC1(1):CCDSEC1(2))), InPar.MaskType);
        else
            if (isempty(Sim1(I1).(InPar.MaskField))),
                Mask1 = zeros(size(Sim1(I1).(InPar.ImageField)(CCDSEC1(3):CCDSEC1(4),CCDSEC1(1):CCDSEC1(2))), InPar.MaskType);
            else
                Mask1 = Sim1(I1).(InPar.MaskField)(CCDSEC1(3):CCDSEC1(4),CCDSEC1(1):CCDSEC1(2));
            end
        end
        if (~isfield(Sim2(I2),InPar.MaskField)),
            Mask2 = zeros(size(Sim2(I2).(InPar.ImageField)(CCDSEC2(3):CCDSEC2(4),CCDSEC2(1):CCDSEC2(2))), InPar.MaskType);
        else
            if (isempty(Sim2(I2).(InPar.MaskField))),
                Mask2 = zeros(size(Sim2(I2).(InPar.ImageField)(CCDSEC2(3):CCDSEC2(4),CCDSEC2(1):CCDSEC2(2))), InPar.MaskType);
            else
                Mask2 = Sim2(I2).(InPar.MaskField)(CCDSEC2(3):CCDSEC2(4),CCDSEC2(1):CCDSEC2(2));
            end
        end
        
        switch lower(InPar.FlagOp)
            case 'or'
                OutSim(Iim).(InPar.MaskField) = bitor(Mask1,Mask2);
            case 'and'
                OutSim(Iim).(InPar.MaskField) = bitand(Mask1,Mask2);
            case 'no'
                OutSim(Iim).(InPar.MaskField) = zeros(size(Mask1),InPar.MaskType);
            otherwise
                error('Unknown FlagOp option');
        end
        OutSim(Iim).(InPar.MaskField) = maskflag_set(OutSim(Iim).(InPar.MaskField),[],InPar.Bit_Divide0,FlagInf);
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
                                        Inf,'COMMENT','','Created by sim_imarith.m written by Eran Ofek',...
                                        Inf,'HISTORY','',InPar.Op,...
                                        Inf,'HISTORY','',sprintf('InImage1=%s',Sim1(I1).(FileField)),...
                                        Inf,'HISTORY','',sprintf('InImage2=%s',Sim2(I2).(FileField)));
    if (~isempty(InPar.AddHead)),
        %--- Add additional header keywords ---
        HeaderInfo = [HeaderInfo; InPar.AddHead];
    end   
    % fix header
    HeaderInfo = cell_fitshead_fix(HeaderInfo);

    OutSim(Iim).(HeaderField) = HeaderInfo;
      
    if (InPar.CopyCat==1 && isfield_notempty(Sim1(I1),CatField)),
        OutSim(Iim).(CatField)        = Sim1(I1).(CatField);
        OutSim(Iim).(CatColField)     = Sim1(I1).(CatColField);
        OutSim(Iim).(CatColCellField) = Sim1(I1).(CatColCellField);
    elseif (InPar.CopyCat==2 && isfield_notempty(Sim2(I2),CatField)),
        OutSim(Iim).(CatField)        = Sim2(I2).(CatField);
        OutSim(Iim).(CatColField)     = Sim2(I2).(CatColField);
        OutSim(Iim).(CatColCellField) = Sim2(I2).(CatColCellField);
    else
        % do nothing
    end
    
end

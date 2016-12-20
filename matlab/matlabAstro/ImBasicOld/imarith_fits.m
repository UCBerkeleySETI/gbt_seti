function [OutList]=imarith_fits(varargin)
%--------------------------------------------------------------------------
% imarith_fits function                                            ImBasic
% Description: Perform a binary arithmatic operations between two sets of
%              images and or constants. The input can be either
%              FITS images, matrices or structure arrays (see read2sim.m),
%              while the output is always a file.
%              This function load and work and save images one by one so
%              it is not memory extensive. Use imarith_sim.m if
%              you want to operate on all images in memory.
%              Use of imarith_sim.m may save time in reading/writing
%              FITS images, but requires a lot of memory.
% Input  : * This can be either a list of ...,key,val,... arguments or
%            the value of the Output, Input1, Operator, Input2 keywords
%            followed by a list of ...,key,val,... arguments.
%            The following keywords are available:
%            'Output' - List of output images. See create_list.m for
%                       details. If empty matrix, then set it to be equal
%                       to the Input1 list.
%            'Input1' - 'First list of images. See create_list.m for
%                       details. Alternatively this could be a scalar.
%                       If you want to operate with a different scalar on
%                       each image then use num2cell - see Example.
%                       If the list contains only one image and the Input2
%                       argument contains multiple images then the
%                       operation will be performed between the single
%                       image and each one of the images in the second
%                       input (Input2).
%            'Operator'- String containing any MATLAB operator.
%                       Note that '.*' is multiplication, 
%                       while '*' is matrix multiplication.
%            'Input2' - Second list of images. Like Input1.
%            'Divide0'  - Set division by zero to specific value, default is 0.
%            'Scalar'   - Scalar operator {'y' | 'n'}, default is 'y'.
%                         If 'n' then the operator will be assume to be
%                         vectorize instead of scalar.
%            'OutPrefix'- Add prefix before output image names,
%                         default is empty string (i.e., '').
%            'OutDir'   - Directory in which to write the output images,
%                         default is empty string (i.e., '').
%            'Header'   - From which input to copy header {1 | 2} or to
%                         use minimal header {0}.
%                         Default is 1, however, if Input1 is scalar than
%                         default will be 2.
%            'AddHead'  - Cell array with 3 columns containing additional
%                         keywords to be add to the header.
%                         See cell_fitshead_addkey.m for header structure
%                         information. Default is empty matrix.
%            'DataType' - Output data type (see fitswrite.m for options), 
%                         default is float32.
%            'CCDSEC'   - Image sction for both images on which to do the
%                         arithmatic operation.
%                         This could be string containing image keyword 
%                         name (e.g., 'CCDSEC'), or a vector of 
%                         [Xmin, Xmax, Ymin, Ymax].
%                         If keyword than it will be obtained from
%                         the first input image.
%                         The output image size will be equal to that specified
%                         in the CCDSEC keyword, so in practice the output 
%                         image will be trimmed only if Trim='y'.
%                         If empty matrix (default; i.e., []) than do not use CCDSEC.
%            'Trim'     - Trim the resultant image {'y' | 'n'} according to CCDSEC).
%                         Default is 'y'. NOT AVAILABLE YET (behaves like default).
%            'Save'     - Save output FITS image {'y' | 'n'}.
%                         Default is 'y'.
%            'Flag'      - Add to the output structure array a field
%                          named 'FlagIm' containing a flag image.
%                          Options are {true|false}. Default is false.
%                          The Flag get its value from the FlagMap.
%                          Default is 1 for Inf, 2 for NaN, 0 otherwise.
%            'SaveFlag'  - Save Flag image {true|false}. The same
%                          paramters as the actual image will be used.
%                          Default is false.
%            'FlagName'  - Cell array of Flag images to save.
%                          Default is empty.
%            'FlagOutPrefix' - Like the OutPrefix argument, but for the
%                          flag images. Default is ''.
% Output  : - Cell array containing output image names.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                   Jun 2010
%    URL : http://weizmann.ac.il/home/eran/matlab/
% Example: [OutList]=imarith_fits('yyout.fits','try1.fits','+',2);
%          OutList=imarith_fits('Input1','@list','Input2','@list','Operator','-','OutPrefix','uu_')
%          [OutList]=imarith_fits('Input1','@list','Input2','@list','Operator','-','OutPrefix','uu_','FlagName',{'A1.fits','A2.fits','A3.fits'},'SaveFlag',true,'Flag',true)
%          OutList=imarith_fits({'out10.fits','out11.fits','out12.fits'},num2cell([1 2 3]),'.*','try1.fits');
%          OutList=imarith_fits('Input1',{'out10.fits','out11.fits','out12.fits'},'Input2','try1.fits','OutPrefix','c_','Operator','./')
% Reliable: 2
%--------------------------------------------------------------------------

ImageField   = 'Im';
HeaderField  = 'Header';
FileField    = 'ImageFileName';


DefV.Output      = [];
DefV.Input1      = [];  % must supply
DefV.Input2      = [];  % must supply
DefV.Operator    = [];  % must supply
DefV.Divide0     = 0;
DefV.Scalar      = 'y';
DefV.OutPrefix   = '';
DefV.OutDir      = '';
DefV.Header      = 1;
DefV.AddHead     = [];
DefV.DataType    = 'float32';
DefV.CCDSEC      = [];
DefV.Trim        = 'y';
DefV.Save        = true;
DefV.Format      = 'FITS';
DefV.Extension   = 1;
DefV.ImreadOpt   = {};
DefV.ImwriteOpt  = {};
DefV.Flag        = false;
DefV.SaveFlag    = false;
DefV.FlagName    = {};
DefV.FlagOutPrefix = '';

OldInput = false;
if (~ischar(varargin{1})),
    OldInput = true;
else
   if (isempty(find(strcmpi(fieldnames(DefV),varargin{1}), 1))),
       OldInput = true;
   end
end
if (OldInput),
    % first input is not one of the allowed keywords,
    % therefore assume old input format is used:
    % old version input: Output,Input1,Operator,Input2
    Output   = varargin{1};
    Input1   = varargin{2};
    Operator = varargin{3};
    Input2   = varargin{4};
    varargin = varargin(5:end);
end

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

if (OldInput),
    InPar.Output   = Output;
    InPar.Input1   = Input1;
    InPar.Operator = Operator;
    InPar.Input2   = Input2;
end

% check validity of input
if (isempty(InPar.Input1)),
    error('Input1 must be supplied');
end
if (isempty(InPar.Input2)),
    error('Input2 must be supplied');
end
if (isempty(InPar.Operator)),
    error('Operator must be supplied');
end

% output
if (isempty(InPar.Output)),
    InPar.Output = InPar.Input1;
end
[~,ListCellOut] = create_list(InPar.Output);


% work on images one by one
[~,ListCell1] = create_list(InPar.Input1);
[~,ListCell2] = create_list(InPar.Input2);
Ninput1       = numel(ListCell1);
Ninput2       = numel(ListCell2);
Nim           = max(Ninput1,Ninput2);
OutList       = cell(1,Nim);
InPar1        = InPar;
for Iim=1:1:Nim,
    InPar1.Input1   = ListCell1{min(Ninput1,Iim)};
    InPar1.Input2   = ListCell2{min(Ninput2,Iim)};
    InPar1.Output   = ListCellOut{Iim};
    if (~isempty(InPar.FlagName)),
       InPar1.FlagName = InPar.FlagName(Iim);
    end
    CellPars        = struct2keyvalcell(InPar1);
    
    [OutList(Iim)]  = imarith_sim(CellPars{:});
end




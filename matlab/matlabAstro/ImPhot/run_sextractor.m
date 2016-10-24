function [CatSex,OutParam,ParamStruct]=run_sextractor(ImageName,ConfigFile,ParamFile,CatName,CatType,varargin)
%--------------------------------------------------------------------------
% run_sextractor function                                           ImPhot
% Description: Run SExtractor from matlab.
%              This function is obsolte. Use sextractor.m instead.
% Input  : - A list of FITS images for which to run SExtractor. This can
%            be a string containing image name or a string containing
%            wild cards (multiple images), or a cell array containing
%            list of images or a file name (preceded by '@') containing
%            a list of images.
%          - Configuration file name, default is matlab.sex (in Path).
%            If empty matrix (i.e., []) use default.
%          - File name or a cell array containing the output parameters
%            (e.g., {'X_IMAGE','Y_IMAGE','MAG_AUTO'}.
%            Default is to to use the default.param file (in Path).
%            If empty matrix use default.
%          - File name in which to save the SExtractor output catalog.
%            This can be a string, cell array, or file name (preceded
%            by '@') containing a list of images.
%            If empty matrix, then donot save output files,
%            default is empty matrix.
%          - Output file name type {'FITS_1.0' | 'FITS_LDAC' | 'ASCII_HEAD'},
%            default is 'ASCII'. If empty matrix use default.
%          * Arbitrary number of pairs of arguments, where the first
%            argument is the configuration parameter name and the second
%            is its value (...,keyword,value,...).
%            see SExtractor manual for details.
% Output : - Cell array of matrices, each matrix containing the
%            SExtractor output for one image.
%          - Cell array of output parameter names.
%          - Structure in which the field names are the output parameters
%            followed by their column index.
%            Note that parenthesis are replaced by underscore.
% Install: 1. Install SExtractor
%          2. Update the path for your local copy of SExtractor (second
%             line of this program).
%          3. Note that the FILTER_NAME in default.sex should contain
%             the full path for the filter.
% Reference: http://terapix.iap.fr/rubrique.php?id_rubrique=91/
% Tested : Matlab 5.3/7.0
%     By : Eran O. Ofek                   July 2003
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: SexMat=run_sextractor('ccd.075.0.fits',...
%                                [],[],'Cat.out',[],'DETECT_THRESH','5');
%          [SexMat,OutPar]=run_sextractor('@list.s',...
%                                [],{'X_IMAGE','Y_IMAGE','MAG_AUTO'},[]);
% Reliable: 1
%--------------------------------------------------------------------------
MSDir      = which_dir(mfilename);
SE_Rel     = '../bin/SExtractor/src/sex-2.5';
Config_Rel = '../bin/SExtractor/config/';
SE_Prog    = sprintf('%s%s%s',MSDir,filesep,SE_Rel);
ConfigPath = sprintf('%s%s%s',MSDir,filesep,Config_Rel);



DefConfigFile = 'default.sex';
DefParamFile  = 'default.param';
DefCatName    = [];
DefCatType    = [];
DefConfigFile = sprintf('%s%s',ConfigPath,DefConfigFile);
DefParamFile  = sprintf('%s%s',ConfigPath,DefParamFile);
DefOutputType = 'ASCII_HEAD';

if (nargin==1),
   ConfigFile   = DefConfigFile;
   ParamFile    = DefParamFile;
   CatName      = DefCatName;
   CatType      = DefCatType;
elseif (nargin==2),
   ParamFile    = DefParamFile;
   CatName      = DefCatName;
   CatType      = DefCatType;
elseif (nargin==3),
   CatName      = DefCatName;
   CatType      = DefCatType;
elseif (nargin==4),
   CatType      = DefCatType;
else
   % do nothing
end

if (isempty(ConfigFile)),
   ConfigFile = DefConfigFile;
end
if (isempty(ParamFile)),
   ParamFile  = DefParamFile;
end
if (isempty(CatType)),
   CatType = DefOutputType;
end


if (iscell(ParamFile)==1),
   % Parameters file is a cell array
   OutParam     = ParamFile;
   ParamFile    = create_list(ParamFile,[],'n');   % convert to file
   IsTmpParam   = 1;
else
   OutParam   = textread(ParamFile,'%s','commentstyle','shell');
   IsTmpParam   = 0;
end
Nparam = length(OutParam);


% Extract extra parameters to pass to SExtractor
Narg = length(varargin);
ExtraPar = ' ';
for I=1:2:Narg-1,
   ExtraPar = sprintf('%s -%s %s',ExtraPar,varargin{I},varargin{I+1});
end

[~,ImCell] = create_list(ImageName,[],'n');
%delete(TmpFileName);

if (isempty(CatName)==1),
   CatIsTmp = 1;
else
   CatIsTmp = 0;
   [TmpFileName,CatCell] = create_list(CatName,[],'n');
   delete(TmpFileName);
end

% do for each FITS image:
Nim   = length(ImCell);
for Iim=1:1:Nim,
    
   Image = ImCell{Iim};

   switch CatIsTmp
    case 1
       CatFile = tempname;
    otherwise
       CatFile = CatCell{Iim};
   end

   RunStr = sprintf('%s %s -c %s -PARAMETERS_NAME %s -CATALOG_NAME %s -CATALOG_TYPE %s %s',SE_Prog,Image,ConfigFile,ParamFile,CatFile,CatType,ExtraPar);

   % run SExtractor
   system(RunStr);

   % read output
   switch lower(CatType)
    case 'ascii_head'
        ReadSex = textread(CatFile,'%n','commentstyle','shell');
        CatSex{Iim}  = reshape(ReadSex,[Nparam, length(ReadSex)./Nparam]).';
    case 'fits_1.0'
       CatSex{Iim} = cell2mat(fitsread(CatFile,'BinTable'));
    case 'fits_ldac'
       CatSex{Iim} = cell2mat(read_fits_ldac(CatFile));
    otherwise
       error('Unknown CatType option');
   end

   switch CatIsTmp
    case 1
       delete(CatFile);
    otherwise
       % do nothing
   end
end

% construct a structure of column names:
ParamStruct = cell2struct(num2cell([1:length(OutParam)]),regexprep(OutParam,{'(',')'},'_'),2);

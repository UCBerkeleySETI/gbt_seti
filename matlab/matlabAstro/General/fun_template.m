function fun_template(FunName,ToolBox,Path)

%--------------------------------------------------------------------------
% fun_template function                                            General
% Description: Generate a functionm template with help section and basic
%              optional commands.
% Input  : - Function name (e.g., 'my_fun1.m').
%          - Toolbox name (e.g., 'General').
%          - Toolbox path. Default is '~/matlab/fun/'.
% Output : null
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: fun_template('trya1.m','General');
% Reliable: 2
%--------------------------------------------------------------------------


if (nargin==2),
    Path = '~/matlab/fun/';
end
FullPath = sprintf('%s%s%s%s',Path,ToolBox,filesep,FunName);

FID = fopen(FullPath,'w');

fprintf(FID,'function []=%s(varargin)\n',FunName(1:end-2));
fprintf(FID,'%%--------------------------------------------------------------------------\n');
fprintf(FID,'%% %s function %s %s\n',FunName(1:end-2),blanks(77-14-length(FunName)-length(ToolBox)),ToolBox);
fprintf(FID,'%% Description: \n');
fprintf(FID,'%% Input  : - \n');
fprintf(FID,'%%          * Arbitrary number of pairs of arguments: ...,keyword,value,...\n');
fprintf(FID,'%%            where keyword are one of the followings:\n');
fprintf(FID,'%%            --- Additional parameters\n');
fprintf(FID,'%%            Any additional key,val, that are recognized by one of the\n');
fprintf(FID,'%%            following programs:\n');
fprintf(FID,'%% Output : - \n');
Version = regexp(version,'\(','split');
Version = Version{2}(1:end-1);
fprintf(FID,'%% License: GNU general public license version 3\n');
fprintf(FID,'%% Tested : Matlab %s\n',Version);
Date   = date;
Month  = Date(4:6);
Year   = Date(8:11);
AuN = sprintf('By : Eran O. Ofek');
fprintf(FID,'%%     %s                    %s %s\n',AuN,Month,Year);
fprintf(FID,'%%    URL : http://weizmann.ac.il/home/eofek/matlab/\n');
fprintf(FID,'%% Example: \n');
fprintf(FID,'%% Reliable: \n');
fprintf(FID,'%%--------------------------------------------------------------------------\n');
fprintf(FID,'\n\n');
% input args
fprintf(FID,'NumVarargs = length(varargin);\n');
fprintf(FID,'if NumVarargs > 3\n');
fprintf(FID,'     errId = ''%s:TooManyInputArguments'';\n',FunName);
fprintf(FID,'     errMsg = ''InPar1, [InPar2, InPar3]'';\n');
fprintf(FID,'     error(errId, errMsg);\n');
fprintf(FID,'end\n');
fprintf(FID,'Gaps = cellfun(@isempty, varargin);\n');
fprintf(FID,'DefArgs = {InPar1Def InPar2Def InPar3Def};    %% default input arguments\n');
fprintf(FID,'Suboptargs = DefArgs(1 : NumVarargs);\n');
fprintf(FID,'varargin(Gaps) = Suboptargs(Gaps);\n');
fprintf(FID,'DefArgs(1 : NumVarargs) = varargin;\n');
fprintf(FID,'[Par1 Par2 Par3] = DefArgs{:}\n');
% key,val...
fprintf(FID,'\n\n');
fprintf(FID,'DefV. = \n');
fprintf(FID,'InPar = set_varargin_keyval(DefV,''n'',''use'',varargin{:});\n');

fclose(FID);


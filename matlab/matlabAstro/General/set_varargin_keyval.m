function Par=set_varargin_keyval(Def,CheckExist,IfEmpty,varargin)
%--------------------------------------------------------------------------
% set_varargin_keyval function                                     General
% Description: The purpose of this program is to handle a list of pairs
%              of keywords and values input arguments.
%              The program is responsible to check if the keywords
%              are valid, and if a specific keyword is not supplied by
%              the user, then the program will set it to its default
%              value.
% Input  : - Structure containing arbitrary number of fields in which
%            the field names represent all the optional input keywords,
%            and the content of each field is the default value for this
%            specific keyword.
%          - Check if input keyword in the varargin list exist in the
%            default parameters structure {'y' | 'n'}. Default is 'y'.
%            If 'y' and keyword does not exist then the program will abort
%            with error, if 'n' andkeyword does not exist then the program
%            will ignore the additional parameter and will not set its value
%            into the output Par structure.
%            Another way to control this parameter is directly from the
%            list of keywords. Specificaly, if One of the keywords/value
%            in the list is 'VerifyKey','n' then the program will set
%            (override) this parameter to 'n'.
%          - What to do in case that the value associated with a keyword is
%            an empty matrix.
%            'use' - will set the value of the keyword in the output Par
%                    structure to an empty matrix (default).
%            'def' - will set the value of the keyword in the output Par
%                    structure to its value in the default structure.
%          * Arbitrary number of pairs of input arguments
%            ...,keyword,value,...
%            Special keywords include:
%            'VerifyKey' - that can be used to override the value of the
%            second input argument (CheckExist);
%            'SetEparFile' - {true|false}. If true, then a new epar 
%                          default file for the user will be created
%                          with the current parameters as default.
%                          Default is false. This should be used only
%                          once for the creation of an epar file.
%            % 'GetFromPar' - NOT operational.
%            %'UserDefault' - {'y'|'n'} (default is 'y'). If this parameter
%            %is 'y' then the program will look for a file named
%            %<caller_function>_PAR.mat and will load it. This mat file
%            %contains the last used set of key,val input arguments
%            %as a default.
% Output : Structure containing the same fields supplied in the default
%          structure (Def). The content of each field is set from the
%          input keyword,value list, or the default list.
% Tested : matlab 7.10
%     By : Eran O. Ofek                    Jun 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% See also: epar.m
% Example: DefV.Save = 'y';
%          InPar = set_varargin_keyval(DefV,'y','use',varargin{:});
% Reliable: 1
%--------------------------------------------------------------------------

DefUserDefault = true;
SetEparFile    = false;

% if (isempty(varargin) && strcmpi(CheckExist,'fast')),
%     Par = Def;
% else

[ST] = dbstack;                % get the name of callef functions
if (length(ST)==1),
   CallerFun = 'session';
else
   CallerFun = ST(2).name;        % Name of caller function 
end

DefInPar.CheckExist = 'y';
DefInPar.IfEmpty    = 'use';
if (isempty(CheckExist)),
   CheckExist = DefInPar.CheckExist;
end
if (isempty(IfEmpty)),
   IfEmpty = DefInPar.IfEmpty;
end

% treating special keywords
% VerifyKey

Ivkk = strcmpi(varargin(1:2:end-1),'VerifyKey');
Ivkv = strcmpi(varargin(2:2:end),'n');
if (sum(Ivkk.*Ivkv)>0),
   % override the value of CheckExist if one of the key,val is 'VerifyKey','n'
   CheckExist = 'n';
end   

% set epar file
Isef = strcmpi(varargin(1:2:end-1),'SetEparFile');
if (any(Isef)),
    SetEparFile = varargin{Isef+1};
end

% UserDefault
%UseEpar = false;
Ivkk = strcmpi(varargin(1:2:end-1),'UserDefault');

if (any(Ivkk)),
    UserDefault = varargin{find(Ivkk)+1}; 
else
    % GetFromPar was either not provide (use default = 'y')
    % or it was provided with value = 'y'.
    UserDefault = DefUserDefault;  %true;
end

% remove UserDefault keyword from varargin
Np   = length(varargin);
PIvkk(1:2:Np-1) = Ivkk;
PIvkk(2:2:Np)   = Ivkk;
varargin = varargin(~PIvkk);


% 
% switch lower(UserDefault)
%  case 'y'
%     % read user default
%     % reset Def parameter according to user default file
%     DefFile = epar(CallerFun,'read');
%     if (~isempty(DefFile)),
%         Def = DefFile;
%     end
%  otherwise
%     % do nothing
% end

%--- read function default parameters using the epar command ---
if (UserDefault),
    UserDefV  = epar(CallerFun,'read');
    UserArgin = struct2keyvalcell(UserDefV);
    varargin  = [UserArgin, varargin];  % concat User default argument before user new input arguments
end



Par = Def;
FN  = fieldnames(Def);
%Nf  = length(FN);

Narg = length(varargin);
if ((Narg.*0.5)~=floor(Narg.*0.5)),
   error('Incorrect number of input arguments to function %s',CallerFun);
end
for Iarg=1:2:Narg-1,
    IndArg = find(strcmpi(FN,varargin{Iarg})==1);
    if (isempty(IndArg)),
        switch lower(CheckExist),
            case 'y'
                error('Keyword name %s given as input argument to function %s is not a valid option',varargin{Iarg},CallerFun);
            case 'n'
                % ignore the missing keyword
                % do nothing
            otherwise
                error('Unknown CheckExist option');
        end
    else
        switch lower(IfEmpty)
            case 'use'
                % use as is
                Par.(FN{IndArg}) = varargin{Iarg+1};
            case 'def'
                % if value is empty then set to default
                if (isempty(varargin{Iarg+1})),
                    % set to default - already default so do nothing
                else
                    % use as is
                    Par.(FN{IndArg}) = varargin{Iarg+1};
                end
            otherwise
                error('Unknown IfEmpty option');
        end
    end
end

% %--- save function default parameters using the epar command ---
% THIS SHOULD BE LEFT COMMENTED OUT
if (SetEparFile),
    KeyValCell = struct2keyvalcell(Par);
    epar(CallerFun,'setpars',KeyValCell{:});
end

% end
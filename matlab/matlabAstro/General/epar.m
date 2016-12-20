function InPar=epar(FunName,Type,varargin)
%--------------------------------------------------------------------------
% epar function                                                    General
% Description: Allow the user to edit key/val parameters of functions.
%              All the functions which use the set_varargin_keyval.m
%              function will save their parameters into a file:
%              ~matlab/.FunPars/<FunName>_PAR.mat.
%              The user then can change the default values of these
%              parameters by "epar <FunName>".
% Input  : - Function name (e.g., 'imarith_fits').
%          - A string indicating the way the function is being executed:
%            'setpars' - set the function default parameters with the
%                        parameters specified in the varargin input.
%                        This is a non-interactive mode used by functions.
%            'int'     - interactive mode that allows the user to set
%                        the values of an existing parameters file.
%                        Default.
%            'read'    - Non interactive mode that will not modify the
%                        value of keyword values in the parameters file,
%                        but will read the content of the parameters file
%                        into a structure of parameters (key,val).
%            'def'     - Revert to function default parameters.
%          * Arbitrary number of pairs of ...,key,val,... to set
%            in the function parameters file.
% Output : - The structure of key,val parameters.
%            If Type is interactive ('int') then will return an empty
%            matrix.
% Tested : Matlab 2011b
%     By : Eran O. Ofek                    Jan 2013
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% See also: lpar.m, set_varargin_keyval.m
% Example: sim_fft(1);          % run once to create parameters file
%          epar sim_fft         % interactive mode
%          lpar sim_fft         % like: epar sim_fft read
%          epar sim_fft def     % revert user parameters file to default
%                               % function parameters
% Reliable: 2
%--------------------------------------------------------------------------
global FunParsFullName


% Default par dir is: '~/matlab/.FunPars'
DirPAR      = sprintf('~%smatlab%s.FunPars',filesep,filesep);  % EDIT IF YOU WANT TO CHANGE PAR directory

% create par dir if needed
if (exist(DirPAR,'dir')==0),
    mkdir(DirPAR);
end


Def.Type = 'int';
if (nargin==1),
    Type = Def.Type;
end
if (isempty(Type)),
    Type = Def.Type;
end

% remove .m from FunName
FunName = regexprep(FunName,'\.m','');

FunPath = which_dir(FunName);

% set the Parameters file name
FunParsName = sprintf('%s_PAR.mat',FunName);
% Full path and file name of Parameters file

FunParsFullName = sprintf('%s%s%s',DirPAR,filesep,FunParsName);

ParFileExist = exist(FunParsFullName,'file');

InPar = [];
switch lower(Type)
    case 'read'
        % Read PAR file and return the InPar variable containing the user
        % default parameters
        if (ParFileExist==0),
            % Parameter files does not exist
            InPar = [];
        else
            InPar = load2(FunParsFullName);
        end
    case 'int'
        % Interactive mode that allows the user to set the values
        % of the parameters
        if (ParFileExist==0),
            error('Parameter files does not exist - run %s.m once to create',FunName);
        else
            DefV = load2(FunParsFullName);
            assignin('base','DefV',DefV);

            openvar('DefV');
            drawnow;
            pause(0.5); % wait for client to become visible

            % Get handle of variables client in the Variable Editor
	        %http://blogs.mathworks.com/community/2008/04/21/variable-editor/
            jDesktop = com.mathworks.mde.desk.MLDesktop.getInstance;
            jClient = jDesktop.getClient('DefV');
            hjClient = handle(jClient,'CallbackProperties');

            % Instrument the client to fire callback when it's closed
            set(hjClient,'ComponentRemovedCallback',{@my_callback_epar,'DefV'});

            % reload changes
            %DefV = load2(FunParsFullName);
            InPar = [];
            
        end
    case 'setpars'  
          
         if (ParFileExist==0),
             % Parameter files does not exist
             % this is the first time the function is exacuated
             if (isempty(varargin)),
                 error('Request to set parameter values but the input is empty');
             else
                 % set the varargin into the DefV structure
                 DefV = cell2struct(varargin(2:2:end),varargin(1:2:end-1),2);
                 % save DefV for function in parameters directory
                 save(FunParsFullName,'DefV');
             end
         else
             if (isempty(varargin)),
                 error('Request to set parameter values but the input is empty');
             else
                 InPar = load2(FunParsFullName);
                 FieldsPar = fieldnames(InPar);
                 for If=1:1:length(FieldsPar),
                    IndF = find(strcmp(FieldsPar{If},varargin(1:2:end-1)));
                    InPar.(FieldsPar{If}) = varargin{IndF.*2};
                 end
                 % save the InPar
                 DefV = InPar;
                 save(FunParsFullName,'DefV');
             end
         end
    case 'def'
        if (ParFileExist~=0),
            delete(FunParsFullName);
        end
    otherwise
        error('Unknown Type option');
end

             


function my_callback_epar(varEditorObj,eventData,varname)
% do nothing
global FunParsFullName

DefV = evalin('base','DefV');
save(FunParsFullName,'DefV');


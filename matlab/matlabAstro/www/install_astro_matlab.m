function install_astro_matlab
%--------------------------------------------------------------------------
% install_astro_matlab function                                        www
% Description: Install or update the astronomy and astrophysics package
%              for matlab.
% Input  : null 
% Output : null
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Feb 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: install_astro_matlab;
% Reliable: 2
%--------------------------------------------------------------------------
Verbose    = true;

BaseURL    = 'http://webhome.weizmann.ac.il/'; %home/eofek/matlab/data/
 
SourceDir{1}  = sprintf('%s%s',BaseURL,'home/eofek/matlab/fun/');
SourceDir{2}  = sprintf('%s%s',BaseURL,'home/eofek/matlab/data/');
SourceSU      = sprintf('%s%s',BaseURL,'home/eofek/matlab/startup.m');

I = 0;
I = I + 1; CritProg{I} = sprintf('%swww/pwget.m',SourceDir{1});
I = I + 1; CritProg{I} = sprintf('%swww/find_urls.m',SourceDir{1});
I = I + 1; CritProg{I} = sprintf('%sGeneral/isempty_cell.m',SourceDir{1});
I = I + 1; CritProg{I} = sprintf('%sGeneral/set_varargin_keyval.m',SourceDir{1});
I = I + 1; CritProg{I} = sprintf('%sGeneral/struct2keyvalcell.m',SourceDir{1});
I = I + 1; CritProg{I} = sprintf('%sGeneral/epar.m',SourceDir{1});
I = I + 1; CritProg{I} = sprintf('%sGeneral/which_dir.m',SourceDir{1});


cd(sprintf('~%s',filesep));
PWD = pwd;

warning('off','MATLAB:MKDIR:DirectoryExists');

DefBaseDir = 'matlabAstro';
fprintf('\n\n');
Inp = input(sprintf('Enter sub directory or full path in which to install the matlab astro package [%s]:',DefBaseDir),'s');
if (isempty(Inp)),
    BaseDir = DefBaseDir;
else
    BaseDir = Inp;
end
fprintf('The MATLAB astronomy package will be installed in: %s\n',BaseDir);


mkdir(BaseDir);
cd(BaseDir);
BaseDir = pwd;
addpath(BaseDir);

Ncrit = numel(CritProg);
for Icrit=1:1:Ncrit,
    wget(CritProg{Icrit});
end

% startup.m
fprintf('\n\n');
Ans = input('Do you want to install the startup.m file (y/n):','s');
if (strcmpi(Ans,'y')),
    wget(SourceSU);
end


Cont = false;
while (~Cont),
    fprintf('\n\n');
    Ans = input('How many wget sessions do you want to run in parallel (1..20):','s');
    MaxWget = str2double(Ans);
    if (isnan(MaxWget) || MaxWget<1 || MaxWget>20),
        fprintf('The number of parallel sessions must be in the range of 1 to 20');
        Cont = false;
    else
        Cont = true;
    end
end

Nsd = numel(SourceDir);
CopyAll = true;
for Isd=1:1:1,
    Tmp=regexp(SourceDir{Isd},'/','split');
    ParentDir = Tmp{end-1};
    mkdir(ParentDir);
    cd(ParentDir);
    
    
    % for each source directory
    [ListDir] = find_urls(SourceDir{Isd},'match',sprintf('%s\\w+/',SourceDir{Isd}));
    Ndir = numel(ListDir);
    for Idir=1:1:Ndir,
        % for each directory (e.g., toolbox) in source directory
        Tmp=regexp(ListDir{Idir},'/','split');
        ToolBox = Tmp{end-1};
        mkdir(ToolBox);
        cd(ToolBox);
        if (Verbose),
            fprintf('Get content of toolbox: %s \n',ToolBox);
        end
        
        if (CopyAll),
            Copy1 = true;
        else
            fprintf('\n\n');
            Ans = input('Do you want to install this data directory (y/n):','s');
            if (strcmpi(Ans,'y')),
                Copy1 = true;
            else
                Copy1 = false;
            end
        end
        
        if (Copy1),
            [ListFile] = find_urls(ListDir{Idir},'match',sprintf('%s\\w+\\.\\w+',ListDir{Idir}));
            pwget(ListFile,'-q -nc',MaxWget);

            % next level (if exist)
            ListSubDir = find_urls(ListDir{Idir},'match',sprintf('%s\\@*\\w+/',ListDir{Idir}));
            Nsubdir    = numel(ListSubDir);
            for Isubdir=1:1:Nsubdir,
                Tmp=regexp(ListSubDir{Isubdir},'/','split');
                SubToolBox = Tmp{end-1};
                mkdir(SubToolBox);
                cd(SubToolBox);
                if (Verbose),
                    fprintf('Get into sub directory: %s\n',SubToolBox);
                end
                ListFile = find_urls(ListSubDir{Isubdir},'match',sprintf('%s\\w+\\.\\w+',ListSubDir{Isubdir}));
                pwget(ListFile,'-q -nc',MaxWget);
                cd ..
            end
        end
        cd ..
    end
    cd(BaseDir);
end


Ans = input('Do you want to install the data directories (y/n):','s');
if (strcmpi(Ans,'y')),
    % install the data directory
    cd(BaseDir)
    mkdir('data');
    cd('data');
    List = find_urls(SourceDir{2},'match','.tar.gz');
    Nlist = numel(List);
    FlagList = false(Nlist,1);
    for Ilist=1:1:Nlist,
        Tmp = regexp(List{Ilist},'/','split');
        DataFileName = Tmp{end};
        Tmp = regexp(DataFileName,'.tar.gz','split');
        DataFileName = Tmp{1};

        fprintf('\n\n');
        Ans = input(sprintf('Do you want to install the %s data directory (y/n):',DataFileName),'s');
        if (strcmpi(Ans,'y')),
            FlagList(Ilist) = true;
        end
    end

    % retrieve data files in tar.gz format
    pwget(List(FlagList),'-q -nc',MaxWget);
    pause(1);

    fprintf('\n\n');
    Ans = input('Do you want to open tar.gz data files into directories ([y]/n):','s');
    if (strcmpi(Ans,'n')),
        % skip open tar files
    else
        % open gzip
        Files = dir('*.tar.gz');
        for If=1:1:numel(Files),
            [Status,Res]=system(sprintf('gzip -d %s',Files(If).name));
            pause(1);
        end

        % open tar and delete tar file
        Files = dir('*.tar');
        for If=1:1:numel(Files),
            [Status,Res]=system(sprintf('tar -xvf %s',Files(If).name));
            [Status,Res]=system(sprintf('rm -rf %s',Files(If).name));
        end
    end
end      

% clean critical programs
cd(BaseDir);
for Icrit=1:1:Ncrit,
    Tmp = regexp(CritProg{Icrit},'/','split');
    delete(Tmp{end});
end
rmpath(BaseDir);

% what next - external toolboxes
fprintf('\n\n');
fprintf('General instructions for additional capabilities:\n')
fprintf('  1. Install cdsclient and edit ./fun/Catalogue/vizquery_path.m (see instructions within\n');
fprintf('  2. Install ds9 and xpa\n')
fprintf('  3. Install SExtractor, epar extractor and change ProgPath accordingly\n');
fprintf('  4. Install SWarp, epar swarp and change SWarpDir accordingly\n');
fprintf('  5. Edit the startup.m file according to your needs\n');
fprintf('\n\n');


function wget(URL)

[Status,Res]=system(sprintf('wget -q -nc %s',URL));
if (Status~=0),
    if (ispc)
        fprintf('Running wget on Windows failed\n');
        fprintf('Make sure wget is installed - e.g., http://gnuwin32.sourceforge.net/packages/wget.htm');
        error('Running wget on Windows failed'); 
    end
    if (ismac || isunix),
        fprintf('Running wget on linux/mac failed\n');
        fprintf('Make sure wget is installed');
        error('Running wget on Windows failed'); 
    end
end
    
function install_eran_matlab
%------------------------------------------------------------------------------
% install_eran_matlab function                                             www
% Description: install or update the astronomy and astrophysics packages for
%              matlab.
% Input  : null
% Output : null
% Tested : Matlab 2011b
%     By : Eran O. Ofek                    Dec 2012
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: install_eran_matlab;
% Reliable: 2
%------------------------------------------------------------------------------

fprintf('Install or update the astronomy and astrophysics packages for matlab\n');

if (isunix | ismac),
   fprintf('Linux/Unix/Mac identified - proceed with installation\n');
else
   error('Windows operating system is not supported - install manually');
end


Str = sprintf('Enter the full path of directory in which to install the package\n default is ~/matlab : ');
Res = input(Str,'s');
if (isempty(Res)),
 Res = '~/matlab';
end

mkdir(Res);
cd(Res);

Files = dir('*');
if (length(Files)>2),
   fprintf('Directory already contains files\n');
   fprintf('Installtion may delete existing files\n');
   Res = input('Are you sure you want to proceed with installation (y/[n]): ','s');

else
   Res = 'y';
end

switch lower(Res)
 case 'y'
    % wget files
    system('wget -nc http://www.weizmann.ac.il/home/eofek/matlab/startup.m');

    mkdir('fun');
    cd('fun');
    system('wget -nc http://www.weizmann.ac.il/home/eofek/matlab/AstroMatlab.tar.gz');

    system('gzip -d AstroMatlab.tar.gz');
    system('tar -xvf AstroMatlab.tar');
    delete('AstroMatlab.tar');

    cd ..

    ResD = input('Do you want to install also the data files (>2GB) ([y]/n): ','s');
    switch lower(ResD)
     case 'n'
        % skip
     otherwise
        mkdir('data');
        cd('data');
        system('wget -nc http://www.weizmann.ac.il/home/eofek/matlab/AstroMatlabData.tar.gz');
        system('gzip -d AstroMatlabData.tar.gz');
        system('tar -xvf AstroMatlabData.tar');
        delete('AstroMatlabData.tar');
        cd ..
    end
 otherwise
    % skip
end









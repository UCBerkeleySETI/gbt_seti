function wget_chandra_obsid(ObsID,varargin)
%--------------------------------------------------------------------------
% wget_chandra_obsid function                                        Swift
% Description: Get all the files associated with a Chandra ObsID
% Input  : - ObsID
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'ChandraCat' - Catalog of Chandra ObsID observations.
%                         To construct the catalog use:
%                         build_chandra_obsid_cat.m.
%                         If empty, will attempt to construct the
%                         catalog. If char array will attempt to load
%                         the catalog from a mat file, otherwise
%                         will assume the input is the catalog.
%                         Default is empty.
%            'Download' - Indicating what to download:
%                         'all' - all data (default).
%                         'primary' - only data in primary directory.
%                         'secondary' - only data in secondary directory.
%            'Output'   - Output data in a 'flat' structure' or
%                         'dir' (directory) structure.
%                         Default is 'dir'.
%            'Ungzip'   - ungzip data {true|false}. Default is true.
%            'CopyTo'   - Directory in which to cd before copying data.
%            'ReGet'    - Get files again if 'primary' dir exist 
%                         in directory {true|false}. Default is false.
%            'Extra'    - Extra parameters for wget.
%            'MaxGet'   - Maximum number of files for parallel get
%                         (see pwget.m). Default is 10.
% Output : null
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: wget_chandra_obsid(1,'ChandraCat',Cat);
% Reliable: 2
%--------------------------------------------------------------------------

DefV.ChandraCat    = [];
DefV.Download      = 'all';   % {'all'|'primary'|'secondary'}
DefV.Output        = 'dir';   % {'dir'|'flat'}
DefV.Ungzip        = true;
DefV.CopyTo        = [];
DefV.ReGet         = false;
DefV.Extra         = '';
DefV.MaxGet        = 10;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


if (isempty(InPar.ChandraCat)),
    Cat = build_chandra_obsid_cat('GetInfo',false,'Verbose',false);
else
    if (ischar(InPar.ChandraCat)),
        Cat = load2(InPar.ChandraCat);
    else
        Cat = InPar.ChandraCat;
    end
end

% Search ObsID
Iobs = find([Cat.ObsID]==ObsID);
URL  = Cat(Iobs).URL;



PWD = pwd;
if (InPar.CopyTo),
    cd(InPar.CopyTo);
end

% check if directory is populated
if (exist('primary','dir')>0 && ~InPar.ReGet),
    % do not get files again
else

    % get file names
    switch lower(InPar.Download)
        case 'all'
            List = ftp_dir_list(URL);
            I1 = find(strcmp({List.subdir},''));
            Ip = find(strcmp({List.subdir},'primary'));
            Is = find(strcmp({List.subdir},'secondary'));
        case 'primary'
            List = ftp_dir_list(sprintf('%sprimary/',URL));
            I1   = [];
            Ip   = (1:1:length(List))';
            Is   = [];
        case 'secondary'
            List = ftp_dir_list(sprintf('%ssecondary/',URL));
            I1   = [];
            Ip   = [];
            Is   = (1:1:length(List))';
        otherwise
            error('Unknown Download option');
    end


    % download
    switch lower(InPar.Output)
        case 'flat'
            pwget({List.URL},InPar.Extra,InPar.MaxGet);        
            if (InPar.Ungzip)
                system('gzip -d *.gz');
            end
        case 'dir'
            pwget({List(I1).URL},InPar.Extra,InPar.MaxGet);
            if (~isempty(Ip)),
                mkdir('primary');
                cd('primary');
                pwget({List(Ip).URL},InPar.Extra,InPar.MaxGet);
                if (InPar.Ungzip)
                    system('gzip -d *.gz');
                end
                cd ..
            end
            if (~isempty(Is)),
                mkdir('secondary');
                cd('secondary');
                pwget({List(Is).URL},InPar.Extra,InPar.MaxGet);
                if (InPar.Ungzip)
                    system('gzip -d *.gz');
                end
                cd ..
            end
       otherwise
            error('Unknown Output option');
    end

end  
cd(PWD);                        
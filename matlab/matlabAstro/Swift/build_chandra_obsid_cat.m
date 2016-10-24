function Cat=build_chandra_obsid_cat(varargin)
%--------------------------------------------------------------------------
% build_chandra_obsid_cat function                                   Swift
% Description: Construct a catalog of all Chandra observations by going
%              over the entire Chandra image archive.
% Input  : * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'GetInfo' - Get information [JD, RA, Dec] for each ObsID
%                        in addition to the OBSID and its location
%                        {true|false}. Default is true.
%           'Verbose'  - {true|false}. Default is true.
% Output : - 
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Cat=build_chandra_obsid_cat;
% Reliable: 2
%--------------------------------------------------------------------------


ArchiveURL = 'ftp://legacy.gsfc.nasa.gov/chandra/data/science/';

DefV.GetInfo  = true;
DefV.Verbose  = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


% search directories for ObsID
AOpage = urlread(ArchiveURL);
AllAO  = regexp(AOpage,'(?<AO>ao\d\d)','tokens');
Nao    = numel(AllAO);
Ind    = 0;


% search directories for ObsID
ListAO = ftp_dir_list(ArchiveURL,false);
Nao  = numel(ListAO);
for Iao=1:1:Nao,
    ArchiveURLao = ListAO(Iao).URL;
    ListCat      = ftp_dir_list(ArchiveURLao,false);
    Ncat    = numel(ListCat);
    for Icat=1:1:Ncat,
        %[Iao,Icat]
        ArchiveURLcat = ListCat(Icat).URL;
        ListObs       = ftp_dir_list(ArchiveURLcat,false);
        Tmp = regexp({ListObs.URL},'/','split');
        Nt  = numel(Tmp);
        for It=1:1:Nt,
            Ind = Ind + 1;
            Cat(Ind).ObsID = str2double(Tmp{It}{end-1});
            Cat(Ind).URL   = ListObs(It).URL;

            if (InPar.GetInfo),
                % read information from oif.fits file
                Furl = sprintf('%soif.fits',ListObs(It).URL);
                if (InPar.Verbose)
                    fprintf('Reading file: %s\n',Furl);
                end
                OIF = urlread(Furl,'Timeout',30);
                Ik  = strfind(OIF,'DATE-OBS=');
                if (~isempty(Ik)),
                    Cat(Ind).StartJD = julday(OIF(Ik(1)+11:Ik(1)+29));
                end
                Ik  = strfind(OIF,'DATE-END=');
                if (~isempty(Ik)),
                    Cat(Ind).EndJD = julday(OIF(Ik(1)+11:Ik(1)+29));
                end
                Ik  = strfind(OIF,'INSTRUME=');
                if (~isempty(Ik)),
                    Cat(Ind).Instrum = OIF(Ik(1)+11:Ik(1)+18);
                end
                Ik  = strfind(OIF,'RA_NOM  =');
                if (~isempty(Ik)),
                    Cat(Ind).RA = str2double(OIF(Ik(1)+11:Ik(1)+30));
                end
                Ik  = strfind(OIF,'DEC_NOM =');
                if (~isempty(Ik)),
                    Cat(Ind).Dec = str2double(OIF(Ik(1)+11:Ik(1)+30));
                end
                Ik  = strfind(OIF,'EXPOSURE=');
                if (~isempty(Ik)),
                    Cat(Ind).ExpTime = str2double(OIF(Ik(1)+11:Ik(1)+30));
                end
            end
            
            if (InPar.Verbose),
                fprintf('ObsID=%d   %d  %d\n',Cat(Ind).ObsID,Iao,Icat);
            end
        end
    end
end    


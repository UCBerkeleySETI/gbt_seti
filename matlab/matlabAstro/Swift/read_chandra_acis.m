function Cat=read_chandra_acis(ObsID,varargin)
%--------------------------------------------------------------------------
% read_chandra_acis function                                         Swift
% Description: Read Chandra ACIS event files associated with a Chandra
%              ObsID and add columns containing the RA and Dec for
%              each event. If needed cd to osbid directory.
% Input  : - Observation ID (numeric) or directory (string) in which to
%            look for event file. If empty, assume the evt file is in
%            the current directory. Default is empty.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs: wget_chandra_obsid.m
% Output : - Structure array of ACIS events catalogs.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Cat=read_chandra_acis(ObsID);
% Reliable: 2
%--------------------------------------------------------------------------


if (nargin==0),
    ObsID = [];
end

%InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

PWD = pwd;
if (ischar(ObsID)),
    % cd to data directory
    cd(ObsID);
else
    % get data
    if (isempty(ObsID)),
        % assume data is in current directory
    else
        wget_chandra_obsid(ObsID,varargin{:});
    end
end

if (isdir('./primary')),
    cd('primary');
end
% Look for evt file
FileEvt = dir('*_evt2.fits*');
Nf      = numel(FileEvt);
if (Nf==0),
    %error('More than one evt2 file');
    Cat = [];
else
        
    % ungzip if needed
    if (any(strcmp(FileEvt(1).name(end-2:end),'.gz'))),
        system_list('gzip -d %s',{FileEvt.name});
    end
    FileEvt = dir('*_evt2.fits');
    Nf      = numel(FileEvt);

    for If=1:1:Nf,

        % read evt2 file
        Head  = fits_get_head(FileEvt(If).name,1);
        Iinst = strcmp(Head(:,1),'INSTRUME');
        if (strcmp(spacedel(Head{Iinst,2}),'ACIS')),
            [Cat,Col]=read_fitstable(FileEvt(If).name);

            %[~,~,ColCell,Col,Table]=get_fitstable_col(FileEvt.name);
            %TableEvt = [Table{Col.time}, Table{Col.ccd_id}, Table{Col.node_id},Table{Col.expno},Table{Col.chipx},Table{Col.chipy},Table{Col.tdetx},Table{Col.tdety},Table{Col.detx},Table{Col.dety},Table{Col.x},Table{Col.y},Table{Col.pha},Table{Col.energy},Table{Col.pi},Table{Col.fltgrade},Table{Col.grade}];
            [RA,Dec]=swxrt_xy2coo(FileEvt(If).name,Cat.Cat(:,Col.x),Cat.Cat(:,Col.y),'chandra');
            Cat.Cat = [Cat.Cat, RA, Dec];
            Cat.ColCell(end+1:end+2) = {'RA','Dec'};
            Cat.Col     = cell2struct(num2cell(1:1:length(Cat.ColCell)),Cat.ColCell,2);
        else
            Cat = [];
        end
    end
end

% cd to original directory
cd(PWD);
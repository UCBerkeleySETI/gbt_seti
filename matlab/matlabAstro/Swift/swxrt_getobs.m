function [Link,Table,Error,ErrDown,Src]=swxrt_getobs(RA,Dec,SearchRadius,Save,Extract)
%-----------------------------------------------------------------------------
% swxrt_getobs function                                                 Swift
% Description: Look for Swift/XRT observations taken in a given coordinates.
%              Return a link to the directories containing the observations,
%              and retrieve the cleaned "pc" event file for each observation.
%              Instelation: (1) install "browse_extract_wget.pl" from
%	       http://heasarc.gsfc.nasa.gov/W3Browse/w3batchinfo.html
%              (2) Set the RelPath variable in this program to specify
%              the path of "browse_extract_wget.pl" script relative
%              to the path of this program.
% Input  : - J2000.0 R.A. in radians or [H M S], or sexagesimal format.
%            Alternatively, if the second argument containing a name resolver
%            than this can be a string containing a target name (e.g., "M31").
%          - J2000.0 Dec. in radians or [Sign D M S], or sexagesimal format.
%            Alternativly, this can be a name resolver
%            {'SIMBAD' | 'NED'}, default is 'SIMBAD'.
%          - Search radius [arcmin]. Default is 60 arcmin.
%          - Save the event file option:
%            'no' - do not save the event file.
%            'pc' - retrieve only the cleaned "pc" event file. Default.
%            'all' - retrieve all the files.
%          - Extract a source count rate {'y' | 'n'}. Default is 'n'.
%            If a source position is specified then the program will
%            call swxrt_src.m and attemp to measure the count rate
%            in the source position.
% Output : - Structure array of links to directories containing the
%            obsevations, and names of the cleaned "pc" event file in
%            each directory. The structure contsins the following fields:
%            .Dir    - URL of the directory containing the files for
%                      a single observation.
%            .PC_EventFileName - PC event file name
%            .Link   - Link to the PC event file name
%            .All    - A cell array of all file names found in the
%                      directory.
%            .AllFullPath - A cell array of all full URLs for files
%                      found in the directory.
%            Note that this structure is containing only unique entries.
%          - Table of observation. A structure containing two fields:
%            .S - A cell array in which each cell contains a single
%                 column in the table of observations of the specific
%                 source searched by this script.
%            .Col - A structure containing the table column names and
%                   index.
%          - A structure array containing a list of entries in the table
%            which were not found in the Swift FTP archive
%            (e.g., the images were not processed yet).
%            The structure contains the following fields:
%            .TableInd - index of entry in the table of observations.
%            .URL      - The non exsiting URL of the directory.
%          - A cell arraycontaining a list of entries in the table
%            which the program was not able to download.
%          - A structure containing the extracted source count rate
%            information. See swxrt_src.m for details.
% Tested : Matlab 7.13
%     By : Eran O. Ofek                    Sep 2012
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Link,T,E,ED]=swxrt_getobs(1,1,60);
%          [Link,T,E,ED]=swxrt_getobs('10:00:00.0','+41:00:00',60);
%          [Link,T,E,ED]=swxrt_getobs('M31','NED',60);
% Reliable: 2
%-----------------------------------------------------------------------------

Dir = which_dir(mfilename);  % PATH of this program
RelPath = '/../bin/HEASARC/browse_extract_wget.pl'; % relative path for browse_extract_wget.pl

RAD     = 180./pi;


Def.SearchRadius = 60;     % [arcmin]
Def.Save         = 'pc';
Def.Extract      = 'n';
if (nargin==2),
   SearchRadius = Def.SearchRadius;
   Save         = Def.Save;
   Extract      = Def.Extract;
elseif (nargin==3),
   Save         = Def.Save;
   Extract      = Def.Extract;
elseif (nargin==4),
   Extract      = Def.Extract;
elseif (nargin==5),
   % do nothing
else
   error('Illegal number of input arguments');
end


Name = [];
if (isstr(Dec)),
   if (max(strcmp({'simbad','ned'},lower(Dec)))),
      Name = RA;
      Resolver = Dec;
   end
end

if (isempty(Name)),
   RA  = convertdms(RA,'gH','r');
   Dec = convertdms(Dec,'gD','r');
end



switch isempty(Name)
 case 1
    RA      = RA.*RAD;  % convert rad to deg
    Dec     = Dec.*RAD; % convert rad to deg
    Command = sprintf('%s%s table=swiftxrlog position=%08.5f,%08.5f radius=%05.1f format=text',Dir,RelPath,RA,Dec,SearchRadius);
 case 0
    Command = sprintf('%s%s table=swiftxrlog position=%s name_resolver=%s radius=%05.1f format=text',Dir,RelPath,Name,upper(Resolver),SearchRadius);
 otherwise
    error('Unknown isnan(Name) option');
end

BaseURL = 'http://heasarc.gsfc.nasa.gov/FTP/swift/data/obs/';


Link    = [];
Table   = [];
Error   = [];
ErrDown = [];

[Sys,Res] = system(Command);

I   = strfind(Res,'_Search_Offset');
Ip  = strfind(Res,'|');
if (isempty(Ip)),
   % No observations were found
else
   Ipp = find(Ip>I,1);
   Res = Res(Ip(Ipp)+1:end);
   
   S=textscan(Res,'%s %s %s %s %s %s %d %s %s %s %s %s %[^\n]','Delimiter','|');

   Col.TargetID = 2;
   Col.ObsID    = 3;
   Col.RA       = 4;
   Col.Dec      = 5;
   Col.StartT   = 6;
   Col.ExpTime  = 7;
   Col.WinSize  = 8;
   Col.OperationMode = 9;
   Col.PointingMode  = 10;
   Col.FlipFlop      = 11;
   Col.SearchOffset  = 12;
   
   
   JD_StartT = julday(strrep(S{Col.StartT},' ','T'));  % JD of start time
   Date      = jd2date(JD_StartT);
   Table.S   = S;
   Table.Col = Col; 
   
   IndED = 0;
   ErrI = 0;
   Error = [];
   Link = [];
   ErrDown = {};
   for I=1:1:length(JD_StartT),
      DirURL = sprintf('%s%04d_%02d/%s/xrt/event/',BaseURL,Date(I,3),Date(I,2),S{Col.ObsID}{I});
      try
         Str=urlread(DirURL);
      
         % e.g., http://heasarc.gsfc.nasa.gov/FTP/swift/data/obs/2011_06/00031306015/xrt/event/

         XRT_FileName = sprintf('sw%s%s',S{Col.ObsID}{I},'xpcw3po_cl.evt.gz');
      
         if (~isempty(strfind(Str,XRT_FileName))),
            Link(I).Dir  = sprintf('%s',DirURL);
            Link(I).PC_EventFileName  = sprintf('%s',XRT_FileName);
            Link(I).Link = sprintf('%s%s',DirURL,XRT_FileName);
         else
            % level 3 doesnt exist - try to get level 2
            XRT_FileName = sprintf('sw%s%s',S{Col.ObsID}{I},'xpcw2po_cl.evt.gz');
   
            if (~isempty(strfind(Str,XRT_FileName))),
               Link(I).Dir  = sprintf('%s',DirURL);
               Link(I).PC_EventFileName  = sprintf('%s',XRT_FileName);
               Link(I).Link = sprintf('%s%s',DirURL,XRT_FileName);
            end
         end
   
         RE = regexp(Str,'<a href="sw.{25,35}">','match');
         All = {};
         AllFullPath = {};
         for Ire=1:1:length(RE),
            All{Ire}         = RE{Ire}(10:end-2);
            AllFullPath{Ire} = sprintf('%s%s',Link(I).Dir,RE{Ire}(10:end-2));
         end
         Link(I).All = All;
         Link(I).AllFullPath = AllFullPath;

      catch
         ErrI = ErrI + 1;
         Error(ErrI).TableInd = I;
         Error(ErrI).URL = DirURL;
      end

   end
end

% multiple exposures under the same ObsID are combined
% so select only uniqe ObsIDs:
if (isempty(Link)),
   % do nothing - no need to save anything
else
   AllLinks  = {Link.Link};
   AllLinks  = AllLinks(find(isempty_cell(AllLinks)==0));
   [Un,IndUn]=unique(AllLinks);
   Link = Link(IndUn);
   
   
   switch lower(Save)
    case 'pc'
       AllLinks = {Link.Link};
       AllLinks  = AllLinks(find(isempty_cell(AllLinks)==0));
       Status = mwget(AllLinks);
       ErrDown = {AllLinks{boolean(Status)}};
    case 'no'
       % do nothing
    case 'all'
       AllLinks = [Link.AllFullPath];
       AllLinks  = AllLinks(find(isempty_cell(AllLinks)==0));
       Status = mwget(AllLinks);
       ErrDown = {AllLinks{boolean(Status)}};
    otherwise
        error('Unknown Save option');
   end
end



switch lower(Extract)
 case 'n'
    % do nothing
    Src = [];
 case 'y'
    error('The Extract option is not implemented - use swxrt_src.m');

%    Nobs = length(Link);
%    for Iobs=1:1:Nobs,
%       Src(Iobs) = swxrt_src(Link(Iobs).PC_EventFileName,RA,Dec);
%    end
 otherwise
    error('Unknown Extract option');
end

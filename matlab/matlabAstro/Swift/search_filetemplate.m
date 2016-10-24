function [Links,Files]=search_filetemplate(Template,SearchDir,OnlyDirName)
%-------------------------------------------------------------------------
% search_filetemplate function                                    General
% Description: Given a directory name, search recursively within the
%              directory and all its subdirectories, file names with a
%              specific template. File path that contains a specific
%              string can be excluded
% Input  : - File template to search (e.g., 'sw*_sk.im*')
%          - Directory in which to search for the file template, all sub
%            directories will be searched recursively.
%            Default is empty matrix.
%            If empty matrix then use present working directory.
%          - A string that will matched against each directory.
%            Only directories which have this string in their name will
%            be searched. Default is empty matrix. If empty matrix, then
%            do not constrain directory name.
% Output : - Cell array of full path per each file that was found.
%          - Cell arrya of file names.
% Tested : Matlab 2011b
%     By : Eran O. Ofek                    Feb 2013
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Links,Files] = search_filetemplate('sw*_sk.im*',[],'uvot');
% Reliable: 2
%-------------------------------------------------------------------------


Def.SearchDir   = [];
Def.OnlyDirName = [];
if (nargin==1),
    SearchDir   = Def.SearchDir;
    OnlyDirName = Def.OnlyDirName;
elseif (nargin==2),
    OnlyDirName = Def.OnlyDirName;
elseif (nargin==3),
    % do nothing
else
    error('Illegal number of input arguments');
end

if (isempty(SearchDir)),
    SearchDir = pwd;
end


Dir      = dir(sprintf('%s%s*',SearchDir,filesep));
if (isempty(Dir)),
    Files = {};
    Links = {};
else
   if (~isempty(OnlyDirName)),
      if (~isempty(findstr(SearchDir,OnlyDirName))),
          DirOK = true;
      else
          DirOK = false;
      end
   else
       DirOK = true;
   end
   if (DirOK),
      FilesD   = dir(sprintf('%s%s%s',SearchDir,filesep,Template));
      Files    = {FilesD.name}.';
      Links{1} = SearchDir;
      Links    = repmat(Links,length(Files),1);
   else
      Files = {};
      Links = {}; 
   end
   
   for Idir=1:1:length(Dir),
       if (Dir(Idir).isdir && ~strcmp(Dir(Idir).name,'.') && ~strcmp(Dir(Idir).name,'..')),
           LinksAdd = sprintf('%s%s%s',SearchDir,filesep,Dir(Idir).name);
          
           [LinksAdd,FilesAdd] = search_filetemplate(Template,LinksAdd,OnlyDirName);
           
           Files = [Files; FilesAdd];
           Links = [Links; LinksAdd];
       end
   end
end


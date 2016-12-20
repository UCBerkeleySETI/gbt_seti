function [Groups,GroupStr,GroupInd,ImListCell]=keyword_grouping1_fits(ImList,Keyword);
%----------------------------------------------------------------------------
% keyword_grouping1_fits function                                    Imbasic
% Description: Group images by a single image header keyword.
%              For example, can be used to select all images taken with
%              a given filter.
%              see also: keyword_grouping_fits.m
% Input  : - List of images (see create_list.m for details).
%            Default is '*.fits'. If empty then use default.
%          - String or cell array of strings in which each string is an
%            header keyword by which to group the images.
%            If several keywords are given, then the grouping is done
%            by the first keyword which exist in the header.
%            Default is {'FILTER','FILTER1'}.
% Output : - Vector in which each cell contains a structure with information
%            about each group.
%            For example:
%               Group(1).List  - contains a cell array of all the images
%                                in the group.
%               Group(1).Ind   - contains the indices of the images in
%                                the original list.
%               Group(1).Group - String containing group name.
%          - Cell array containing all the group names.
%          - Cell array in which each cell containing the indices of the
%            images in each group in the original list.
%          - Cell array of the original list of images.
%            in the
% Tested : Matlab 7.10
%     By : Eran O. Ofek                       Jul 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [Groups,GroupStr,GroupInd,ImListCell]=keyword_grouping1_fits('b_*.fits');
%          [Groups,GroupStr,GroupInd,ImListCell]=keyword_grouping1_fits('b_*.fits','FILTERSL');
% Reliable: 2
%----------------------------------------------------------------------------

Def.ImList  = '*.fits';
Def.Keyword = {'FILTER','FILTER1'};

if (nargin==0),
   ImList   = Def.ImList;
   Keyword  = Def.Keyword;
elseif (nargin==1),
   Keyword  = Def.Keyword;
elseif (nargin==2),
   % do nothing
else
   error('Illegal number of input arguments');
end

if (isempty(ImList)),
   ImList   = Def.ImList;
end

if (~iscell(Keyword)),
   Keyword = {Keyword};
end

[~,ImListCell] = create_list(ImList,NaN);
Nim = length(ImListCell);

ImKey = cell(Nim,1);
for Iim=1:1:Nim,
   KeywordVal = get_fits_keyword(ImListCell{Iim},Keyword);

   % find firt exsiting keyword
   Ik = find(isnan_cell(KeywordVal)==0,1);
   if (isempty(Ik)),
      error('Keywords not exist in image header');
   else
      ImKey{Iim} = KeywordVal{Ik};
   end
end

[GroupStr,GroupInd] = group_cellstr(ImKey,'n');
Ng = length(GroupStr);
%Groups = zeros(Ng,1);
for Ig=1:1:Ng,
   Groups(Ig).List  = ImListCell(GroupInd{Ig});
   Groups(Ig).Ind   = GroupInd{Ig};
   Groups(Ig).Group = GroupStr{Ig};
end


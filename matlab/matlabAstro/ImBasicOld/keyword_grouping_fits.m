function [Unique,UniqueCol,MatCell,ImListCell]=keyword_grouping_fits(ImList,Keys,KeysFlag);
%----------------------------------------------------------------------------
% keyword_grouping_fits function                                     ImBasic
% Description: Group images by multiple image header keywords.
%              For example, can be used to select all images taken with
%              a given filter and type.
%              see also: keyword_grouping1_fits.m
% Input  : - List of images (see create_list.m for details).
%            Default is '*.fits'. If empty then use default.
%            Alternatively, it can be a structure array (e.g., A(1).B...)
%            in which each array element corresponds to an image
%            and each field corresponds to a keyword.
%          - String or cell array of strings in which each string is an
%            header keyword by which to group the images.
%            If several keywords are given, then the grouping is done
%            by the combination of all keywords.
%            Default is {'FILTER'}.
%          - Vector of flags (1) or (0) indicating which Keys to use
%            in the 'Keys' cell array. Default is all ones.
% Output : - 'Unique' structure array of unique lines.
%            Each element coresponds to a unique line and contains two fields:
%            .Line - The values in the unique line, where strings were converted
%                    to numeric values.
%            .Ind - contains the indices of the unique lines.
%          - 'UniqueCol' is a structure array containing the unique
%            values in each column.
%          - 'MatCell' is the input matrix converted to numeric values.
%          - A cell array of all the imput images.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                       Jul 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [Unique,UniqueCol,MatCell,ImListCell]=keyword_grouping_fits('b_*.fits',{'FILTER1','FILTER2','FILTER3'});
%          or the following will use only FILTER1 and FILTER2 keywords:
%          [Unique,UniqueCol,MatCell,ImListCell]=keyword_grouping_fits('b_*.fits',{'FILTER1','FILTER2','FILTER3'},[1 1 0]);
% Reliable: 2
%----------------------------------------------------------------------------

Def.ImList  = '*.fits';
Def.Keys    = {'FILTER'};
Def.KeysFlag = [];

if (nargin==0),
   ImList   = Def.ImList;
   Keys     = Def.Keys;
   KeysFlag = Def.KeysFlag;
elseif (nargin==1),
   Keys     = Def.Keys;
   KeysFlag = Def.KeysFlag;
elseif (nargin==2),
   KeysFlag = Def.KeysFlag;
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end

if (isempty(ImList)),
   ImList   = Def.ImList;
end

if (~iscell(Keys)),
   Keys = {Keys};
end

if (isempty(KeysFlag)),
   KeysFlag = ones(length(Keys),1);
end


if (~isstruct(ImList))
   [~,ImListCell] = create_list(ImList,NaN);

   [~,KeyVal] = mget_fits_keyword(ImList,Keys);
else
   KeyVal     = ImList;
   ImListCell = {};
end


Nim = length(KeyVal);

%---------------------------------
%--- construct all dark frames ---
%---------------------------------
Nkeys = length(Keys);

Iuse  = find(KeysFlag);
Nuse  = length(Iuse);
Prop  = cell(Nim,Nuse);

Iu = 0;
for Ikeys=1:1:Nkeys,
   if (KeysFlag(Ikeys)),
      Iu = Iu + 1;
      [Prop{1:Nim,Iu}] = deal(KeyVal.(Keys{Ikeys}));
   end
end

[Unique,UniqueCol,MatCell]=unique_cell_grouping(Prop);


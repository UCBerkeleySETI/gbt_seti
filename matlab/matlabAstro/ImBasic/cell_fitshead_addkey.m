function [NewCellHead]=cell_fitshead_addkey(CellHead,varargin)
%--------------------------------------------------------------------------
% cell_fitshead_addkey function                                    ImBasic
% Description: A utility program to add new keywords, values and
%              comments to a cell array containing A FITS header
%              information.
%              The FITS header cell array contains an arbitrary number
%              of rows and 3 columns, where the columns are:
%              {keyword_name, keyword_val, comment}.
%              The comment column is optional.
% Input  : - Cell array containing the FITS header information.
%          * An header cell array to add at the end of the existing
%            array (e.g., cell_fitshead_addkey(CellHead,AddHead);).
%
%            Alternatively, vecor of positions in the existing header
%            array in which to add the new keys
%            (e.g., cell_fitshead_addkey(CellHead,NewPos,AddHead);).
%
%            Alternatively, triplets of :...,key,value,comment,...
%            to add at the end of the existing header
%            (e.g., cell_fitshead_addkey(CellHead,'NAXIS',1024,'number of axes');)
%
%            Alternatively, quadraplets of :...,pos,key,value,comment,...
%            to add at a given position in the existing header specified by
%            the integer pos.
%            (e.g., cell_fitshead_addkey(CellHead,Pos,'NAXIS',1024,'number of axes');)
% Output : - The new header cell array.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%--------------------------------------------------------------------------
if (isempty(CellHead)),
   CellHead = cell(0,3);
end

Narg = length(varargin);
if (Narg==0),
   % Do nothing
   AddHead = [];
elseif (Narg==1),
   AddHead = varargin{1};
   VecPos  = zeros(size(AddHead,1),1)+Inf;
elseif (Narg==2),
   VecPos     = varargin{1};
   AddHead = varargin{2};
else
   if (Narg./3==floor(Narg./3) && ischar(varargin{1})==1),
      % assume triplets: ...,key,value,comment,...
      Counter = 0;
      AddHead = cell(Narg./3,3);
      VecPos  = zeros(Narg./3,1) + Inf;
      for Iarg=1:3:Narg,
         Counter = Counter + 1; 
         [AddHead{Counter,1:3}] = deal(varargin{Iarg:1:Iarg+2});
      end
   elseif (Narg./4==floor(Narg./4) && isnumeric(varargin{1})==1),
      % assume quadraplets: ...,pos,key,value,comment,...
      Counter = 0;
      AddHead = cell(Narg./4,3);
      VecPos  = zeros(Narg./4,1);
      for Iarg=1:4:Narg,
         Counter = Counter + 1; 
         VecPos(Counter) = varargin{Iarg};
         [AddHead{Counter,1:3}] = deal(varargin{Iarg+1:1:Iarg+3});
      end
   else
      error('Unknown format of additional parameters');
   end
end

[~,Ncr] = size(CellHead);
[~,Nar] = size(AddHead);

if (Ncr==Nar),
   % do nothing
else
   if (Ncr==2 && Nar==3),
      CellHead{1,3} = [];
   elseif (Ncr==3 && Nar==2),
      AddHead{1,3} = [];
   else
      error('Illegal number of columns');
   end
end


% sort AddHead by VecPos
NewCellHead = CellHead;

[SortedVecPos,SortedInd] = sort(VecPos);
SortedAddHead = AddHead(SortedInd,:);
Nadd = length(SortedVecPos);
for Iadd=1:1:Nadd,
   NewCellHead = insert_ind(NewCellHead,SortedVecPos(Iadd)+(Iadd-1),SortedAddHead(Iadd,:));
end

NewCellHead = cell_fitshead_fix(NewCellHead);

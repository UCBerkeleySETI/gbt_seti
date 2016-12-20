function [Unique,UniqueCol,MatCell]=unique_cell_grouping(Cell);
%-----------------------------------------------------------------------------
% unique_cell_grouping function                                       General
% Description: Given a cell matrix containing eithr strings or numeric
%              values, find unique lines in the cell matrix.
% Input  : - 2-D cell array.
% Output : - Structure array of unique lines.
%            Each element coresponds to a unique line and contains two fields:
%            .Line - The values in the unique line, where strings were converted
%                    to numeric values.
%            .Ind - contains the indices of the unique lines.
%          - A structure array containing the unique
%            values in each column.
%          - The input matrix converted to numeric values.
% Tested : matlab 7.11
%     By : Eran O. Ofek                      June 2011
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: B={'a','b',[1];'a','b',[1];'c','c',[1]}
%          [Unique,UniqueCol,MatCell]=unique_cell_grouping(B);
% Reliable: 2
%-----------------------------------------------------------------------------

[Nline,Ncol] = size(Cell);
MatCell = zeros(Nline,Ncol);

for Icol=1:1:Ncol,
   if (~ischar(Cell{1,Icol})),
      % numeric
      MatCell(:,Icol) = [Cell{:,Icol}].';  %'
      UniqueCol(Icol).Val = unique(MatCell(:,Icol));
   else
      % strings
      % replace strings by unique numbers
      UniqueCol(Icol).Val = unique(Cell(:,Icol));
      for Iun=1:1:length(UniqueCol(Icol).Val),
	 MatCell(strcmp(Cell(:,Icol),UniqueCol(Icol).Val{Iun}),Icol) = Iun;
      end
   end
end

UniqueMatCell = unique(MatCell,'rows');
Nun = size(UniqueMatCell,1);
for Iun=1:1:Nun,
   Unique(Iun).Ind = find(sum(bsxfun(@eq,MatCell,UniqueMatCell(Iun,:)),2)==Ncol);
   Unique(Iun).Line = UniqueMatCell(Iun,:);
end

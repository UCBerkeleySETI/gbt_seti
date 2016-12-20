function [MatchList]=match_lists_index(List1,List2,Columns)
%--------------------------------------------------------------------------
% match_lists_index function                                       General
% Description: Match the lines in two matrices according to the values in
%              one of the columns in each matrix.
% Input  : - First list to match
%          - Second list to match
%          - Columns to match, default is index matching with
%            the first columns.
%            In case two numbers are given, its taken as the column in List1
%            and in List2, respectively.
% Output : - Matched list structure containing the following fields:
%            .MatchList1 - Matrix containing the lines of List1 which are
%                          matched to List2.
%            .MatchList2 - Matrix containing the lines of List2 which are
%                          matched to List1.
%            .I1         - Indices of lines in List1 which are
%                          matched to List2.
%            .I2         - Indices of lines in List1 which are
%                          matched to List1.
%            .In1        - Indices of lines in List1 which are
%                          not matched to List2.
%            .In2        - Indices of lines in List1 which are
%                          not matched to List`.
% Example: [MatchList]=match_lists_index(List1,List2,[1 1]);
% Tested : Matlab 7.13
%     By : Eran O. Ofek                  December 2000
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: L1 = [[1;2;3;4;5], rand(5,1)];
%          L2 = [[1;3;5;6;8;9], rand(6,1)];
%          [MatchList,UnMatch1,UnMatch2,I1,I2,In1,In2]=match_lists_index(L1,L2);
% Reliable: 2
%--------------------------------------------------------------------------
Def.Columns = [1 1];
if (nargin==2),
   Columns = Def.Columns;
end

if (length(Columns)==1),
   Columns = [Columns, Columsn];
end

Col1 = Columns(1);
Col2 = Columns(2);

N1 = size(List1,1);
N2 = size(List2,1);

[~,MatchList.I1,MatchList.I2] = intersect(List1(:,Col1),List2(:,Col2));
MatchList.MatchList1 = List1(MatchList.I1,:);
MatchList.MatchList2 = List2(MatchList.I2,:);
MatchList.In1 = setdiff([1:1:N1].',MatchList.I1);
MatchList.In2 = setdiff([1:1:N2].',MatchList.I2);


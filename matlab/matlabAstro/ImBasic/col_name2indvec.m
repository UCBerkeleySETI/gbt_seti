function [Ind,SubsetNames]=col_name2indvec(Superset,Subset,CS)
%--------------------------------------------------------------------------
% col_name2indvec function                                         ImBasic
% Description: Given a cell array of names and another cell array
%              containing a subset of column names, return the indices
%              of the subset columns in the superset cell array.
% Input  : - A cell array (superset) of names.
%            E.g., {'A','B','C','D'}.
%          - A subset cell array of names.
%            E.g., {'A',C'}.
%            Alternatively, this can be a vector of column indices.
%            If numeric empty then will return all the columns in the
%            superset. If cell rmpty then will return empty indices.
%            Default is empty numeric (i.e., return all indices).
%          - Case sensitive {true|false}. Default is false.
% Output : - Vector of indices of subset names in the superset.
%            Index is NaN if not found.
%          - Cell array of subset names.
% See also: col_name2ind.m
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Feb 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Ind,SN]=col_name2indvec({'A','B','C','D'},{'A','D'})
%          [Ind,SN]=col_name2indvec({'A','B','C','D'},[1 4])
%          [Ind,SN]=col_name2indvec({'A','B','C','D'},{})
%          [Ind,SN]=col_name2indvec({'A','B','C','D'},[])
% Reliable: 2
%--------------------------------------------------------------------------

if (nargin==1),
    Subset = [];
    CS     = false;
elseif (nargin==2),
    CS     = false;
elseif (nargin==3),
    % do nothing
else
    error('Illegal number of input arguments');
end

if (CS),
    FunStrCmp = @strcmp;
else
    FunStrCmp = @strcmpi;
end

Nsuper = numel(Superset);
if (isempty(Subset) && isnumeric(Subset)),
    Ind = (1:1:Nsuper);
elseif (isempty(Subset) && iscell(Subset)),
    Ind = [];
elseif (isnumeric(Subset)),
    Ind = Subset;
else
    Nsub   = numel(Subset);
    Ind    = zeros(1,Nsub);
    for Isub=1:1:Nsub,
        IndF = find(FunStrCmp(Subset{Isub},Superset));
        if (isempty(IndF)),
            Ind(Isub) = NaN;
        else
            Ind(Isub) = IndF;
        end
    end
end

SubsetNames = Superset(Ind(~isnan(Ind)));
        
    
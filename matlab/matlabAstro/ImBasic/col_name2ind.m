function [ColCell,varargout]=col_name2ind(Col,varargin)
%--------------------------------------------------------------------------
% col_name2ind function                                            ImBasic
% Description: Convert a columns index structure to cell of column names,
%              and convert specific column names to indices.
% Input  : - Either a cell array of column names (e.g., {'X','Y','Mag'}).
%            or a structure of column names/index (e.g., Col.X=1, Col.Y=2).
%          * Arbitrary number of arguments. These can be, either a scalar
%            (column index, including empty), or a string containing column 
%            name.
% Output : - Cell array of columns name.
%          * The column index (output per input).
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% See also: col_name2indvec.m
% Example: [ColCell,Col.X,Col.Y]=col_name2ind({'X','Y','Mag'},'X',2);
%          Col.X = 1; Col.Y = 2; Col.Mag = 3;
%          [ColCell,ColMag]=col_name2ind(Col,'Mag');
% Reliable: 2
%--------------------------------------------------------------------------


if (iscell(Col)),
    ColCell = Col;
elseif (isstruct(Col)),
    % generate cell
    FN  = fieldnames(Col);
    Nfn = numel(FN);
    ColCell = cell(1,Nfn);
    for Ifn=1:1:Nfn,
        ColCell{Col.(FN{Ifn})} = FN{Ifn};
    end    
else
    error('Unknown Col type option');
end

Narg = length(varargin);
varargout = cell(1,Narg);
for Iarg=1:1:Narg,        
    if (isnumeric(varargin{Iarg})),
        % numeric or empty
        varargout{Iarg} = varargin{Iarg};
    elseif (ischar(varargin{Iarg})),
        % string column name
        varargout{Iarg} = find(strcmp(ColCell,varargin{Iarg}));
        if (length(varargout{Iarg})>1),
            error('Two identical column names');
        end
    else
        error('Unknown input option');
    end
end

        
        
    
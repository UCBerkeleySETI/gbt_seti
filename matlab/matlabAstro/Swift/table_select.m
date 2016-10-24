function Flag=table_select(Cat,varargin)
%--------------------------------------------------------------------------
% table_select function                                              Swift
% Description: Given a table (e.g., X-ray events), select events based on
%              selection criteria.
% Input  : - Structure array containing the following fields:
%            .Cat - matrix of X-ray photons table.
%            .Col - structure of column names and indices.
%            Alternatively this can be a table class. 
%            Alternatively a cell array of {Cat,Col}.
%            Alternatively a matrix.
%          * Arbitrary number of pairs of arguments, each containing
%            a cell array of {ColumnName, Function, Value}.
%            For example, {'x',@gt,100} select photons for which the
%            colume 'x' values are larger then 100.
%            If the first input argument is a matrix then ColumnName is
%            a column index (e.g., {1,@gt,100}).
% Output : - Flag for rows satisfying the conditions.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: F=table_select(Cat,{'chipx',@gt,50},{'chipx',@lt,1024-50},{'chipy',@gt,50},{'chipy',@lt,1024-50},{'ccd_id',@eq,1},{'energy',@gt,200},{'energy',@lt,10000});
%          F=table_select(Cat,{1,@gt,50},{2,@lt,1024-50});
% Reliable: 2
%--------------------------------------------------------------------------

Narg = numel(varargin);

if (isstruct(Cat) || issim(Cat)),
    N = size(Cat.Cat,1);
elseif (istable(Cat)),
    N = size(Cat,1);
elseif (iscell(Cat)),
    N = size(Cat{1},1);
elseif (isnumeric(Cat)),
    N = size(Cat,1);
else
    error('Illegal Cat input type');
end
Flag = false(N,Narg);
    
    
for Iarg=1:1:Narg,
    Fun          = varargin{Iarg}{2};
    Val          = varargin{Iarg}{3};
    if (isstruct(Cat) || issim(Cat)),
        ColInd       = Cat.Col.(varargin{Iarg}{1});
        Flag(:,Iarg) = Fun(Cat.Cat(:,ColInd),Val);
    elseif (istable(Cat)),
        Flag(:,Iarg) = Fun(Cat.(varargin{Iarg}{1}),Val);
    elseif (iscell(Cat)),
        ColInd       = Cat{2}.(varargin{Iarg}{1});
        Flag(:,Iarg) = Fun(Cat.Cat(:,ColInd),Val);
    elseif (isnumeric(Cat)),
        ColInd       = varargin{Iarg}{1};
        Flag(:,Iarg) = Fun(Cat.Cat(:,ColInd),Val);
    else
        error('Illegal Cat input type');
    end
end
Flag = all(Flag,2);

    
    
    
    
    
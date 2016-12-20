function [BadF]=filter_badtimes(Cat,BadTimes,TimeCol)
%--------------------------------------------------------------------------
% filter_badtimes function                                           Swift
% Description: Given a structure array containing X-ray event catalogs
%              and a list of bad times, remove the bad times from the list.
% Input  : - Structure array containing the following fields:
%            .Cat - matrix of X-ray photons table.
%            .Col - structure of column names and indices.
%            Alternatively this can be a table class. 
%            Alternatively a cell array of {Cat,TimeColumnIndex}.
%            Alternatively a matrix (in this case time column is assumed to
%            be 1).
%            The program assumes the time column name is 'time'.
%          - Matrix of bad time ranges [Start End] to remove.
%          - Time column. Either a time columne name or index.
%            Default is 'time'.
% Output : - Flag indicating rows at bad times.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [GoodI,BadI]=filter_badtimes(Cat,BadTimes);
% Reliable: 2
%--------------------------------------------------------------------------

if (nargin<3),
    TimeCol = 'time';
end

Nbt = size(BadTimes,1);
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
BadFlag = false(N,Nbt);

for Ibt=1:1:Nbt,
    if (isstruct(Cat) || issim(Cat)),
        if (isnumeric(TimeCol)),
            ColInd = TimeCol;
        else
            ColInd         = Cat.Col.(TimeCol);
        end
        BadFlag(:,Ibt) = Cat.Cat(:,ColInd) >= BadTimes(Ibt,1) & Cat.Cat(:,ColInd) < BadTimes(Ibt,2);
    elseif (istable(Cat)),
        BadFlag(:,Ibt) = Cat.Cat.(TimeCol) >= BadTimes(Ibt,1) & Cat.Cat(TimeCol) < BadTimes(Ibt,2);
    elseif (iscell(Cat)),
        if (isnumeric(TimeCol)),
            ColInd = TimeCol;
        else
            ColInd         = Cat{2}.(TimeCol);
        end
        BadFlag(:,Ibt) = Cat.Cat(:,ColInd) >= BadTimes(Ibt,1) & Cat.Cat(:,ColInd) < BadTimes(Ibt,2);
    elseif (isnumeric(Cat)),
        if (isnumeric(TimeCol)),
            ColInd = TimeCol;
        else
            ColInd = 1;
        end
        BadFlag(:,Ibt) = Cat.Cat(:,ColInd) >= BadTimes(Ibt,1) & Cat.Cat(:,ColInd) < BadTimes(Ibt,2);
    else
        error('Illegal Cat input type');
    end
end
BadF  = any(BadFlag,2);
%GoodF = ~BadI;



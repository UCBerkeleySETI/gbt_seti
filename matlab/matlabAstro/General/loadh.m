function Data=loadh(FileName,VarName,varargin)
%--------------------------------------------------------------------------
% loadh function                                                   General
% Description: load a matrix from HDF5 file. 
%              If dataset name is not provided than will read all
%              datasets into a structure. This function doesn't support
%              groups.
%              This is becoming faster than matlab (2014a) for matices with
%              more than ~10^4 elements.
% Input  : - File name.
%          - variable name (dataset). If empty or not provided than will
%            attempt to read all datasets.
% Output : - Datasets.
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    May 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Data=loadh('R.hd5','R');
% Reliable: 2
%--------------------------------------------------------------------------

Def.VarName = [];
if (nargin<2),
    VarName = Def.VarName;
end

if (isempty(VarName)),
    % check for available datasets in HD5 file
    Info = h5info(FileName);
    if (~isempty(Info.Datasets)),
        Nds = numel(Info.Datasets);
        for Ids=1:1:Nds,
            Data.(Info.Datasets(Ids).Name) = loadh(FileName,Info.Datasets(Ids).Name);
        end   
    end
else
    DataSet = sprintf('/%s',VarName);
    Data    = h5read(FileName,DataSet);
end


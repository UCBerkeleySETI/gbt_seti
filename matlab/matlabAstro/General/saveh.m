function VarName=saveh(FileName,Data,VarName,varargin)
%--------------------------------------------------------------------------
% saveh function                                                   General
% Description: Save a matrix into HDF5 file. If file exist then will
%              add it as a new dataset to the same file.
%              This is becoming faster than matlab (2014a) for matices with
%              more than ~10^5 elements.
% Input  : - File name.
%          - Data to save (e.g., matrix).
%          - Dataset (variable) name under which to store the data
%            in the HDF5 file. If varaible already exist, then function
%            will fail. Default is 'V'.
%            If empty use 'V'.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            to pass to the h5create. See h5create.m for details.
% Output : - Variable name (dataset) in which the matrix was stored.
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    May 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: VarName=saveh('R.hd5',T,'G/A');
% Reliable: 2
%--------------------------------------------------------------------------

Def.VarName = 'V';
if (nargin<3),
    VarName = Def.VarName;
end
if (isempty(VarName)),
    VarName = Def.VarName;
end

DataSet = sprintf('/%s',VarName);
if (isnumeric(Data)),
    % save numeric data  
    h5create(FileName,sprintf('/%s',VarName),size(Data),varargin{:},'Datatype',class(Data));
    h5write(FileName,DataSet,Data);
else
    error('Non numeric datatype are not supheported yet');
end


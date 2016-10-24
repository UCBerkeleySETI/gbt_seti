function [Cat,varargout]=simcat2array(Sim,Columns)
%--------------------------------------------------------------------------
% simcat2array function                                            ImBasic
% Description: Given a single elemnt structure array or SIM with 'Cat'
%              and 'Col' fields, return a matrix containing the catalog
%              fields, and the indices of some specified columns.
% Input  : - Sim or a structure array with a single element.
%          - Either a cell array of column names, a cell array of column
%            indices, or a numeric array of column indices.
% Output : - The content of the 'Cat' field.
%          * Column indices per each requested column name/index.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Cat,Col.X,Col.Y,Col.Mag]=simcat2array(Sim(1),{'XWIN_IMAGE','YWIN_IMAGE','MAG_AUTO'});
%          [Cat,Col.X,Col.Y,Col.Mag]=simcat2array(Sim(1),{1 2 5});
%          [Cat,Col.X,Col.Y,Col.Mag]=simcat2array(Sim(1),[1 2 5]);
% Reliable: 2
%--------------------------------------------------------------------------


%ImageField     = 'Im';
%HeaderField    = 'Header';
%FileField      = 'ImageFileName';
%MaskField       = 'Mask';
%BackImField     = 'BackIm';
%ErrImField      = 'ErrIm';
CatField        = 'Cat';
CatColField     = 'Col';
%CatColCellField = 'ColCell';


Cat = Sim.(CatField);

Ncol = numel(Columns);
if (isnumeric(Columns)),
    Columns = num2cell(Columns);
    [varargout{1:1:Ncol}] = deal(Columns{:});
elseif (iscellstr(Columns)),
    % assume Columns is a cell array
    varargout = cell(Ncol,1);
    for Icol=1:1:Ncol,
        varargout{Icol} = Sim.(CatColField).(Columns{Icol});
    end
elseif (iscell(Columns)),
    [varargout{1:1:Ncol}] = deal(Columns{:});
else
    error('Illegal Columns input type');
end

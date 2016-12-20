function [Out,Col,ColCell,ColUnits,ColTypeChar,ColRepeat,ColScale,ColZero,ColNulval,ColTdisp]=read_fitstable(TableName,varargin)
%--------------------------------------------------------------------------
% read_fitstable function                                          ImBasic
% Description: Read binary or ascii FITS tables.
% Input  : - List of FITS tables to read. Any input valid to create_list.m.
%            If multiple files are provided then all the files hould be of
%            the same type (e.g., fits binary table).
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'TableType'- FITS table type {'auto'|'bintable'|'table'}.
%                         Default is 'auto'.
%                         'auto' will attempt to read the table type
%                         from the 'XTENSION' header keyword.
%            'HDUnum'   - HDU number in which the FITS table header is.
%                         Default is 2.
%            'ModColName' - If the program failed because some columns
%                         have name starting with numbers or invalid signs
%                         set this to true (default is false).
%                         This will modify the column names.
%            'OutTable' - Type of table output:
%                         'struct' - structure array in which the catalog
%                                   is a matrix. Default.
%                         'struct_t' - structure array in which the catalog
%                                   is a table.
%                         'cell1v' - For a single table output only.
%                                   A cell array of columns.
%                         'cellv' - Cell array of cell array of columns.
%                         'mat1'  - For a single table output only.
%                                   A matrix.
%                         'cellm' - A cell array of matrices.
%                         'table1' - For a single table output only.
%                                   A table.
%                         'cell_t' - A cell array of tables.   
%            'XTENkey'  - Header keyword from which to read the table type.
%                         Default is 'XTENSION'.
%            'StartRow' - First row to read from FITS table. Default is [].
%                         If empty, then read the entire table.
%            'NRows'    - Number of rows to read from FITS table.
%                         Default is []. If empty, then read the entire
%                         table.
%            'OutClass' - A function to use in order to force all the
%                         columns to be of the same class (e.g., @single).
%                         Default is @double. If empty, then will keep
%                         the original class. This option shoyld be used
%                         if you want to read the data into a matrix.
%            'NullVal'  - If the column is of numeric type will attempt
%                         to replace the FITS null value with this value.
%                         Default is NaN.
%            'BreakRepCol'- {true|false}. If true and FITS table columns
%                         are repeating then will change column information
%                         according to the matrix column count.
%                         Default is true.
%            'CatField' - Structure field in which to store the table.
%                         Default is 'Cat'.
%            'ColField' - Structure field in which to store the columns.
%                         Default is 'Cat'.
%            'ColCellField'- Structure field in which to store the cell
%                         of columns. Default is 'Cat'.
%            'ColUnitsField'- Structure field in which to store the cell
%                         of column units. Default is 'Cat'.
% Output : - FITS table output.
%          - Structure array of column name and indices.
%          - Cell array of column names.
%          - Cell array of column units.
%          - Cell array of columns type.
%          - Cell array of column repeatition.
%          - Cell array of column scale.
%          - Cell array of column zero.
%          - Cell array of column null value.
%          - Cell array of column Tdisp.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Out,Col,ColCell,ColUnits,ColTypeChar,ColRepeat,ColScale,ColZero,ColNulval,ColTdisp]=read_fitstable(TableName)
%          [Out,Col,ColCell]=read_fitstable(TableName);
%          Cat=read_fitstable('VII_241.fits','TableType','table','ModColName',true);
% Reliable: 2
%--------------------------------------------------------------------------

CatField        = 'Cat';
ColField        = 'Col';
ColCellField    = 'ColCell';
ColUnitsField   = 'ColUnits';
        

DefV.TableType      = 'auto';    % {'auto'|'bintable'|'table'}
DefV.HDUnum         = 2;
DefV.ModColName     = false;
DefV.OutTable       = 'struct';  % {'struct'|'struct_t'|...}
DefV.XTENkey        = 'XTENSION';
DefV.StartRow       = [];
DefV.NRows          = [];
DefV.OutClass       = @double;
DefV.NullVal        = NaN;       % [] do nothing
DefV.BreakRepCol    = true;   
DefV.CatField       = CatField;
DefV.ColField       = ColField;
DefV.ColCellField   = ColCellField;
DefV.ColUnitsField  = ColUnitsField;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% prep list of fits table names
[~,ListTableName] = create_list(TableName,NaN);
Nfile = numel(ListTableName);

switch lower(InPar.TableType)
    case 'auto'
        HeadCell = fits_get_head(ListTableName{1},InPar.HDUnum);
        Ixten    = find(strcmp(HeadCell(:,1),InPar.XTENkey));
        if (isempty(Ixten)),
            error('Automatic table identification mode was unable to access FITS table type');
        end
        InPar.TableType = HeadCell{Ixten,2};
    otherwise
        % do nothing
end

switch lower(InPar.TableType)
    case 'bintable'
        Fun_getColParms = @fits.getBColParms;
        
    case 'table'
        Fun_getColParms = @fits.getAColParms;
        
    otherwise
        error('Unknown TableType option');
end

if (isempty(InPar.StartRow) || isempty(InPar.NRows)),
    CellRowPar = {};
else
    CellRowPar = {InPar.StartRow,InPar.NRows};
end



switch lower(InPar.OutTable)
    case {'struct','struct_t'}
        Out = struct_def({InPar.CatField,InPar.ColField,InPar.ColCellField},Nfile,1);
    case {'cellv','cellm','cell_t'}
        Out = cell(Nfile,1);
    otherwise
        % do nothing
end


import matlab.io.*
for Ifile=1:1:Nfile,
    % for each fits table
    Fptr = fits.openFile(ListTableName{Ifile});
    fits.movAbsHDU(Fptr,InPar.HDUnum);
    Ncol = fits.getNumCols(Fptr);
    %[ttype,tunit,typechar,repeat,scale,zero,nulval,tdisp]= fits.getBColParms(Fptr,2);
    ColCell     = cell(1,Ncol);
    ColUnits    = cell(1,Ncol);
    ColTypeChar = cell(1,Ncol);
    ColRepeat   = cell(1,Ncol);
    ColScale    = cell(1,Ncol);
    ColZero     = cell(1,Ncol);
    ColNulval   = cell(1,Ncol);
    ColTdisp    = cell(1,Ncol);
    ColData     = cell(1,Ncol);
    for Icol=1:1:Ncol,
        [ColCell{Icol},ColUnits{Icol},ColTypeChar{Icol},ColRepeat{Icol},ColScale{Icol},ColZero{Icol},ColNulval{Icol},ColTdisp{Icol}]= Fun_getColParms(Fptr,Icol);
        [ColData{Icol}] = fits.readCol(Fptr,Icol,CellRowPar{:});
        if (~isempty(InPar.OutClass)),
            ColData{Icol} = InPar.OutClass(ColData{Icol});
        end
        if (~isempty(InPar.NullVal) && ~isempty(ColNulval{Icol}) && isnumeric(ColData{Icol})),
            ColData{Icol}(ColData{Icol}==ColNulval{Icol}) = InPar.NullVal;
            
        end
        % override ColRepeat using the actual data
        ColRepeat{Icol} = size(ColData{Icol},2);
    end
    fits.closeFile(Fptr);
    
    % deal with repeating columns
    if (InPar.BreakRepCol),
        Nnc  = sum(cell2mat(ColRepeat));
        NewColCell = cell(Nnc,1);
        
        Icol1 = 1;
        for Icol=1:1:Ncol,            
            IcolN = Icol1 + ColRepeat{Icol} - 1;
            %Icol1 = Icol1 + ColRepeat{Icol}; % at the end of the loop
            for Irep=1:1:ColRepeat{Icol},
                if (ColRepeat{Icol}>1),
                    NewColCell{Icol1+Irep-1} = sprintf('%s_%d_',ColCell{Icol},Irep);
                else
                    NewColCell{Icol1+Irep-1} = ColCell{Icol};
                end
            end
            [NewColUnits{Icol1:IcolN}] = deal(ColUnits{Icol});
            [NewColTypcChar{Icol1:IcolN}] = deal(ColTypeChar{Icol});
            [NewColRepeat{Icol1:IcolN}] = deal(1);
            [NewColScale{Icol1:IcolN}] = deal(ColScale{Icol});
            [NewColZero{Icol1:IcolN}] = deal(ColZero{Icol});
            [NewColTdisp{Icol1:IcolN}] = deal(ColTdisp{Icol});
            Icol1 = Icol1 + ColRepeat{Icol}; % at the end of the loop

        end
        ColCell     = NewColCell;
        ColUnits    = NewColUnits;
        ColTypeChar = NewColTypcChar;
        ColRepeat   = NewColRepeat;
        ColScale    = NewColScale;
        ColZero     = NewColZero;
        ColTdisp    = NewColTdisp;   
    end
    if (InPar.ModColName),
        % modify column names
        ColCell = regexprep(ColCell,{'-','/','(',')','&','@','#','^','%','*','~'},'');
        ColCell = strcat('T',ColCell);
    end
    Col             = cell2struct(num2cell(1:1:length(ColCell)),ColCell,2);
    
    % output
    switch lower(InPar.OutTable)
        case 'struct'
            Out(Ifile).(InPar.CatField)      = [ColData{:}];
            Out(Ifile).(InPar.ColField)      = Col;
            Out(Ifile).(InPar.ColCellField)  = ColCell;
            %Out(Ifile).(InPar.ColUnitsField) = ColUnits;
        case 'struct_t'
            Out(Ifile).(InPar.CatField)      = table(ColData{:});
            Out(Ifile).(InPar.ColField)      = Col;
            Out(Ifile).(InPar.ColCellField)  = ColCell;
            Out(Ifile).(InPar.CatField).Properties.VariableNames = ColCell;
            Out(Ifile).(InPar.CatField).Properties.VariableUnits = ColUnits;
        case 'cell1v'
            % assume there is only one table
            Out = ColData;
        case 'cellv'
            Out{Ifile} = ColData;
        case 'mat1'
            Out = [ColData{:}];
        case 'cellm'
            Out{Ifile} = [ColData{:}];
        case 'table1'
            Out = table(ColData{:});
            Out.Properties.VariableNames = ColCell;
            Out.Properties.VariableUnits = ColUnits;
        case 'cell_t'
            Out{Ifile} = table(ColData{:});
            Out{Ifile}.Properties.VariableNames = ColCell;
            Out{Ifile}.Properties.VariableUnits = ColUnits;
            
        otherwise
            error('Unknown OuTable option');
    end
    
            
end


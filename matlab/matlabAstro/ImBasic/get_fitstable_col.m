function [Fields,InfoKey,FieldsMat,Col,Table]=get_fitstable_col(FitsName,FitsType,Index)
%--------------------------------------------------------------------------
% get_fitstable_col function                                       ImBasic
% Description: Read a FITS table, and get the fits table column names from
%              the FITS header.
% Input  : - String containing FITS table name.
%          - Fits type: {'Table' | 'BinTable'}, default is 'BinTable'.
%          - Read keywords from BinaryTable with specific index.
%            (NaN will read Primary header). Default is 1.
% Output : - Cell array containing column names in FITS table.
%          - The FITS table specific header.
%          - Cell array containing column names in FITS table, assuming
%            that the FITS table content was converted to matrix using
%            the cell2mat.m command.
%            This may be different than the first output if some of the
%            columns contains multiple columns.
%          - Structure in which its field is the column name and its
%            value contains the column index.
%            Assuming that the FITS table content was converted to matrix
%            using the cell2mat.m command.
%          - The FITS table converted to matrix using cell2mat.
% Tested : Matlab 7.8
%     By : Eran O. Ofek                    Apr 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [~,~,~,Col,Table]=get_fitstable_col('a.fits');
% Reliable: 1
%--------------------------------------------------------------------------

Def.FitsType = 'BinTable';
Def.Index    = 1;

if (nargin==1),
   FitsType   = Def.FitsType;
   Index      = Def.Index;
elseif (nargin==2),
   Index      = Def.Index;
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end

if (ischar(FitsName)==1)
   % read FITS image
       Info       = fitsinfo(FitsName);
 else
   % keyword structure
   Info       = FitsName;
end

if (isstruct(Info)==1),
   if (isnan(Index)==1),
      InfoKey = Info.PrimaryData.Keywords;
   else
      InfoKey = Info.BinaryTable(Index).Keywords;
   end
else
   % do nothing - Assumes Info is a cell array
    InfoKey = Info;
end

% check number of columns:
I_TFIELDS=find(strcmpi({InfoKey{:,1}},'TFIELDS')==1);
TFIELDS  = InfoKey{I_TFIELDS,2};
% read all column names:
Fields = cell(1,TFIELDS);
Iind   = 0;
for Ic=1:1:TFIELDS,
   I_F = find(strcmpi({InfoKey{:,1}},sprintf('TTYPE%d',Ic))==1);
   Fields{Ic} = InfoKey{I_F,2};

   I_Format = find(strcmpi({InfoKey{:,1}},sprintf('TFORM%d',Ic))==1);
   try
      Ncol(Ic) = str2num(InfoKey{I_Format,2}(1));
   catch
      Ncol(Ic) = 1;
   end

   if (Ncol(Ic)==1),
      Iind = Iind + 1;
      FieldsMat{Iind} = InfoKey{I_F,2};
   else
      for Icol=1:1:Ncol(Ic),
         Iind = Iind + 1;
         FieldsMat{Iind} =   sprintf('%s_%d_',InfoKey{I_F,2},Icol);
      end
   end
end


Ncol = length(FieldsMat);
[Col] = cell2struct(num2cell(1:1:Ncol),FieldsMat,2);


if (nargout>4),
    Table = fitsread(FitsName,FitsType,Index);
end



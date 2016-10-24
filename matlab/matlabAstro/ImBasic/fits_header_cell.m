function [HeadCell,Size,DataType]=fits_header_cell(Image,FieldName,Ind)
%--------------------------------------------------------------------------
% fits_header_cell function                                        ImBasic
% Description: Read FITS image header to cell header.
%              This function can be used instead of fitsinfo.m
%              This function is obsolte: use fits_get_head.m instead.
% Input  : - Image name;
%          - Field name containing the header in the structure returned by
%            fitsinfo.m. Default is 'PrimaryData'. If empty then use
%            default. If NaN then will attempt to look for the correc
%            field.
%          - Index of image. Default is 1.
% Output : - Cell array of image header.
%          - Image size.
%          - Image data type.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [HeadCell,Size,DataType]=fits_header_cell('lred0121.fits');
% Reliable: 2
%--------------------------------------------------------------------------



Def.FieldName = 'PrimaryData';
Def.Ind       = 1;
if (nargin==1),
    FieldName = Def.FieldName;
    Ind       = Def.Ind;
elseif (nargin==2),
    Ind       = Def.Ind;
elseif (nargin==3),
    % do nothing
else
    error('Illegal number of input arguments');
end

Header = fitsinfo(Image);

if (isempty(FieldName)),
    FieldName = Def.FieldName;
end

if (isnan(FieldName)),
    % look for header info in header structure
    FN=fieldnames(Header);
    for I=1:1:length(FN),
        if (isstruct(Header.(FN{I}))),
            FieldName = FN{I};
        end
    end
end

HeadCell = Header.(FieldName)(Ind).Keywords;
Size     = Header.(FieldName)(Ind).Size;
DataType = Header.(FieldName)(Ind).DataType;







function CellS=sprintf2cell(String,Mat)
%--------------------------------------------------------------------------
% sprintf2cell function                                            General
% Description: Generate a cell array of strings using the sprintf function,
%              where the sprintf arguments, per string in cell is
%              taken from a row in a matrix.
% Input  : - Template string. E.g., 'Name%05d_c%02d.fits'.
%          - Matrix of arguments, where the number of rows is the
%            number of requested cells, and the number of columns
%            is the number of arguments.
% Output : - Cell array of strings.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Feb 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: CellS=sprintf2cell('APASS_htm%05d.mat',(1:1:10)');
%          CellS=sprintf2cell('APASS_htm%05d_%02d.mat',[1 2;3 4]);
%          CellS=sprintf2cell('%02d:%02d:%06.3f',convertdms(rand(10,1),'r','H'));
% Reliable: 2
%--------------------------------------------------------------------------

N = size(Mat,1);
CellS = cell(N,1);
for I=1:1:N,
    CellS{I} = sprintf(String,Mat(I,:));
end


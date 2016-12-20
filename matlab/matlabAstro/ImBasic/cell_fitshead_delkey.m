function [NewCellHead]=cell_fitshead_delkey(CellHead,Keys)
%--------------------------------------------------------------------------
% cell_fitshead_delkey function                                    ImBasic
% Description: A utility program to delete a keywords, values and
%              comments from a cell array containing A FITS header
%              information.
%              The FITS header cell array contains an arbitrary number
%              of rows and 3 columns, where the columns are:
%              {keyword_name, keyword_val, comment}.
%              The comment column is optional.
% Input  : - Cell array containing the FITS header information.
%          - A string containing a keyword name to delete,
%            or a cell of strings containing keyword names to delete.
% Output : - The new header cell array.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%--------------------------------------------------------------------------

if (ischar(Keys)),
   Keys = {Keys};
elseif (iscell(Keys)),
   % do nothing
else
   error('Unknown Keys DataType');
end

NewCellHead = CellHead;
Nkeys = length(Keys);
for Ikeys=1:1:Nkeys,
   [~,Lines]=cell_fitshead_getkey(NewCellHead,Keys{Ikeys});
   NewCellHead = delete_ind(NewCellHead,Lines);
   Nl = length(Lines);
   for I=1:1:Nl-1,
      [~,Lines]=cell_fitshead_getkey(NewCellHead,Keys{Ikeys});
      NewCellHead = delete_ind(NewCellHead,Lines);
   end
end

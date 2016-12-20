function HeadStr=fitshead(File,varargin)
%--------------------------------------------------------------------------
% fitshead function                                                ImBasic
% Description: Read FITS file header, convert it to a string, and
%              print it to screen. Supress empty header lines.
% Input  : - A fits image file name, or a structure array containing
%            an .Header field.
%          * Additional arguments that will be passed to fitsinfo.m.
% Output : - A string containing the header, with carridge return
%            between lines.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Sep 2013
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: fitshead PTF201001131475_2_o_14235_00.w.fits
%          HeadStr=fitshead('PTF201001131475_2_o_14235_00.w.fits');
% Reliable: 2
%--------------------------------------------------------------------------

HeaderField = 'Header';

if (isstruct(File)),
   HeaderCell = File.(HeaderField);
else
   H = fitsinfo(File);
   HeaderCell = H.PrimaryData.Keywords;
end
N = size(HeaderCell,1);
HeadStr = '';
for I=1:1:N,
   if (isempty(HeaderCell{I,1}) && isempty(HeaderCell{I,2}) && isempty(HeaderCell{I,3})),
      % empty line - supress
   else
      if (ischar(HeaderCell{I,2})),
          HeadStr = sprintf('%s%8s%20s%s\n',HeadStr,HeaderCell{I,1},HeaderCell{I,2},HeaderCell{I,3});
      else
          HeadStr = sprintf('%s%8s%20.10f%s\n',HeadStr,HeaderCell{I,1},HeaderCell{I,2},HeaderCell{I,3});
      end
   end
end


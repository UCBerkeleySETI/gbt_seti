function split_multiext_fits(ImList,Type)
%-----------------------------------------------------------------------------
% split_multiext_fits function                                        ImBasic
% Description: Given a list of multi extension FITS files break each file
%              to its multiple extensions. Each extension will be given
%              the header of the primary FITS file with the extension header.
% Input  : - List of FITS images (see create_list.m for details).
%          - Method by which to set the name of the new file:
%            'e' - add '.<extension_number>' to the end of file (default).
% Output : null
% Tested : Matlab 7.10
%     By : Eran O. Ofek                    Oct 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%-----------------------------------------------------------------------------

Def.Type   = 'e';
if (nargin==1),
   Type    = Def.Type;
elseif (nargin==2),
   % do nothing
else
   error('Illegal number of input arguments');
end

[~,ImListCell] = create_list(ImList,NaN);
Nim = length(ImListCell);

for Iim=1:1:Nim;
   Head = fitsinfo(ImListCell{Iim});
   Next = length(Head.Image);   % number of extensions
   % for each extension
   for Iext=1:1:Next,
      % set the header for the current extension
      % combine the primary header with the extension header
      HeadExt = [Head.PrimaryData.Keywords;Head.Image(Iext).Keywords];

      switch lower(Type)
       case 'e'
          FileName = sprintf('%s.%d',ImListCell{Iim},Iext);
       otherwise
          error('Unknown Type option');
      end
      Image  = fitsread(ImListCell{Iim},'Image',Iext);
      fitswrite(Image,FileName,HeadExt);
   end
end

function [Size,UniqueSize,SubList,IlistUn]=find_size_fits(ImList)
%------------------------------------------------------------------------
% find_size_fits function                                        ImBasic
% Description: Given a list of 2D FITS images return the image sizes as
%              specified in the image headers (NAXIS1, NAXIS2 keywords).
%              Optionally, also look for all the unique sizes of images
%              among the list and return sublists of all the images with
%              a given size.
% Input  : - List of FITS images (see create_list.m for details).
%            Default is '*.fits'.
% Output : - Structure containing the the size information for each
%            FITS file. Structure fields are:
%            .Name   - Cell array of FITS file names.
%            .NAXIS1 - Vector of sizes of the first dimension of each image.
%            .NAXIS2 - Vector of sizes of the second dimension of each image.
%          - Matrix containing the unique sizes found among the images.
%            This is a two column matrix in which the first column
%            containing NAXIS1 and the second column is NAXIS2.
%          - Cell array in which each cell containing a cell array
%            of images which have a unique size.
%            The i-th cell corresponds to the i-th line in the UniqueSize
%            matrix (previous output argument).
%          - Cell array in which each cell contain a vector of indices
%            of images in the input list which have a unique size.
%            The i-th cell corresponds to the i-th line in the UniqueSize
%            matrix (previous output argument).
% Tested : Matlab 7.10
%     By : Eran O. Ofek                    Jun 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Size,UniqueSize,SubList,IlistUn]=find_size_fits('lred*.fits');
% Reliable: 2
%------------------------------------------------------------------------

Def.ImList = '*.fits';
if (nargin==0),
   ImList  = Def.ImList;
elseif (nargin==1),
   % do nothing
else
   error('Illegal number of input arguments');
end

[~,ImListCell] = create_list(ImList,NaN);
Nim = length(ImListCell);

Size.Naxis1 = zeros(Nim,1);
Size.Naxis2 = zeros(Nim,1);
for Iim=1:1:Nim,
   KeywordVal = get_fits_keyword(ImListCell{Iim},{'NAXIS1','NAXIS2'});
   Size.Name{Iim}   = ImListCell{Iim};
   Size.NAXIS1(Iim) = KeywordVal{1};
   Size.NAXIS2(Iim) = KeywordVal{2};
end

if (nargout>1),
   % Unique sizes
   UniqueSize = unique([Size.NAXIS1.', Size.NAXIS2.'],'rows');
   Nunique    = size(UniqueSize,1);
   for Iunique=1:1:Nunique,
      IlistUn{Iunique} = find(Size.NAXIS1==UniqueSize(Iunique,1) & Size.NAXIS2==UniqueSize(Iunique,2));
      SubList{Iunique} = ImListCell(IlistUn{Iunique});
   end

end

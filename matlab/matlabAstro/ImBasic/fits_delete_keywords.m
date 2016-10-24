function fits_delete_keywords(ImageName,Keywords)
%--------------------------------------------------------------------------
% fits_delete_keywords function                                    ImBasic
% Description: Delete a list of header keywords from a list of
%              FITS images.
% Input  : - List of FITS image names to read. See create_list.m for
%            options.
%          - Cell array of keyword names to delete.
% Output : null
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jun 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: fits_delete_keywords('A.fits',{'PTFPID','OBJECT'})
% Reliable: 2
%--------------------------------------------------------------------------


if (~iscell(Keywords)),
    Keywords = {Keywords};
end
Nkey = numel(Keywords);

[~,List] = create_list(ImageName,NaN);
Nim = numel(List);

import matlab.io.*
for Iim=1:1:Nim,
    Fptr = fits.openFile(List{Iim},'readwrite');
    for Ikey=1:1:Nkey,
        fits.deleteKey(Fptr,Keywords{Ikey});
    end
    fits.closeFile(Fptr);
end

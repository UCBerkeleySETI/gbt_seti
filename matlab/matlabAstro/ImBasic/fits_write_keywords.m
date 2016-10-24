function fits_write_keywords(ImageName,KeyCell)
%--------------------------------------------------------------------------
% fits_write_keywords function                                     ImBasic
% Description: Insert new, or update existing FITS header keywords in
%              a list of FITS images.
% Input  : - List of FITS image names to edit. See create_list.m for
%            options.
%          - A cell array of two or three columns of key/cal/comments to
%            add to FITS header.
% Output : null
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jun 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: fits_write_keywords('A.fits',{'try','A','comm';'try2',6,'what'});
% Reliable: 2
%--------------------------------------------------------------------------


Nkey = size(KeyCell,1);

[~,List] = create_list(ImageName,NaN);
Nim = numel(List);

import matlab.io.*
for Iim=1:1:Nim,
    Fptr = fits.openFile(List{Iim},'readwrite');
    for Ikey=1:1:Nkey,
        fits.writeKey(Fptr,KeyCell{Ikey,:});
    end    
    fits.closeFile(Fptr);
end
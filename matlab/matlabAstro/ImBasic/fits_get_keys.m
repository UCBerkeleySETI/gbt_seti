function [KeysVal,KeysComment,Struct]=fits_get_keys(Image,Keys,HDUnum,Str)
%--------------------------------------------------------------------------
% fits_get_keys function                                           ImBasic
% Description: Get the values of specific keywords from a single
%              FITS file header. Use only for existing keywords.
% Input  : - FITS image name.
%          - Cell array of keys to retrieve from the image header.
%          - HDU number. Default is 1.
%            If NaN, then set to 1.
%          - Check if the keyword value is char and try to convert to
%            a number {false|true}. Default is false.
% Output : - Cell array of keyword values.
%          - Cell array of keyword comments.
%          - Structure containing the keyword names (as fields)
%            and their values.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jul 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [KeysVal,KeysComment,Struct]=fits_get_keys('A.fits',{'NAXIS1','NAXIS2'});
% Reliable: 2 
%--------------------------------------------------------------------------

if (nargin==2),
    HDUnum = 1;
    Str    = false;
elseif (nargin==3),
    Str    = false;
else
    % do nothing
end

if (isnan(HDUnum)),
    HDUnum = 1;
end

import matlab.io.*
Fptr = fits.openFile(Image);
N = fits.getNumHDUs(Fptr);
if (HDUnum>N),
    fits.closeFile(Fptr);
    error('requested HDUnum does not exist');
end
fits.movAbsHDU(Fptr,HDUnum);

if (ischar(Keys)),
    Keys = {Keys};
end

Nkey = numel(Keys);


KeysVal     = cell(size(Keys));
KeysComment = cell(size(Keys));
for Ikey=1:1:Nkey,
    [KeysVal{Ikey},KeysComment{Ikey}] = fits.readKey(Fptr,Keys{Ikey});
    if (ischar(KeysVal{Ikey}) && Str),
        Tmp = str2double(KeysVal{Ikey});
        if (isnan(Tmp)),
            % do nothing - keep as a string
        else
            KeysVal{Ikey} = Tmp;
        end
    end
            
end
fits.closeFile(Fptr);

if (nargout>2),
   Struct = cell2struct(KeysVal,Keys,2);
end
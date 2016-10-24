function [KeysVal,KeysComment,Struct,List]=fits_mget_keys(Images,Keys,HDUnum,Str)
%--------------------------------------------------------------------------
% fits_mget_keys function                                          ImBasic
% Description: Get the values of specific keywords from a list of
%              FITS files header.
% Input  : - List of FITS image names. See create_list.m for options.
%          - Cell array of keys to retrieve from the image header.
%          - HDU number. Default is 1.
%          - Check if the keyword value is char and try to convert to
%            a number {false|true}. Default is false.
% Output : - Cell array (per image) of cell array of keyword values.
%          - Cell array (per image) of cell array of keyword comments.
%          - Structure array (element per image) containing the keyword
%            names (as fields) and their values.
%          - Cell array containing the list of images.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jul 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example:
% [KeysVal,KeysComment,Struct,List]=fits_mget_keys('PTF_201202*.fits',{'NAXIS1','NAXIS2'});
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

[~,List] = create_list(Images,NaN);
Nim = numel(List);
KeysVal     = cell(size(List));
KeysComment = cell(size(List));
for Iim=1:1:Nim,
   [KeysVal{Iim},KeysComment{Iim},Struct(Iim)]=fits_get_keys(List{Iim},Keys,HDUnum,Str);
end

function fitswrite_my1(Image,FileName,HeaderInfo,DataType,varargin)
%--------------------------------------------------------------------------
% fitswrite_my function                                            ImBasic
% Description: Write a simple 2D FITS image.
% Input  : - A 2D matrix to write as FITS file.
%          - String containing the image file name to write.
%          - Cell array containing header information to write to image.
%            The keywords: SIMPLE, BITPIX, NAXIS, NAXIS1, NAXIS2, EXTEND
%            will be re-written to header by default.
%            The keywords BSCALE and BZERO wil be written to header if
%            not specified in the header information cell array.
%            Alternatively this could be a character array (Nx80)
%            containing the header (no changes will be applied).
%            If not given, or if empty matrix (i.e., []) than write a
%            minimal header.
%          - DataType in which to write the image, supported options are:
%            'int8',8            
%            'int16',16
%            'int32',32
%            'int64',64
%            'single','float32',-32    (default)
%            'double','float64',-64            
%          * Arbitrary number of pairs of input arguments: 
%            ...,keyword,value,... - possible keys are:
%            'OverWrite'    - Over write existing image {true|false},
%                             default is true.
% Output : null
% See also: fitswrite_me.m, fitswrite_nd.m, sim2fits.m, sims2fits.m
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Examples: [Flag,HeaderInfo]=fitswrite_my(rand(2048,1024),'Example.fits');
% Reliable: 2
%--------------------------------------------------------------------------


Def.HeaderInfo = [];
Def.DataType   = -32;
if (nargin==2),
   HeaderInfo = Def.HeaderInfo;
   DataType   = Def.DataType;
elseif (nargin==3),
   DataType   = Def.DataType;
end

% set default for additional keywords:
%DefV.IdentifyInt = 'y';
%DefV.ResetDefKey = 'y';
DefV.OverWrite   = 'true';

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});


switch DataType
 case {'uint8','int8',8}
    FunDataType = @uint8;
    BitPix      = 'byte_img';
 case {'int16',16}
    FunDataType = @int16;
    BitPix      = 'short_img';
 case {'int32',32}
    FunDataType = @int32;
    BitPix      = 'long_img';
 case {'int64',64}
    FunDataType = @int64;
    BitPix      = 'longlong_img';
 case {'single','float32',-32}
    FunDataType = @single;
    BitPix      = 'float_img';
 case {'double','float64',-64}
    FunDataType = @double;
    BitPix      = 'double_img';
 otherwise
    error('Unknown DataType option');
end

% convert image data type
Image = FunDataType(Image);

Nhead = size(HeaderInfo,1);

if (InPar.OverWrite),
    delete(FileName);
end

import matlab.io.*
Fptr = fits.createFile(FileName);
fits.createImg(Fptr,BitPix,size(Image));
% write header
for Ihead=1:1:Nhead,
    switch lower(HeaderInfo{Ihead,1})
        case {'simple','bitpix','naxis','naxis1','naxis2','EXTEND'}
            % do nothing - these keywords are automatically generated
        case 'history'
            fits.writeHistory(Fptr,HeaderInfo{Ihead,2});
        case 'comment'
            fits.writeComment(Fptr,HeaderInfo{Ihead,2});
        otherwise
            fits.writeKey(Fptr,HeaderInfo{Ihead,:});
    end
end
%data = reshape(1:256*512,size(Image));
%data = int32(data);
fits.writeImg(Fptr,Image);
fits.closeFile(Fptr);


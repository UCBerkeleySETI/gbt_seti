function [Flag,HeaderInfo]=fitswrite_nd(Image,FileName,HeaderInfo,DataType,varargin)
%--------------------------------------------------------------------------
% fitswrite_nd function                                            ImBasic
% Description: Write a multi-dimensional FITS image to a file.
% Input  : - An image, a cube or a higfher diemsion image to save as
%            a FITS file.
%          - String containing the image file name to write.
%          - Cell array containing header information to write to image.
%            The keywords: SIMPLE, BITPIX, NAXIS, NAXIS1, NAXIS2, EXTEND
%            will be re-written to header by default.
%            The keywords BSCALE and BZERO wil be written to header if
%            not specified in the header information cell array.
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
%            'OverWrite'    - Over write existing image {tru|false},
%                             default is true.
% Output : - Flag indicating if image was written to disk (1) or not (0).
%          - Actual header information written to file.
% Tested : MATLAB R2013b
%     By : Eran O. Ofek                    Mar 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Flag,HeaderInfo]=fitswrite_nd(rand(2048,1024,2),'Example.fits');
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
DefV.OverWrite   = true;

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});



switch DataType
 case {'int8',8}
    DataType = 'uint8';
 case {'int16',16}
    DataType = 'uint16';
 case {'int32',32}
    DataType = 'uint32';
 case {'int64',64}
    DataType = 'uint64';
 case {'single','float32',-32}
    DataType = 'single';
 case {'double','float64',-64}
    DataType = 'double';
 otherwise
    error('Unknown DataType option');
end



   
%--- Set the FITS "mandatory" keywords ---
%--- add BSCALE and BZERO ---
% check if BZERO and BSCALE are already in HeaderInfo 
if (~isempty(HeaderInfo)),
    CheckH = cell_fitshead_getkey(HeaderInfo,{'BZERO','BSCALE'},'NaN');
    if (any(isnan([CheckH{:,2}]))),
       HeaderInfo = cell_fitshead_addkey(HeaderInfo,...
                             'BZERO',single(0),'offset data range to that of unsigned short',...
                             'BSCALE',single(1),'default scaling factor');
    end
end

%--- Write creation date to header ---
Time = get_atime([],0,0);
HeaderInfo = cell_fitshead_addkey(HeaderInfo,...
                                  'CRDATE',Time.ISO,'Creation date of FITS file',...
                                  'COMMENT','','File Created by MATLAB fitswrite.m written by Eran Ofek');


    
[Nline] = size(HeaderInfo);

if (InPar.OverWrite)
    % delete existing FileName if exist
    if (exist(FileName,'file')~=0),
        delete(FileName);
    end
end


%--- write the image ---

%Fpixels = [1 1];
import matlab.io.*
%Fptr = fits.openFile(FileName,'READWRITE');
Fptr = fits.createFile(FileName);
fits.createImg(Fptr,DataType,size(Image));
fits.writeImg(Fptr,Image); %,Fpixels);
for Inl=1:1:Nline,
    switch lower(HeaderInfo{Inl,1})
        case 'comment'
            fits.writeComment(Fptr,HeaderInfo{Inl,3});
        case 'history'
            fits.writeHistory(Fptr,HeaderInfo{Inl,3});
        otherwise
            fits.writeKey(Fptr,HeaderInfo{Inl,1},HeaderInfo{Inl,2},HeaderInfo{Inl,3});
    end
  
end

fits.closeFile(Fptr);
Flag = sign(Fptr);


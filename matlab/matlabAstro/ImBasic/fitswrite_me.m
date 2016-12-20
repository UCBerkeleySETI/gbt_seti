function [Flag,HeaderInfo]=fitswrite_me(Images,FileName,HeaderInfo,DataType,varargin)
%--------------------------------------------------------------------------
% fitswrite_me function                                            ImBasic
% Description: Write a set of images to a single multi-extension FITS file.
% Input  : - A cell array of images to write. The number of FITS extensions
%            will be as the number of cells. Each cell may contain
%            an image of any dimension.
%          - String containing the image file name to write.
%          - Cell array containing header information to write to the
%            first header data unit (HDU) image.
%            Alternatively, this can be a cell array of headers cell
%            array to write to each one of the extensions HDU.
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
% Tested : Matlab R2013b
%     By : Eran O. Ofek                    Mar 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Flag,HeaderInfo]=fitswrite_me(rand(2048,1024,2),'Example.fits');
%          fitswrite_me({rand(2048,1024,2),rand(10,10),rand(20,20)},'Example.fits');
% Reliable: 2
%--------------------------------------------------------------------------


Def.HeaderInfo = {cell(0,3)};
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


if (~iscell(Images)),
    Images = {Images};
end
Nim = numel(Images);

if (isempty(HeaderInfo)),
    HeaderInfo = {HeaderInfo};
else
    if (~isempty(HeaderInfo{1})),
        HeaderInfo = {HeaderInfo};
    end
end

Nhdu = numel(HeaderInfo);
for Ihdu=Nhdu+1:Nim,
   HeaderInfo{Ihdu} = cell(0,3);
end

if (InPar.OverWrite)
    % delete existing FileName if exist
    if (exist(FileName,'file')~=0),
        delete(FileName);
    end
end

% create the FITS file
import matlab.io.*
Fptr = fits.createFile(FileName);


% for each image
for Iim=1:1:Nim,
    %--- Set the FITS "mandatory" keywords ---
    %--- add BSCALE and BZERO ---
    % check if BZERO and BSCALE are already in HeaderInfo 
    if (~isempty(HeaderInfo{Iim})),
        CheckH = cell_fitshead_getkey(HeaderInfo{Iim},{'BZERO','BSCALE'},'NaN');
        if (any(isnan(CheckH{1,2}))),
           HeaderInfo{Iim} = cell_fitshead_addkey(HeaderInfo{Iim},...
                                 'BZERO',single(0),'offset data range to that of unsigned short',...
                                 'BSCALE',single(1),'default scaling factor');
        end
    end
    %--- Write creation date to header ---
    Time = get_atime([],0,0);
    HeaderInfo{Iim} = cell_fitshead_addkey(HeaderInfo{Iim},...
                                      'CRDATE',Time.ISO,'Creation date of FITS file',...
                                      'COMMENT','','File Created by MATLAB fitswrite.m written by Eran Ofek');


    [Nline] = size(HeaderInfo{Iim});

    %--- write the image ---
    %Fpixels = [1 1];
    %Fptr = fits.openFile(FileName,'READWRITE');
    fits.createImg(Fptr,DataType,size(Images{Iim}));
    fits.writeImg(Fptr,Images{Iim}); %,Fpixels);
    for Inl=1:1:Nline,
        
        switch lower(HeaderInfo{Iim}{Inl,1})
            case 'comment'
                fits.writeComment(Fptr,HeaderInfo{Iim}{Inl,3});
            case 'history'
                fits.writeHistory(Fptr,HeaderInfo{Iim}{Inl,3});
            otherwise
                if (isempty(HeaderInfo{Iim}{Inl,3})),
                    HeaderInfo{Iim}{Inl,3} = '\';
                end
                fits.writeKey(Fptr,HeaderInfo{Iim}{Inl,1},HeaderInfo{Iim}{Inl,2},HeaderInfo{Iim}{Inl,3});
        end

    end

end

fits.closeFile(Fptr);
Flag = sign(Fptr);


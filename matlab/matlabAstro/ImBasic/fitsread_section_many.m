function Out=fitsread_section_many(ImageName,StartPix,EndPix,varargin)
%--------------------------------------------------------------------------
% fitsread_section_many function                                   ImBasic
% Description: Read a rectangular region of interest from a single
%              FITS image.
% Input  : - List of FITS image names to read. See create_list.m for
%            options.
%          - Start pixels position [x, y].
%          - End pixels position [x, y].
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'OutputType' - Store the image subsections in one of the
%                           following formats:
%                           'sim' - structure image array
%                                   (see images2sim.m).
%                           'cell' - A cell array of images.
%                           'cube' - A cube of images, in which the
%                                    third dim corresponds to the image
%                                    index.
%            'ReadHead'  - A flag {false|true} indicating if the image
%                          header will be stored in the structure array.
%                          Relevant only for the OutputType=sim option.
%                          Default is false.
% Output : - Cube, SIM or cell array of image subsections.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jun 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Out=fitsread_section_many('PTF_2014*.fits',[11 11],[70 70],'OutputType','cube');
% Reliable: 2
%--------------------------------------------------------------------------

ImageField  = 'Im';
HeaderField = 'Header';
FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';


DefV.OutputType = 'sim';  % {'cell' | 'sim' | 'cube'}
DefV.ReadHead   = false;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


[~,List] = create_list(ImageName,NaN);
Nim = numel(List);

import matlab.io.*

switch lower(InPar.OutputType)
    case 'sim'
        %
    case 'cell'
        Out = cell(1,Nim);
    case 'cube'
        Out = zeros(EndPix(1)-StartPix(1)+1, EndPix(2)-StartPix(2)+1, Nim);
    otherwise
        error('Unknown OutputType option');
end

for Iim=1:1:Nim,
    
   Fptr = fits.openFile(List{Iim});
   switch lower(InPar.OutputType)
       case 'sim'
           Out(Iim).(ImageField) = fits.readImg(Fptr,StartPix,EndPix);
           if (InPar.ReadHead)
               Out(Iim).(HeaderField) = fitsinfo(List{Iim});
               Out(Iim).(FileField)   = List{Iim};
           end
       case 'cell'
           Out{Iim} = fits.readImg(Fptr,StartPix,EndPix);
       case 'cube'
           Out(:,:,Iim) = fits.readImg(Fptr,StartPix,EndPix);
       otherwise
           error('Unknown OutputType option');
   end           
   fits.closeFile(Fptr);

   
end



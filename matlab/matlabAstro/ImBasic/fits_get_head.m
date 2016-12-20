function HeadCell=fits_get_head(Image,HDUnum)
%--------------------------------------------------------------------------
% fits_get_head function                                           ImBasic
% Description: Read a specific Header Data Unit (HDU) in a FITS file
%              into a cell array of {Keyword, Value, comment}.
% Input  : - FITS file name.
%          - Index of HDU. Default is 1.
% Output : - Cell array of {Keyword, Value, comment} containing the FITS
%            header.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jul 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: HeadCell=fits_get_head(Image,3);
% Reliable: 2
%--------------------------------------------------------------------------

if (nargin==1),
    HDUnum = 1;
end

if (isnan(HDUnum)),
    HDUnum = 1;
end

KeyPos = 9;
ComPos = 32;

import matlab.io.*
Fptr = fits.openFile(Image);
Nhdu = fits.getNumHDUs(Fptr);
if (Nhdu>=HDUnum),
    Htype = fits.movAbsHDU(Fptr,HDUnum);
    
    Nkey = fits.getHdrSpace(Fptr);
    HeadCell = cell(Nkey,3);
    for Ikey = 1:1:Nkey
       Card     = fits.readRecord(Fptr,Ikey);
       LenCard = length(Card);
       if (LenCard>=9),
               
           if (strcmpi(Card(KeyPos),'=')),
               HeadCell{Ikey,1}  = spacedel(Card(1:KeyPos-1));
               % update comment position due to over flow
               Islash = strfind(Card(ComPos:end),'/');
               if (isempty(Islash)),
                   UpdatedComPos = ComPos;
               else
                   UpdatedComPos = ComPos + Islash(1)-1;
               end
               Value = Card(KeyPos+1:min(LenCard,UpdatedComPos-1));
               PosAp = strfind(Value,'''');
       
               if (isempty(PosAp)),
                    % possible number
                    Value = str2num(Value);
               else
                   if (length(PosAp)>=2),
                       % a string
                       Value = Value(PosAp(1)+1:PosAp(2)-1);
                   else
                       
                       Value = Card(PosAp(1)+10:end);
                   end
               end
                   
               HeadCell{Ikey,2}  = Value; %Card(KeyPos+1:min(LenCard,ComPos-1));
               if (LenCard>UpdatedComPos),
                  
                   HeadCell{Ikey,3}  = Card(UpdatedComPos+1:end);    
               else
                   HeadCell{Ikey,3}  = '';
               end
           
           else
               
               % look for history and comment keywords
               if (strcmpi(Card(1:7),'HISTORY'))
                   HeadCell{Ikey,1} = 'HISTORY';
                   HeadCell{Ikey,2} = Card(KeyPos:end);
                   HeadCell{Ikey,3} = '';
               end
               if (strcmpi(Card(1:7),'COMMENT'))
                   HeadCell{Ikey,1} = 'COMMENT';
                   HeadCell{Ikey,2} = Card(KeyPos:end);
                   HeadCell{Ikey,3} = '';
               end
           end
       end
    end
    
else
    error('HDUnum requested is larger than the number of HDUs in FITS file');
end

fits.closeFile(Fptr);



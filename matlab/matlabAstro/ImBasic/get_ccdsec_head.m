function [CCDSEC]=get_ccdsec_head(Head,Keyword,OutType)
%--------------------------------------------------------------------------
% get_ccdsec_head function                                         ImBasic
% Description: Get an parse CCD section keyword value from a cell array
%              containing an image header or a structure array
%              containing multiple image headers.
% Input  : - An Nx3 cell array containing the image header,
%            or a structure array in which the Header field contains
%            an Nx3 cell array of image headers.
%          - String containing CCD section keyword to get and parse.
%            Alternatively this can be an array of CCDSEC
%            [xmin xmax ymin ymax]. In this case
%            the function will return the CCDSEC array as is.
%            If empty then will return the image size.
%          - Output type:
%            'mat'  - matrix. Default.
%            'cell' - Each cell contains a CCDSEC.
%            'struct' - A structure array in which each element contains
%                     a CCDSEC field.
% Output : - Matrix of [Xmin Xmax Ymin Ymax] parsed from the CCD section
%            keyword value. Line per image.
% Tested : - Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [CCDSEC]=get_ccdsec_head(H.PrimaryData.Keywords,'CCDSEC');
% Reliable: 2
%--------------------------------------------------------------------------

HeaderField  = 'Header';

Def.OutType = 'mat';
if (nargin==2),
    OutType = Def.OutType;
end

if (isnumeric(Keyword) && ~isempty(Keyword)),
    % already CCDSEC
    CCDSEC = Keyword;
else
    if (~iscell(Keyword) && ~isempty(Keyword)),
       Keyword = {Keyword};
    end

    if (iscell(Head)),
        Sim(1).(HeaderField) = Head;
        clear Head;
        Head = Sim;
    end
    Nim = numel(Head);
    switch lower(OutType)
        case 'mat'
           CCDSEC = zeros(Nim,4);
        otherwise
           % do nothing
    end

    for Iim=1:1:Nim,
        
        if (isempty(Keyword)),
            [NewCellHead]=cell_fitshead_getkey(Head(Iim).(HeaderField),{'NAXIS1','NAXIS2'},'NaN');
            
            Xmin = 1;
            Ymin = 1;
            Xmax = NewCellHead{1,2};
            Ymax = NewCellHead{2,2};
            
            
        else
            [NewCellHead]=cell_fitshead_getkey(Head(Iim).(HeaderField),Keyword,'NaN');
            CCDSEC_Val = NewCellHead{2};

            if (isnan(CCDSEC_Val)),
                error('can not find CCDSEC keyword in image header');
            else
                Splitted   = regexp(CCDSEC_Val,'[:\[\],]','split');

                Xmin       = str2double(Splitted{2});
                Xmax       = str2double(Splitted{3});
                Ymin       = str2double(Splitted{4});
                Ymax       = str2double(Splitted{5});
            end
        end
         switch lower(OutType)
             case 'mat'
                 CCDSEC(Iim,:) = [Xmin, Xmax, Ymin, Ymax];
             case 'cell'
                 CCDSEC{Iim} = [Xmin, Xmax, Ymin, Ymax];
             case 'struct'
                 CCDSEC(Iim).CCDSEC = [Xmin, Xmax, Ymin, Ymax];
             otherwise
                 error('Unknown OutType option');
         end
    end
end
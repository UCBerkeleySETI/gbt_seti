function WCS=fits_get_wcs(Sim,varargin)
%--------------------------------------------------------------------------
% fits_get_wcs function                                           ImAstrom
% Description: Get the WCS keywords information from a SIM image structure
%              array or FITS images.
% Input  : - A structure array of images or SIM class with header
%            information. Alternatively, this can be a string or a cell
%            array of strings containing FITS image names, or a 3 column
%            cell containing a single header.
%            If the input is a FITS image then this function will read
%            only the header.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'HDUnum' - HDU number from which to read the header.
%                       Default is 1.
% Output : - WCS structure array containing all the WCS keyword values.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Dec 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: WCS=fits_get_wcs('*.fits');
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

DefV.HDUnum = 1;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


KeywordCell{1}  = 'RADECSYS';
KeywordCell{2}  = 'CTYPE1';
KeywordCell{3}  = 'CTYPE2';
KeywordCell{4}  = 'CUNIT1';
KeywordCell{5}  = 'CUNIT2';
KeywordCell{6}  = 'CRPIX1';
KeywordCell{7}  = 'CRPIX2';
KeywordCell{8}  = 'CRVAL1';
KeywordCell{9}  = 'CRVAL2';
KeywordCell{10} = 'CD1_1';
KeywordCell{11} = 'CD1_2';
KeywordCell{12} = 'CD2_1';
KeywordCell{13} = 'CD2_2';
KeywordCell{14} = 'CDELT1';
KeywordCell{15} = 'CDELT2';
KeywordCell{16} = 'PC1_1';
KeywordCell{17} = 'PC1_2';
KeywordCell{18} = 'PC2_1';
KeywordCell{19} = 'PC2_2';
KeywordCell{20} = 'LONPOLE';
KeywordCell{21} = 'LATPOLE';
KeywordCell{22} = 'EQUINOX';
%KeywordCell{23} = 'MJD-OBS';


if (iscell(Sim) && size(Sim,1)>1 && size(Sim,2)==3),
    CellWCS  = cell_fitshead_getkey(Sim,KeywordCell,'NaN','first');
    WCS      = cell2struct(CellWCS(:,2).',KeywordCell,2);
    % Get the sip distortion information 
    if ((~isempty(strfind(WCS.CTYPE1,'SIP')))||(~isempty(strfind(WCS.CTYPE2,'SIP')))) 
        WCS.sip = fits_get_sip(Sim);
    end
else    
    if (issim(Sim) || isstruct(Sim)),
        Nim = numel(Sim);
        Issip = zeros(Nim,1);                               %initialization
        for Iim=1:1:Nim,
            CellWCS  = cell_fitshead_getkey(Sim(Iim).(HeaderField),KeywordCell,'NaN','first');
            WCS(Iim) = cell2struct(CellWCS(:,2).',KeywordCell,2);
            
            % Get sip distortion information
            if ((~isempty(strfind(WCS(Iim).CTYPE1,'SIP'))) || (~isempty(strfind(WCS(Iim).CTYPE2,'SIP'))))         
                WCS_sip(Iim) = fits_get_sip(Sim(Iim).(HeaderField));
                Issip(Iim) = 1;
            end
                   
        end
        if (sum(Issip)>0)                   % => at least one of the images has sip distortion information.
            for Iim = 1:1:Nim,
                if (Issip(Iim))
                    WCS(Iim).sip = WCS_sip(Iim);
                else
                    WCS(Iim).sip = NaN;
                end
            end
        end
    else
        if (ischar(Sim)),
            [~,List] = create_list(Sim,NaN);
        elseif (iscell(Sim)),
            List = Sim;
        else
            error('Unknown input type');
        end
        Nim = numel(List);
        Issip = zeros(Nim,1);  
        for Iim=1:1:Nim,
            HeadCell = fits_get_head(List{Iim},InPar.HDUnum);
            CellWCS  = cell_fitshead_getkey(HeadCell,KeywordCell,'NaN','first');
            WCS(Iim) = cell2struct(CellWCS(:,2).',KeywordCell,2);
            
            % Get sip distortion information
            if ((~isempty(strfind(WCS(Iim).CTYPE1,'SIP')))||(~isempty(strfind(WCS(Iim).CTYPE2,'SIP'))))
                WCS_sip(Iim) = fits_get_sip(HeadCell);
                Issip(Iim) = 1;
            end
               
        end
        if (sum(Issip)>0)           % => at least one sip
            for Iim = 1:1:Nim,
                if (Issip(Iim))
                    WCS(Iim).sip = WCS_sip(Iim);
                else 
                    WCS(Iim).sip = NaN;
                end
            end
        end

    end
end


% fix WCS key/val
Nim = numel(WCS);
for Iim=1:1:Nim,
    
    if (isnan(WCS(Iim).CUNIT1)),
       WCS(Iim).CUNIT1 = 'deg';
    end
    if (isnan(WCS(Iim).CUNIT2)),
       WCS(Iim).CUNIT2 = 'deg';
    end

    WCS(Iim).CUNIT1 = spacedel(WCS(Iim).CUNIT1);
    WCS(Iim).CUNIT2 = spacedel(WCS(Iim).CUNIT2);
    
    if (~isnan(WCS(Iim).PC1_1)),
        % use PC matrix instead of CD matrix
        % see Eq. 1 in Calabertta & Greisen (2002)
        if (isnan(WCS(Iim).CD1_1)),
            WCS(Iim).CD1_1 = WCS(Iim).CDELT1.*WCS(Iim).PC1_1;
            WCS(Iim).CD1_2 = WCS(Iim).CDELT1.*WCS(Iim).PC1_2;
            WCS(Iim).CD2_1 = WCS(Iim).CDELT2.*WCS(Iim).PC2_1;
            WCS(Iim).CD2_2 = WCS(Iim).CDELT2.*WCS(Iim).PC2_2;
        else
            % ignore PC matrix sibcd CD matrix is given too
        end
    end


    if (isnan(WCS(Iim).CD1_1)),
       % try to use CDELT1/2
       WCS(Iim).CD1_1 = WCS(Iim).CDELT1; %CellWCS{14};
       WCS(Iim).CD2_2 = WCS(Iim).CDELT2; %CellWCS{15};
       WCS(Iim).CD1_2 = 0;
       WCS(Iim).CD2_1 = 0;
    end

    WCS(Iim).CD = [WCS(Iim).CD1_1, WCS(Iim).CD1_2; WCS(Iim).CD2_1, WCS(Iim).CD2_2];
end
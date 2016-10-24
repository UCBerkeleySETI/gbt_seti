function WCS=get_fits_wcs(FitsName,TableInd,TableName)
%-------------------------------------------------------------------------
% get_fits_wcs function                                          ImAstrom
% Description: Get WCS keywords from FITS image.
%              Obsolote - See instead fits_get_wcs.m
% Input  : - FITS file name (string) or a structure containing
%            the image header (as returned by fitsinfo function).
%          - In case the header data unit is a vector containing several
%            extension, then you can specify the extension name.
%            Default is NaN.
%          - String containing the name of the header data unit
%            If TableInd is NaN or empty than default is 'PrimaryData',
%            else default is BinaryTable(TableInd).
% Output : - Structure of WCS keywords
%            WCS.RADECSYS
%            WCS.CTYPE1
%            WCS.CTYPE2
%            WCS.CUNIT1   - default is 'deg'.
%            WCS.CUNIT2   - default is 'deg'.
%            WCS.CRPIX1
%            WCS.CRPIX2
%            WCS.CRVAL1
%            WCS.CRVAL2
%            WCS.CD1_1      
%            WCS.CD1_2      
%            WCS.CD2_1      
%            WCS.CD2_2 
%            WCS.CD
%            WCS.LONPOLE
%            WCS.LATPOLE
%            WCS.EQUINOX
%            WCS.MJD-OBS
% Tested : Matlab 7.0
%     By : Eran O. Ofek                      June 2005 
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
% Example: WCS=get_fits_wcs('PTF_201305223828_c_p_scie_t091110_u016150547_f02_p100019_c00.ctlg',[],'Image')
%-------------------------------------------------------------------------

Def.TableInd  = NaN;
Def.TableName = [];
if (nargin==1),
   TableInd   = Def.TableInd;
   TableName  = Def.TableName;
elseif (nargin==2),
   TableName  = Def.TableName;
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end



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


%[CellWCS,WCS]   = get_fits_keyword(FitsName,KeywordCell,TableInd,TableName);


HeadCell = fits_get_head(FitsName,TableInd);
CellWCS  = cell_fitshead_getkey(HeadCell,KeywordCell,'NaN','first');
WCS      = cell2struct(CellWCS(:,2).',KeywordCell,2);

%WCS.RADECSYS    = CellWCS{1};
%WCS.CTYPE1      = CellWCS{2};
%WCS.CTYPE2      = CellWCS{3};
%WCS.CUNIT1      = CellWCS{4};
%WCS.CUNIT2      = CellWCS{5};
%WCS.CRPIX1      = CellWCS{6};
%WCS.CRPIX2      = CellWCS{7};
%WCS.CRVAL1      = CellWCS{8};
%WCS.CRVAL2      = CellWCS{9};
%WCS.CD1_1       = CellWCS{10};
%WCS.CD1_2       = CellWCS{11};
%WCS.CD2_1       = CellWCS{12};
%WCS.CD2_2       = CellWCS{13};
%WCS.CDELT1      = CellWCS{14};
%WCS.CDELT2      = CellWCS{15};

if (isnan(WCS.CUNIT1)),
   WCS.CUNIT1 = 'deg';
end
if (isnan(WCS.CUNIT2)),
   WCS.CUNIT2 = 'deg';
end

if (~isnan(WCS.PC1_1)),
    % use PC matrix instead of CD matrix
    % see Eq. 1 in Calabertta & Greisen (2002)
    if (isnan(WCS.CD1_1)),
        WCS.CD1_1 = WCS.CDELT1.*WCS.PC1_1;
        WCS.CD1_2 = WCS.CDELT1.*WCS.PC1_2;
        WCS.CD2_1 = WCS.CDELT2.*WCS.PC2_1;
        WCS.CD2_2 = WCS.CDELT2.*WCS.PC2_2;
    else
        % ignore PC matrix sibcd CD matrix is given too
    end
end


if (isnan(WCS.CD1_1)),
   % try to use CDELT1/2
   WCS.CD1_1 = CellWCS{14};
   WCS.CD2_2 = CellWCS{15};
   WCS.CD1_2 = 0;
   WCS.CD2_1 = 0;
end

WCS.CD = [WCS.CD1_1, WCS.CD1_2; WCS.CD2_1, WCS.CD2_2];

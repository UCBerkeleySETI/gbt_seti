function Spec=read_sdss_spec(File,DRformat)
%------------------------------------------------------------------------------
% read_sdss_spec function                                                 sdss
% Description: Read SDSS spectra in FITS format into matlab structure.
% Input  : - String containing FITS file name, or cell array of FITS
%            file names.
%          - SDSS file format {'new' | 'old'}. Default is 'new'.
% Output : - A structure array containing the structure of spectra.
%            Each element containing the following fields:
%            .Wave        - Wavelength [Ang].
%            .Pix         - pixel index
%            .Flux        - Flux [erg cm^-2 s^-1 A^-1]
%            .FluxContSub - Continuum subtracted flux [erg cm^-2 s^-1 A^-1]
%            .Error       - Error in flux [erg cm^-2 s^-1 A^-1]
%            .Mask        - Mask containing flags (e.g., 0 if OK).
% Tested : Matlab 7.3
%     By : Eran O. Ofek                    Feb 2008
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%------------------------------------------------------------------------------
Def.DRformat = 'new';
if (nargin==1),
    DRformat = Def.DRformat;
elseif (nargin==2),
    % do nothing
else
    error('Illegal number of input arguments');
end

if (iscell(File)==1),
   N = length(File);
   SpecList = File;
else
   SpecList{1} = File;
   N = 1;
end

%Spec = cell(N,1);
for I=1:1:N,
    switch lower(DRformat)
       case 'old'
           NormConst = 1e-17;

           %Spec = cell(N,1);
           % Instructions:
           % http://www.sdss.org/dr6/products/spectra/read_spSpec.html
           Key=get_fits_keyword(SpecList{I},{'COEFF0','COEFF1'});
           MatF = fitsread(SpecList{I});
           Nl   = size(MatF,2);
           Ind  = (0:1:Nl-1).';
           
           Spec(I).Wave        = 10.^(Key{1} + Ind.*Key{2});
           Spec(I).Pix         = Ind + 1;
           Spec(I).Flux        = MatF(1,:).'.*NormConst;
           Spec(I).FluxContSub = MatF(2,:).'.*NormConst;
           Spec(I).Error       = MatF(3,:).'.*NormConst;
           Spec(I).Mask        = MatF(4,:).';
        case 'new'
           [~,~,~,Col,Table]=get_fitstable_col(File);
           
           %plot(10.^Table{Col.loglam},Table{Col.flux})
           Spec(I).Wave        = 10.^Table{Col.loglam};
           Spec(I).Flux        = Table{Col.flux};
           Spec(I).Mask        = Table{Col.or_mask};
           Spec(I).Sky         = Table{Col.sky};
           Spec(I).Model       = Table{Col.model};
           Spec(I).Wdisp       = Table{Col.wdisp};
           
        otherwise
           error('Unknown DRformat option');
           
    end
           
end

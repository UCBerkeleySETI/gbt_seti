function [Energy,EnergyErr]=hst_acs_zp_apcorr(Aper,Filter)
%--------------------------------------------------------------------------
% hst_acs_zp_apcorr function                                        ImPhot
% Description: Given aperture radius for photometry, and Hubble Space
%              Telecsope (HST) ACS-WFC filter name, return the fraction,
%              and error in fraction, of the encircled energy within the
%              aperture.
% Input  : - Aperture radius [arcseconds].
%          - HST/ACS/WFC filter name
% Output : - Fraction of energy encircled within aperture.
%          - Error in the fraction of energy encircled within aperture.
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Feb 2007
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reference: Sirianni et al. (2005)
% Example: [Energy,EnergyErr]=hst_acs_zp_apcorr(0.3,'F475W');
% Reliable: 2
%--------------------------------------------------------------------------

load HST_ACS_WFC_Encircled_Energy.dat;
Data = HST_ACS_WFC_Encircled_Energy;
ColAper = 1;
switch lower(Filter)
 case 'f435w'
    Col = [2 3];
 case 'f475w'
    Col = [4 5];
 case 'f555w'
    Col = [6 7];
 case 'f606w'
    Col = [8 9];
 case 'f625w'
    Col = [10 11];
 case 'f775w'
    Col = [12 13];
 case 'f814w'
    Col = [14 15];
 case 'f850lp'
    Col = [16 17];
 case 'f892n'
    Col = [18 19];
 otherwise
    error('Unknown Filter option');
end

Energy    = interp1(Data(:,ColAper),Data(:,Col(1)),Aper,'spline');
EnergyErr = interp1(Data(:,ColAper),Data(:,Col(2)),Aper,'nearest');

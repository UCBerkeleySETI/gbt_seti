function S=sn_det2psf_signal(SN,Sigma,B,R)
%--------------------------------------------------------------------------
% sn_det2psf_signal function                                        ImPhot
% Description: Given detection S/N calculate the PSF signal.
% Input  : - S/N for detection
%          - Width of Gaussian PSF (sigma in pix).
%          - Background [e-]
%          - Readnoise [e-].
% Output : - Signal [e-]
% See also: sn_psf_det.m, optimal_phot_aperture.m, sn_aper_phot.m,
%           sn_psfn_phot.m, sn_psf_phot.m, sn_calc.m
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Sep 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: S=sn_det2psf_signal(10,2,100,0)
% Reliable: 2
%--------------------------------------------------------------------------

S  = SN.*sqrt(4.*pi.*(B+R.^2).*Sigma.^2);

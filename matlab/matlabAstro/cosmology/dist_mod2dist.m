function [Dist,Z]=dist_mod2dist(DM)
%--------------------------------------------------------------------------
% dist_mod2dist function                                         cosmology
% Description: Convert distance modulous to luminosity distance and
%              redshift.
% Input  : - Distance modulous [mag].
% Output : - Luminosity distance [pc].
%          - Redshift.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Dist,Z]=dist_mod2dist(35)
% Reliable: 2
%--------------------------------------------------------------------------


Dist = 10.^(0.2.*(DM+5));
if (nargout>1),
    Z    = inv_lum_dist(Dist,'LD');
end

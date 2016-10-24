function E=true2eccentric_anomaly(Nu,ecc)
%--------------------------------------------------------------------------
% true2eccentric_anomaly function                                    ephem
% Description: 
% Input  : - 
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
% Output : - 
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Dec 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: E=true2eccentric_anomaly(1,1);
% Reliable: 2
%--------------------------------------------------------------------------

E = 2.*atan(sqrt((1-ecc)./(1+ecc)).*tan(0.5.*Nu));

function [SumMag,FracFlux]=sum_mag(Mag,Dim)
%--------------------------------------------------------------------------
% sum_mag function                                                  ImPhot
% Description: Sum a set of magnitudes.
% Input  : - Matrix or vector of magnitudes to sum.
%          - Dimesnions along to sum. Default is 1.
% Output : - Sum of magnitude in mag.
%          - Fraction of flux for each entry out of the total flux.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Apr 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [SumMag,FracFlux]=sum_mag([15;17;19])
% Reliable: 2
%--------------------------------------------------------------------------

Def.Dim = 1;
if (nargin==1),
    Dim = Def.Dim;
end

Flux     = 10.^(-0.4.*Mag);
SumFlux  = nansum(Flux,Dim);
SumMag   = -2.5.*log10(SumFlux);
FracFlux = bsxfun(@rdivide,Flux,SumFlux);

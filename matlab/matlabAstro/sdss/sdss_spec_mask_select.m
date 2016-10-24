function [GoodI,BadI]=sdss_spec_mask_select(Mask);
%------------------------------------------------------------------------------
% sdss_spec_mask_select function                                          sdss
% Description: Given a mask of the SDSS spectra in decimal format return
%              indices for all the masks which are good and bad.
%              Good mask defined to have "SP_MASK_OK" | "SP_MASK_EMLINE".
% Input  : - Vector of masks in decimal base.
% Output : - Vector of good indices.
%          - Vector of bad indices.
% Tested : MATLAB 7.8
%     By : Eran O. Ofek                   October 2009
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%------------------------------------------------------------------------------


Good{1} = '1000000000000000000000000000000';
Good{2} = '0000000000000000000000000000000';
Ng      = length(Good);

N = length(Mask);

BinMask = dec2bin(Mask,27);

Res = zeros(N,Ng);
for I=1:1:N,
   for Ig=1:1:Ng,
      Res(I,Ig) = strcmp(BinMask(I,:),Good{Ig});
   end
end

GoodI = find(Res(:,1)==1 | Res(:,2)==1);
BadI  = find(Res(:,1)==0 & Res(:,2)==0);

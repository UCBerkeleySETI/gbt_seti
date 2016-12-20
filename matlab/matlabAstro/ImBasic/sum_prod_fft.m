function [SumFFT,CubeFFT]=sum_prod_fft(C1,C2,Sigma,DimIm,ConjC1)
%--------------------------------------------------------------------------
% sum_prod_fft function                                            ImBasic
% Description: Calculate the sum of products of 2-D fast fourier transform
%              of two cubes. I.e., calculate
%              sum_{i}{conj(fft(P_i))*fft(R_i)/Sigma^2}
%              were the conj is optional and sigma is an optional
%              normalization. P_i and R_i are 2D arrays.
% Input  : - A 3-D cube (P).
%          - A 3-D cube (R).
%          - Sigma.
%          - Dimension in the cube on which to sum. Default is 3.
%            This means that the fft2 will be done on dim 1 and 2
%            and the summation on the third dimension.
%          - A flag indicating if complex conjugate of the first cube (P)
%            will be performed. Default is true.
% Output : - Sum of products of FFTs.
%          - Cube of products of FFTs (no summation applied).
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    May 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: C1=rand(1024,1024,50); Sigma=rand(50,1);
%          SumFFT=sum_prod_fft(C1,C1,Sigma,3);
% Reliable: 2
%--------------------------------------------------------------------------

Def.Sigma  = 1;
Def.DimIm  = 3;
Def.ConjC1 = true;
if (nargin==2),
    Sigma  = Def.Sigma;
    DimIm  = Def.DimIm;
    ConjC1 = Def.ConjC1;
elseif (nargin==3),
    DimIm  = Def.DimIm;
    ConjC1 = Def.ConjC1;
elseif (nargin==4),
    ConjC1 = Def.ConjC1;
elseif (nargin==5),
    % do nothing
else
    error('Illegal number of input arguments');
end

    
%DefV. = 
%InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

%DimIm = 3;
DimFFT = setdiff([1 2 3],DimIm);
C1 = permute(C1,[DimFFT(1), DimFFT(2), DimIm]);
C2 = permute(C2,[DimFFT(1), DimFFT(2), DimIm]);
%C1 = shiftdim(C1,DimIm - 3);
%C2 = shiftdim(C2,DimIm - 3);

InvVar = 1./Sigma.^2;
Size = size(InvVar);
if (Size(1)>=Size(2)),
    InvVar = shiftdim(InvVar,-2);
else
    InvVar = shiftdim(InvVar,-1);
end

F1 = fft(fft(C1,[],1),[],2);
F2 = fft(fft(C2,[],1),[],2);
if (ConjC1),
    F1 = conj(F1);
end
% if (ConjC2),
%     F2 = conj(F2);
% end
if (nargout>1),
    CubeFFT = bsxfun(@times,F1.*F2,InvVar);
    SumFFT  =  squeeze(sum(CubeFFT,3));
else
    SumFFT = squeeze(sum(bsxfun(@times,F1.*F2,InvVar),3));
end


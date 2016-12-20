function Filt=medfilt2nan(Mat,Size,Loop);
%--------------------------------------------------------------------------
% medfilt2nan function                                             ImBasic
% Description: 2-D median filter that ignores NaN's. This is similar to
%              medfilt2.m, but ignoring NaNs.
% Input  : - Matrix.
%          - Filter Size. Default is 3. Must be an odd number.
%          - Use loops {'y' | 'n'}, default is 'n'. Use loops in case
%            memory is limited.
% Output : Median filtered matrix.
% Tested : Matlab 7.11
%     By : Eran O. Ofek                   January 2011
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%--------------------------------------------------------------------------
Def.Size = 3;
Def.Loop = 'n';
if (nargin==1),
   Size  = Def.Size;
   Loop  = Def.Loop;
elseif (nargin==2),
   Loop  = Def.Loop;
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end

HalfSize = Size(1).*0.5-0.5;
SizeMat  = size(Mat);
%Mat   = Mat(1+HalfSize:(SizeMat(1)-HalfSize),1+HalfSize:(SizeMat(2)-HalfSize));


TotalSizeMat = numel(Mat);


[G1,G2]=meshgrid([1:1:Size(1)],[1:1:Size(1)]);
BasicInd = reshape(sub2ind(SizeMat,G1,G2),numel(G1),1);
Rep      = repmat(BasicInd,1,(SizeMat(1)-HalfSize.*2).*(SizeMat(2)-HalfSize.*2));


OrigVec = [1:1:SizeMat(1)-HalfSize.*2] - 1;
IndVec  = [1:1:SizeMat(1)-HalfSize.*2]';
%for I=1:1:(SizeMat(2)-HalfSize.*2),
%   Vec(IndVec+(SizeMat(1)-HalfSize.*2).*(I-1)) = OrigVec + SizeMat(1).*(I-1);
%end

A = reshape(repmat(OrigVec,(SizeMat(2)-HalfSize.*2),1)',1, ...
        (SizeMat(2)- HalfSize.*2)*(SizeMat(1)-HalfSize.*2));
B = [0:(SizeMat(2)-HalfSize.*2)-1]*SizeMat(1);
B = reshape(repmat(B,(SizeMat(1)-HalfSize.*2),1),1, ...
       (SizeMat(1)-HalfSize.*2)*(SizeMat(2)-HalfSize.*2));
Vec = A + B;


SuperInd = bsxfun(@plus,Rep,Vec);

switch lower(Loop)
 case 'y'
    VecS = [1:1:SizeMat(2)-HalfSize.*2];
    for I=1:1:(SizeMat(1)-HalfSize.*2),
       Med(VecS+(SizeMat(2)-HalfSize.*2).*(I-1)) = nanmedian(Mat(SuperInd(:,VecS+(SizeMat(2)-HalfSize.*2).*(I-1)) ));
    end
 case 'n'
    Med      = nanmedian(Mat(SuperInd));
 otherwise
    error('Unknown Loop option');
end
Filt     = reshape(Med, SizeMat(1)-HalfSize.*2, SizeMat(2)-HalfSize.*2);
Filt     = padarray(Filt,[HalfSize HalfSize],NaN,'both');



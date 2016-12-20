function [Mean,StD,NpixUse]=clip_image_mean(Mat,varargin)
%--------------------------------------------------------------------------
% clip_image_mean function                                         ImBasic
% Description: Given a cube of images (image index is the third dimension),
%              calculate the sigma clipped mean/median.
% Input  : - Cube of images in which the 3rd dimension is the image index.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'MeanFun' - Function handle for mean calculation
%                        (e.g., @nanmedian). Default is @nanmean.
%            'RejectMethod' - Clipping rejection method:
%                        'std' - standard deviation
%                        'minmax' - min/max number of images.
%                        'perc'   - lower/upper percentile of images.
%            'Reject' - Rejection parameters [low high].
%                       e.g., [3 3] for 'std', [0.05 0.85] for 'perc'.
%            'MaxIter'- Number of sigma clipping iterations. 0 for no
%                       sigma clipping.
% Output : - Mean image
%          - StD image
%          - Image of number of images used per pixel.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Mean,StD,NpixUse]=clip_image_mean(rand(2,2,100));
% Reliable: 2
%--------------------------------------------------------------------------

Dim = 3;

DefV.MeanFun          = @nanmean;   % Fun(Mat,Dim)
DefV.RejectMethod     = 'std';      % 'minmax','std','perc',
DefV.Reject           = [3 3];      % low/high rejection
DefV.MaxIter          = 1;          % 0 no iterations
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


Mean = squeeze(InPar.MeanFun(Mat,Dim));
StD  = squeeze(std(Mat,[],Dim));
Size = size(Mat);
Nim  = Size(3);
NpixUse = ones(size(Mean)).*Nim;

switch lower(InPar.RejectMethod)
    case 'perc'
        
        InPar.Reject = [ceil(InPar.Reject(1).*Nim), ceil(Nim - InPar.Reject(2).*Nim)];
        InPar.RejectMethod = 'minmax';
    otherwise
        % do nothing
end
        
Iter     = 0;
ContLoop = Iter<InPar.MaxIter;
while ContLoop,
    Iter = Iter + 1;
    switch lower(InPar.RejectMethod)
        case 'std'
            StD  = squeeze(std(Mat,[],Dim));
            Flag = bsxfun(@gt,Mat,Mean-StD.*abs(InPar.Reject(1))) & bsxfun(@lt,Mat,Mean+StD*abs(InPar.Reject(1)));
            MatN = Mat;
            MatN(~Flag) = NaN;
            NpixUse = sum(~isnan(MatN),Dim);
            Mean = squeeze(InPar.MeanFun(MatN,Dim));
        case 'minmax'
            % deal also with 'perc'
            SortedMat = sort(Mat,Dim);
            SortedMat(:,:,1:InPar.Reject(1))         = NaN;
            SortedMat(:,:,1:end-InPar.Reject(2):end) = NaN;
            NpixUse = sum(~isnan(SortedMat),Dim);
            Mean = squeeze(InPar.MeanFun(SortedMat,Dim));
        case 'none'
            NpixUse = sum(~isnan(Mat),Dim);
            Mean = squeeze(InPar.MeanFun(Mat,Dim));
        otherwise
            error('Unknown RejectMethod option');
    end
    ContLoop = Iter<InPar.MaxIter;
end




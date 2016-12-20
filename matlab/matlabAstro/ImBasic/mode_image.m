function [Mode,Percentile,MeanErr]=mode_image(CurImage,RemOut)
%---------------------------------------------------------------------------
% mode_image function                                               ImBasic
% Description: Calculate the mode value of a 1D or 2D matrix.
%              This function use bins widths which are adjusted to
%              the noise statistics of the image.
% Input  : - A 2D matrix for which to calculate the mode.
%          - Remove outliers prior to mode calculation. Default is true.
% Output : - The mode of the image.
%          - The lower and upper 68.27%, 95.45%, 99.73% percentiles of the
%            values in the image (3 by 2 matrix). See err_cl.m for
%            details.
%          - Estimate of the error in the mean value of the image
%            based on dividing half the 68% percentile by the square root
%            of number of pixels.
%            This is used as the resolution inw hich the mode is
%            calculated.
% Tested : Matlab 7.10
%     By : Eran O. Ofek                      June 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [Mode,Percentile,MeanErr]=mode_image(randn(1000,1000))
%          [Mode,Percentile,MeanErr]=mode_image(randn(100000,1))
% Reliable: 2
%---------------------------------------------------------------------------


if (nargin==1),
    RemOut = true;
end

Percentile = err_cl(CurImage(:),[0.6827; 0.9545; 0.9973]);
if (RemOut),
    CurImage = CurImage(CurImage>Percentile(1,1) & CurImage<Percentile(1,2));
    Percentile = err_cl(CurImage(:),[0.6827; 0.9545; 0.9973]);
end
% Estimate the error in the mean
MeanErr = 10.*0.5.*(Percentile(1,2) - Percentile(1,1))./ ...
                sqrt(numel(CurImage)-1);

% Estimate the mode of the image
Factor = 1./MeanErr;
Mode = mode(round(Factor.*CurImage(:))./Factor);

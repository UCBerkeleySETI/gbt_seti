function FFTI=shifted_fft2(I_NS)
%--------------------------------------------------------------------------
% shifted_fft2 function                                             Optics
% Description: Given a matrix, return fftshift(fftshift(fft2(Matrix),1),2).
%              This is the shifted fft of a matrix.
% Input  : - A matrix.
% Output : - The shifted fft2 of the matrix.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Jan 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: FFTI=shifted_fft2(I_NS);
% Reliable: 2
%--------------------------------------------------------------------------

FFTI = fftshift(fftshift(fft2(I_NS),1),2);

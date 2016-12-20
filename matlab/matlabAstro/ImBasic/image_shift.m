function [OutImage]=image_shift(Image,Shift,varargin)
%--------------------------------------------------------------------------
% image_shift function                                             ImBasic
% Description: Shift an image in X and Y.
% Input  : - A single 2D image to shift.
%          - [ShiftX, ShiftY] in pixels.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'XData' - A two-element vector that, when combined with
%                      'YData', specifies the spatial location of the
%                      output image in the 2-D output space X-Y. The two
%                      elements of 'XData' give the x-coordinates
%                      (horizontal) of the first and last columns of the
%                      output image, respectively.
%                      Default is empty. If empty will set to
%                      ['MinShiftX', NAXIS1+'MaxShiftX'].
%            'YData' - Like 'XData', but for the Y axis.
%                      If empty will set to
%                      ['MinShiftY', NAXIS2+'MaxShiftY'].
%            'MaxShiftX' - Control the size of the output image.
%                      Default is empty. If empty use Shift(1).
%                      This is usefull if several images with different
%                      shift are required to have the same size.
%                      In this case use max(ShiftX).
%            'MaxShiftY' - Like 'MaxShiftX', but for the Y-axis.
%            'MinShiftX' - Control the size of the output image.
%                      Default is empty. If empty then 1.
%                      This is usefull if several images with different
%                      shift are required to have the same size.
%                      In this case use min(ShiftX).
%            'MinShiftY' - Like 'MinShiftX', but for the Y-axis.
%            'TInterpolant' - Transformation interpolant.
%                      See makeresampler.m for options.
%                      Default is 'cubic'.
%            'TPadMethod' - Transformation padding method.
%                      See makeresampler.m for options.
%                      Default is 'bound'.
%            'FillValues' - Value to use in order to fill missing data
%                      points. Default is NaN.
% Output : - Shifted image.
% Tested : Matlab R2013a
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: OutIm=image_shift(rand(5,5),[2 3]);
% Reliable: 2
%--------------------------------------------------------------------------


DefV.XData        = [];
DefV.YData        = [];
DefV.MaxShiftX    = [];
DefV.MaxShiftY    = [];
DefV.MinShiftX    = [];
DefV.MinShiftY    = [];
DefV.TInterpolant = 'cubic';
DefV.TPadMethod   = 'bound';
DefV.FillValues   = NaN;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

[NAXIS2, NAXIS1] = size(Image);

if (isempty(InPar.MaxShiftX)),
    InPar.MaxShiftX = Shift(1);
end
if (isempty(InPar.MaxShiftY)),
    InPar.MaxShiftY = Shift(2);
end
if (isempty(InPar.MinShiftX)),
    InPar.MinShiftX = 1;
end
if (isempty(InPar.MinShiftY)),
    InPar.MinShiftY = 1;
end

if (isempty(InPar.XData)),
    InPar.XData = [InPar.MinShiftX, NAXIS1+InPar.MaxShiftX];
end
if (isempty(InPar.YData)),
    InPar.YData = [InPar.MinShiftY, NAXIS2+InPar.MaxShiftY];
end

if (isnumeric(Shift)),
    AffineMatrix = [1 0 Shift(1); 0 1 Shift(2); 0 0 1];
    TForm = maketform('affine',AffineMatrix');
else
    TForm = Shift;
end

RSamp    = makeresampler(InPar.TInterpolant,InPar.TPadMethod);
OutImage = imtransform(Image,TForm,RSamp,...
                      'FillValues',InPar.FillValues,'XData',InPar.XData,'YData',InPar.YData);
   

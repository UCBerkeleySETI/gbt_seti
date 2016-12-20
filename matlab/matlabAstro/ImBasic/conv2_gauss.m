function Output=conv2_gauss(Array,Type,Size,Par,Shape,Cut)
%--------------------------------------------------------------------------
% conv2_gauss function                                             ImBasic
% Description: Convolve an image with a Gaussian or a top hat.
% Input  : - 2D array.
%          - Kernel function type, options are:
%            'gauss' - gaussian (default).
%            'flat'  - top hat function (e.g., 000111000)
%          - Kernel box size (odd number), default is 3.
%          - Kernel parameters:
%            for 'gauss': [SigmaX, SigmaY, Rho, Norm],
%                         default is [1 1 0 1].
%            for 'flat' : [Norm],
%                         default is [1].
%          - Kernel shape: {'circle' | 'box'}, default is 'circle'.
%            If 'circle', then radius = half the box size.
%          - Cut edges (so that the size of the output array is the same
%            as the input array). {'y' | 'n'}, default is 'y'.
% Output : - Convolved image
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Jan 2010
%    URL : http://weizmann.ac.il/home/eofek/matlab/ 
% Example: Output=conv2_gauss(rand(100,100));
%          Output=conv2_gauss(rand(100,100),'gauss',20,[5 5 0 1]);
% Reliable: 2
%--------------------------------------------------------------------------
DefType  = 'gauss';
DefSize  = 3;
DefPar   = [1 1 0 1];
DefShape = 'circle';
DefCut   = 'y';

if (nargin==1),
   Type     = DefType;
   Size     = DefSize;
   Par      = DefPar;
   Shape    = DefShape;
   Cut      = DefCut;
elseif (nargin==2),
   Size     = DefSize;
   Par      = DefPar;
   Shape    = DefShape;
   Cut      = DefCut;
elseif (nargin==3),
   Par      = DefPar;
   Shape    = DefShape;
   Cut      = DefCut;
elseif (nargin==4),
   Shape    = DefShape;
   Cut      = DefCut;
elseif (nargin==5),
   Cut      = DefCut;
elseif (nargin==6),
   % do nothing
else
   error('Illegal number of input arguments');
end


Center      = ceil(Size.*0.5);

switch Type
 case 'gauss'
    % Gaussian kernel
    [MatX,MatY] = meshgrid((1:1:Size),(1:1:Size));
    Kernel      = bivar_gauss(MatX,MatY,[Center Center [Par] 0]);
 case 'flat'
    % top hat kernel
    [MatX,MatY] = meshgrid((1:1:Box),(1:1:Box));
    Kernel      = ones(Size,Size);
 otherwise
    error('Unknown Type option');
end

switch Shape
 case 'circle'
    MatR = sqrt((MatX-Center).^2 + (MatY-Center).^2);
    I    = find(MatR>(0.5.*Size));
    Kernel(I) = 0;
 case 'box'
    % do nothing
 otherwise
    error('Unknown Shape option');
end 

%Output = conv2(Array,Kernel,'same');
Output = conv_fft2(Array,Kernel,'same');

switch Cut
 case 'y'
    Output = Output(Center:1:end-Center+1, Center:1:end-Center+1);
 case 'n'
    % do nothing
 otherwise
    error('Unknown Cut option');
end 

    


function [MatVal,MatX,MatY]=ds9_getbox(Coo,Method,CooType)
%------------------------------------------------------------------------------
% ds9_getbox function                                                      ds9
% Description: Get from the ds9 display the pixel values in a specified
%              box region.
% Input  : - Box ccordinates (in 'image' coordinates):
%            [X_left_corner,  Y_left_corner,  X_width,      Y_height]
%            or
%            [X_center,       Y_center,       X_semiwidth,  Y_semiheight]
%          - Method of box position {'corner'|'center'},
%            default is 'corner'.
%            Note that in case of Method='center' and CooType='image',
%            box size will be 2*semiwidth+1.
%          - Coordinate type {'image'|'physical'|'fk4'|'fk5'|'icrs'},
%            default is 'image'.
%            THIS OPTION IS NOT SUPPORTED!
% Output : - Matrix of pixel values in box.
%          - Matrix of X position in 'image' coordinates, corresponding
%            to the matrix of pixel values.
%          - Matrix of Y position in 'image' coordinates, corresponding
%            to the matrix of pixel values.
% Reference: http://hea-www.harvard.edu/RD/ds9/ref/xpa.html
% Tested : Matlab 7.0
%     By : Eran O. Ofek                       Feb 2007
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%------------------------------------------------------------------------------
DefMethod   = 'corner';
DefCooType  = 'image';

if (nargin==1),
   Method   = DefMethod;
   CooType  = DefCooType;
elseif (nargin==2),
   CooType  = DefCooType;
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end
if (isempty(Method)==1),
   Method  = DefMethod;
end

% convert Coordinate to 'corner' meethod
switch lower(Method)
 case 'corner'
    CornerCoo = Coo;
 case 'center'
    CornerCoo = [Coo(1) - Coo(3), Coo(2) - Coo(4), 2.*Coo(3)+1, 2.*Coo(4)+1];
 otherwise
    error('Unknown Method option');
end


SizeMat = [CornerCoo(4), CornerCoo(3)];   % [SizeY, SizeX]
MatVal  = zeros(SizeMat(1),SizeMat(2));

switch CooType
    case 'image'

        [~,Res] = ds9_system(sprintf('xpaget ds9 data %s %f %f %f %f no',CooType,round(CornerCoo)));

        ResMat = sscanf(Res,'%d,%d = %f\n',[3 SizeMat(1).*SizeMat(2)]);
        ResMat = ResMat.';
        % relative position in matrix (relative to corner)
        RelX = ResMat(:,1) - min(ResMat(:,1)) + 1;
        RelY = ResMat(:,2) - min(ResMat(:,2)) + 1;

        Ind         = sub2ind(SizeMat,RelY,RelX);
        MatVal(Ind) = ResMat(:,3);
        MatVal      = MatVal; %.';

    case 'fk5'
        error('NOT SUPPORTED');
        %[~,Res] = ds9_system(sprintf('xpaget ds9 data %s %f %f %f %f no',CooType,round(CornerCoo)));
        %ResMat = sscanf(Res,'%f,%f = %f\n',[3 SizeMat(1).*SizeMat(2)]);
        %ResMat = ResMat.';
    
        
    otherwise
        error('CooType option is currently unsupported');
end

if (nargout>1),
   [MatX,MatY] = meshgrid((1:1:SizeMat(2)),(1:1:SizeMat(1)));
   MatX        = MatX + CornerCoo(1)-1;
   MatY        = MatY + CornerCoo(2)-1;
end

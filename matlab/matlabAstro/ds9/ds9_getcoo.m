function [CooX,CooY,Value,Key]=ds9_getcoo(N,CooType,Mode)
%------------------------------------------------------------------------------
% ds9_getcoo function                                                      ds9
% Description: Interactively get the coordinates (X/Y or WCS) and value
%              of the pixel selected by the mouse (left click) or by clicking
%              any character on the ds9 display.
% Input  : - Number of points to get, default is 1.
%          - Coordinate type {'fk4'|'fk5'|'icrs'|'eq'|
%                             'galactic'|'image'|'physical'},
%            default is 'image'.
%            'eq' is equivalent to 'icrs'.
%          - Operation mode:
%            'any'   - will return after any character or left click is
%                      pressed (default).
%            'key'   - will return after any character is pressed.
%            'mouse' - will return after mouse left click is pressed.
% Output : - X/RA or Galactic longitude. If celestial coordinates, the
%            return the result in radians.
%          - Y/Dec or Galactic latitude. If celestial coordinates, the
%            return the result in radians.
%          - Pixel value.
%          - Cell array of string of clicked events.
%            '<1>' for mouse left click.
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Feb 2007
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [X,Y,V]=ds9_getcoo(3,'fk5');   % return the WCS RA/Dec position
%          [X,Y,V,Key] = ds9_getcoo(1,'icrs','key');
%          [X,Y,V,Key] = ds9_getcoo(2,'icrs','mouse');
%          [X,Y,V,Key] = ds9_getcoo(1,'image','any');
% Reliable: 2
%------------------------------------------------------------------------------
RAD = 180./pi;

Def.N       = 1;
Def.CooType = 'image';
Def.Mode    = 'any';
if (nargin==0),
   N       = Def.N;
   CooType = Def.CooType;
   Mode    = Def.Mode;
elseif (nargin==1),
   CooType = Def.CooType;
   Mode    = Def.Mode;
elseif (nargin==2),
   Mode    = Def.Mode;
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end

switch lower(Mode)
 case 'mouse'
    Mode = '';
end

switch lower(CooType)
 case {'eq','equatorial'}
    CooType = 'icrs';
 otherwise
    % do nothing
end

switch lower(CooType)
 case {'fk4','fk5','icrs'}
    String = sprintf('wcs %s degrees',lower(CooType));
 case {'gal','galactic'}
    String = 'wcs galactic degrees';
 case {'image'}
    String = 'image';
 case {'physical'}
    String = 'physical';
 otherwise
    error('Unknown CooType option');
end

CooX  = zeros(N,1);
CooY  = zeros(N,1);
Value = zeros(N,1);
Key   = cell(N,1);
for I=1:1:N,
   %--- get Coordinates (from interactive mouse click) ---
   [Status,Coo] = ds9_system(sprintf('xpaget ds9 imexam %s coordinate %s',Mode,String));

   if (~isempty(Mode)),
      Ispace = findstr(Coo,' ');
      Key{I} = Coo(1:Ispace(1)-1);
      Coo = Coo(Ispace(1):end);
   else
      Key{I} = '<1>';
   end

   %--- set crosshair position to Coordinates ---
   ds9_system(sprintf('xpaset -p ds9 crosshair %s %s',Coo(1:end-1),String));
   %--- get Coordinates of crosshair ---
   [Status,CooIm] = ds9_system(sprintf('xpaget ds9 crosshair image'));
    
   %--- get Pixel value at crosshair position ---
   [Status,Val] = ds9_system(sprintf('xpaget ds9 data image %s 1 1 yes',CooIm(1:end-1)));
   
   %--- Exit crosshair mode ---
   ds9_system(sprintf('xpaset -p ds9 mode none'));
   
   
   Val          = sscanf(Val,'%f');
   CooVal       = sscanf(Coo,'%f %f');
   switch lower(CooType)
    case {'fk4','fk5','icrs','gal','galactic'}
       X            = CooVal(1)./RAD;
       Y            = CooVal(2)./RAD;
    otherwise
       X            = CooVal(1);
       Y            = CooVal(2);
   end

   CooX(I)  = X;
   CooY(I)  = Y;
   Value(I) = Val;
end

ds9_system(sprintf('xpaset -p ds9 mode pointer'));

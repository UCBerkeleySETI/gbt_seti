function [NewRA,NewDec]=interp_coo(Time,RA,Dec,NewTime,InterpMethod);
%--------------------------------------------------------------------------
% interp_coo function                                                ephem
% Description: Interpolate on celestial ccordinates as a function of time.
%              Use the built in matlab interpolation functions.
% Input  : - Vector of Times (e.g., JD).
%          - Vector of longitudes [radians].
%          - Vector of latitudes [radians].
%          - Vector of new times in which to interpolate position.
%          - Algorithm (see interp1.m for details), default is 'cubic'.
% Output : - Vector of longitudes interpolated to the to the vector
%            of new times.
%          - Vector of latitudes interpolated to the to the vector
%            of new times.
% Tested : Matlab 7.8
%     By : Eran O. Ofek                     March 2010
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 2
%--------------------------------------------------------------------------

DefInterpMethod = 'cubic';
if (nargin==4),
   InterpMethod = DefInterpMethod;
elseif (nargin==5),
   % do nothing
else
   error('Illegal number of input arguments');
end

CD = cosined([RA, Dec]);
NewCD = zeros(length(NewTime),3);

NewCD(:,1) = interp1(Time,CD(:,1),NewTime,InterpMethod);
NewCD(:,2) = interp1(Time,CD(:,2),NewTime,InterpMethod);
NewCD(:,3) = interp1(Time,CD(:,3),NewTime,InterpMethod);

NewCoo = cosined(NewCD);
NewRA  = NewCoo(:,1);
NewDec = NewCoo(:,2);

function invy;
%------------------------------------------------------------------------------
% invy function                                                       plotting
% Description: Invert the y-axis of the current axis.
%              This is a shortcut command to axis('ij') and
%              set(gca,'YDir','Inverse').
% Input  : null
% Output : null
% Tested : Matlab 4.2
%     By : Eran O. Ofek                  February 1994
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Reliable: 1
%------------------------------------------------------------------------------
axis('ij');

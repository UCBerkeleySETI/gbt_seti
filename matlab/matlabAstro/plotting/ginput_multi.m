function [X,Y]=ginput_multi
%--------------------------------------------------------------------------
% ginput_multi function                                           plotting
% Description: Get multiple ginput coordinates, for arbitrary number of
%              points, using the mouse. Use left click to select multiple
%              coordinates, and right click to abort.
% Input  : null
% Output : - List of select X coordinates.
%          - List of select Y coordinates.
% Tested : Matlab 5.3
%     By : Eran O. Ofek                    Jun 2005
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%--------------------------------------------------------------------------

Coo    = zeros(0,2);
Button = 1;

while (Button==1),
   [X,Y,Button] = ginput(1);
   if (Button==1),
      Coo   = [Coo; [X,Y]];
   end
end


X = Coo(:,1);
Y = Coo(:,2);

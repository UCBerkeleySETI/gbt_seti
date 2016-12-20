function OpenFlag=ds9_start(Wait)
%---------------------------------------------------------------------
% ds9_start function                                              ds9
% Description: Checks whether ds9 is running, and if it is not
%              running then start ds9.
% Input  : - Number of seconds to wait after starting ds9.
%            Default is 3.
% Output : - A logical flag indicating if the program opened and ds9
%            display. I.e., if there is already an open ds9 a false
%            will be returned.
% Bugs:  : The "ps -C" command doesnt work on MAC.
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
%           Linux/UNIX
% Tested : Matlab 7.13
%     By : Eran O. Ofek                    Aug 2012
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%---------------------------------------------------------------------

Def.Wait = 3;  % s
if (nargin==0),
   Wait = Def.Wait;
end

% This doesn't work on MAC
%Status=system('ps -C ds9');
Status = system('ps -A | grep ds9 | grep -v grep');
if (Status==1),
   % open ds9
   system('ds9&');
   pause(Wait);
   OpenFlag = true;
else
   % ds9 exist
   OpenFlag = false;
end





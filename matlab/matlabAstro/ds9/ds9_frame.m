function FN=ds9_frame(Frame)
%-----------------------------------------------------------------------
% ds9_frame function                                                ds9
% Description: Switch ds9 display to a given frame number.
% Input  : - Frame - This can be a function
%            {'new'|'first'|'last'|'next'|'prev'}
%            or a number, default is 'first'.
%            Create frame if doesnot exist.
% Output : - Current frame number
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Feb 2007
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: FN=ds9_frame(1);
% Reliable: 1
%-----------------------------------------------------------------------
DefFrame = 'first';
if (nargin==0),
   Frame = DefFrame;
end
if (isempty(Frame)==1),
   Frame = DefFrame;
end

%--------------------
%--- switch frame ---
%--------------------
if (isstr(Frame)==1),
   ds9_system(sprintf('xpaset -p ds9 frame %s',Frame));
else
   ds9_system(sprintf('xpaset -p ds9 frame frameno %d',Frame));
end

[Status,Res] = ds9_system(sprintf('xpaget ds9 frame frameno'));
FN = sscanf(Res,'%d');


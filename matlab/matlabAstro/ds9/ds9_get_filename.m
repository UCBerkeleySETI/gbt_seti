function Res=ds9_get_filename
%-----------------------------------------------------------------------
% ds9_get_filename function                                         ds9
% Description: Get the file name of the image which is displayed in the
%              current ds9 frame.
% Input  : null
% Output : - File name.
% Tested : Matlab 7.11
%     By : Eran O. Ofek                    Jul 2011
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 1
%-----------------------------------------------------------------------

[Stat, Res] = ds9_system('xpaget ds9 file');

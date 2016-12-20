function [Status,Coo]=ds9_system(string)
%-----------------------------------------------------------------------------
% ds9_system function                                                     ds9
% Description: Matlab sets DYLD_LIBRARY_PATH incorrectly on OSX for ds9 to
%              work. This function is a workround
% Input  : - String to pass to system
% Output : null
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
% Reference: http://hea-www.harvard.edu/RD/ds9/ref/xpa.html
% Tested : MATLAB R2011b
%     By : N. Konidaris                    Oct 2013
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%------------------------------------------------------------------------------

if ismac
   news = strcat('set DYLD_LIBRARY_PATH "";', string);
   [Status,Coo]=system(news);
else
   [Status,Coo]=system(string);
end

function [RA,Dec]=ds9_sdssnavi(Browser)
%------------------------------------------------------------------------------
% ds9_sdssnavi function                                                    ds9
% Description: Click on a position in an image displayed in ds9 and this
%              program will open the SDSS navigator web page for the
%              coordinates.
% Input  : - Broweser type:
%            'browser' - existing system browser (default).
%            'new'     - Matlab internal browser.
% Output : - J2000.0 RA of position [radians].
%          - J2000.0 Dec of position [radians].
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
% Tested : Matlab 7.11
%     By : Eran O. Ofek                    May 2011
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%------------------------------------------------------------------------------
Def.Browser = 'browser';

if (nargin==0),
   Browser = Def.Browser;
elseif (nargin==1),
   % do nothing
else
   error('Illegal number of input arguments');
end

[RA,Dec]=ds9_getcoo(1,'eq');


[LinkNavi]=get_sdss_finding_chart(RA,Dec);

web(LinkNavi{1},sprintf('-%s',Browser));

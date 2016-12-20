function FileName=ds9_save(FileName)
%--------------------------------------------------------------------------
% ds9_save function                                                    ds9
% Description: Save an image in the ds9 dispaly as a FITS image.
% Input  : - FITS image name to save.
%            If empty will generate a temporary file name.
%            Default is empty.
% Output : - Saved FITS file name.
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
% Reference: http://hea-www.harvard.edu/RD/ds9/ref/xpa.html
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: FileName=ds9_save;
% Reliable: 2
%--------------------------------------------------------------------------

if (nargin==0),
    FileName = [];
end

if (isempty(FileName)),
    FileName = tempname;
end

ds9_system(sprintf('xpaset -p ds9 save fits %s image',FileName));

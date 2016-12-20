function NewHeader=cell_fitshead_fix(Header)
%--------------------------------------------------------------------------
% cell_fitshead_fix function                                       ImBasic
% Description: Given an Nx3 cell array of FITS header. Remove blank lines
%              and make sure the END keyword is at the end of the header.
% Input  : - An Nx3 cell array of FITS header.
% Output : - A fixed header.
% Tested : Matlab R2013a
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: NewHeader=cell_fitshead_fix(Header);
% Reliable: 2
%--------------------------------------------------------------------------

FlagEmpty = strcmp(Header(:,1),'') & strcmpi(Header(:,2),'') & strcmpi(Header(:,3),'');
NewHeader = Header(~FlagEmpty,:);

% remove END
FlagEnd   = strcmp(NewHeader(:,1),'END');
NewHeader = NewHeader(~FlagEnd,:);

% add END
NewHeader = [NewHeader; {'END','',''}];

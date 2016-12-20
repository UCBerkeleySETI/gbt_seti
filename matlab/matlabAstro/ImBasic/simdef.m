function Sim=simdef(N,M)
%--------------------------------------------------------------------------
% simdef function                                                  ImBasic
% Description: Define a SIM class with multiple entries (array). 
% Input  : - Number of rows. Default is 1.
%          - Number of columns. Default is 1.
% Output : - SIM class array.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Nov 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=simdef(5);
% Reliable: 2
%--------------------------------------------------------------------------

if (nargin==0),
    N = 1;
    M = 1;
elseif (nargin==1),
    M = 1;
else
    % do nothing
end

for I=1:1:N,
    for J=1:1:M,
        Sim(I,J) = SIM;
    end
end

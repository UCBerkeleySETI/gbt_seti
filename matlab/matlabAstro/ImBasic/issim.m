function Ans=issim(Obj)
%--------------------------------------------------------------------------
% issim function                                                   ImBasic
% Description: Check if object is of SIM class.
% Input  : - Object
% Output : - {true|false}.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Oct 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: issim(S1);
% Reliable: 2
%--------------------------------------------------------------------------


Ans = isa(Obj,'SIM');

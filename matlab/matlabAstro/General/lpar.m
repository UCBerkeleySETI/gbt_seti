function Par=lpar(FunName)
%--------------------------------------------------------------------------
% lpar function                                                    General
% Description: List user default parameters for a function (see epar.m).
% Input  : - Function name.
% Output : - Structure array containing user default parameters.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: lpar sim_fft
% Reliable: 2
%--------------------------------------------------------------------------

Par = epar(FunName,'read');

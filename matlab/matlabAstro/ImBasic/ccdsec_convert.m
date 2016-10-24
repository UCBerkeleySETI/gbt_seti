function [Output]=ccdsec_convert(Input)
%--------------------------------------------------------------------------
% ccdsec_convert function                                          ImBasic
% Description: Convert CCDSEC format (e.g., '[1:100,201:301]') from string
%              to vector and vise versa.
% Input  : - A string containing a CCDSEC (e.g., '[1:100,201:301]').
%            or a vector of CCDSEC (e.g., [1 100 201 301]).
% Output : - If the input is string then will return a vector and
%            vise versa.
% Tested : Matlab R2011b
%     By : Eran O. Ofek             Aug 2013
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Vec=ccdsec_convert('[1:100,201:301]');
%          ccdsec_convert(ccdsec_convert('[1:100,201:301]'));
% Reliable: 2
%--------------------------------------------------------------------------

if (isnumeric(Input)),
   Output = sprintf('[%d:%d,%d:%d]',Input);
elseif (ischar(Input)),
   RE = regexp(Input,'[:,\[\]]','split');
   Output = zeros(1,4);
   Output(1) = str2double(RE{2});
   Output(2) = str2double(RE{3});
   Output(3) = str2double(RE{4});
   Output(4) = str2double(RE{5});
else
   error('Unknwon input option');
end

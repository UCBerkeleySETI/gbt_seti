function Bit=get_bitmask_def(InParPar,Field)
%--------------------------------------------------------------------------
% get_bitmask_def function                                         ImBasic
% Description: A database of bit mask definitions for astronomical
%              images. This define the default bit mask definitions.
% Input  : - The value of the Bit mask index. If this is a number
%            (bit index) then this number will be returned as is.
%            If a function handle than will call this function with the 
%            second input argument as a parameter, and will get back
%            the bit index.
%            For example for such a function see:
%            def_bitmask_specpipeline.m
%          - Bit name.
% Output : - Bit index.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% See also: maskflag_check.m, maskflag_set.m
% Example: Bit=get_bitmask_def(@def_bitmask_specpipeline,'Bit_ImSaturated');
% Reliable: 2
%--------------------------------------------------------------------------

if (isnumeric(InParPar)),
    Bit = InParPar;
elseif (isa(InParPar,'function_handle') || ischar(InParPar)),
    % look for field name in DB
    Bit = feval(InParPar,Field);
end


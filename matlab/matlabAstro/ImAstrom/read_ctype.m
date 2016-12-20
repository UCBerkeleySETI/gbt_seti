function Out=read_ctype(String,Fields)
%--------------------------------------------------------------------------
% read_ctype function                                             ImAstrom
% Description: Given a FITS header CTYPE keyword value (e.g., 'RA---TAN')
%              return the coordinate type (e.g., RA) and transformation
%              type (e.g., 'TAN').
% Input  : - A string containing a CTYPE header keyword value.
%            Alternatively, this can be a structure array which contains
%            a CTYPE1 or a CTYPE2 fields (or other specified fields).
%          - In case the first input argument is a structure then this
%            is a cell array of field names to extract.
%            Default is {'CTYPE1','CTYPE2'}.
% Output : - If the first input is a tring then this is a structure
%            containing the fields .Coo and .Tran. and optionally .Dist
%            If the first input is a structure array then this is
%            a structure array containing field names (e.g., 'CTYPE1')
%            which bythemselfs containing the fields .Coo and .Tran.
%            and optionally .Dist
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Nov 2013
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Out=read_ctype('RA---TAN');
%          WCS=get_fits_wcs('PTF201001131475_2_o_14235_00.w.fits');
%          Out=read_ctype(WCS);
% Reliable: 2
%--------------------------------------------------------------------------

Def.Fields = {'CTYPE1','CTYPE2'};
if (nargin==1),
    Fields  = Def.Fields;
end
if (~iscell(Fields)),
    Fields = {Fields};
end

if (ischar(String)),
    Split    = regexp(String,'-','split');
    Pair     = Split(~isempty_cell(Split));
    Out.Coo  = Pair{1};
    Out.Tran = Pair{2};
    if (numel(Pair)>2),
        Out.Dist = Pair{3};
    end
elseif (isstruct(String)),
    N = numel(String);
    for I=1:1:N,
        for If=1:1:numel(Fields),
            Split    = regexp(String.(Fields{If}),'-','split');
            Pair     = Split(~isempty_cell(Split));
            Out(I).(Fields{If}).Coo  = Pair{1};
            Out(I).(Fields{If}).Tran = Pair{2};
            if (numel(Pair)>2),
                Out(I).(Fields{If}).Dist = Pair{3};
            end
        end
    end
else
    error('Unknown String type');
end

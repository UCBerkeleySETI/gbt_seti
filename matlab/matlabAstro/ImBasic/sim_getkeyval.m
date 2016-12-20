function [Val,Struct]=sim_getkeyval(Sim,Key,varargin)
%--------------------------------------------------------------------------
% sim_getkeyval function                                           ImBasic
% Description: Give a structure array of images and their headers,
%              get the values of a specific header keyword.
%              See sim_getkeyvals.m for a multiple keyword version.
% Input  : - Structure array of images. See images2sim.m for options.
%          - A single keyword name.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'HeaderField' - Header field name in the structure array.
%                            Default is 'Header'.
%            'ConvNum'     - Force conversion of keyword value to a number
%                            {true|false}. Default is false.
%            'NanIfEmpty'  - If true then will return NaN if header not
%                            found, otherwise will return an empty value.
%                            Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - Cell array of keyword values.
%          - Structure array of keyword values.
% See also: sim_getkeyvals.m
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Val,Struct]=sim_getkeyval(Sim,'NAXIS1')
% Reliable: 2
%-----------------------------------------------------------------------------


%ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';


DefV.HeaderField = HeaderField;
DefV.ConvNum     = false;
DefV.NanIfEmpty  = true;
InPar = set_varargin_keyval(DefV,'m','use',varargin{:});

HeaderField = InPar.HeaderField;


Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);
Val = cell(Nim,1);
for Iim=1:1:Nim,
    if (isfield_notempty(Sim(Iim),HeaderField)),
        CellHead = cell_fitshead_getkey(Sim(Iim).(HeaderField),Key,'NaN');
        Val{Iim} = CellHead{1,2};
        if (InPar.ConvNum),
            Val{Iim} = str2num_nan(Val);
        end
    else
        if (InPar.NanIfEmpty),
            Val{Iim} = NaN;
        else
            Val{Iim} = [];
        end
    end
end

if (nargout>1),
    Struct = cell2struct(Val,Key,2);
end
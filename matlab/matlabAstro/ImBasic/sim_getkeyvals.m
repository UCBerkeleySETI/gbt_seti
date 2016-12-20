function [Val,Struct]=sim_getkeyvals(Sim,Keys,varargin)
%--------------------------------------------------------------------------
% sim_getkeyvals function                                          ImBasic
% Description: Give a structure array of images and their headers,
%              get the values of a specific header keywords.
% Input  : - Structure array of images. See images2sim.m for options.
%          - A cell array of keywords.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'HeaderField' - Header field name in the structure array.
%                            Default is 'Header'.
%            'ConvNum'     - Force conversion of keyword value to a number
%                            {true|false}. Default is false.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - Cell array of of cell array of keyword values.
%          - Structure array of keyword values.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Val,Struct]=sim_getkeyvals(Sim,{'NAXIS1','NAXIS2'})
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

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (~iscell(Keys)),
    Keys = {Keys};
end

HeaderField = InPar.HeaderField;


Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);
Nkey = numel(Keys);
Val = cell(Nim,1);
for Iim=1:1:Nim,
    if (isfield_notempty(Sim(Iim),HeaderField)),
        
        CellHead = cell_fitshead_getkey(Sim(Iim).(HeaderField),Keys,'NaN');
        Val{Iim} = CellHead(:,2);
        if (InPar.ConvNum),
            %Val{Iim} = num2cell(str2num_nan(Val{Iim}));
            Val{Iim} = num2cell(str2double_check(Val{Iim}));
        end

        if (nargout>1),
            Struct(Iim) = cell2struct(Val{Iim}.',Keys,2);
        end
    else
        Val{Iim} = num2cell(nan(1,Nkey));
        if (nargout>1),
            Struct(Iim) = cell2struct(Val{Iim}.',Keys,2);
        end
    end
    
end

function IsKeyVal=is_head_keyval(Header,Im2SimPar,varargin)
%--------------------------------------------------------------------------
% is_head_keyval function                                          ImBasic
% Description: Given image headers in a 3 column cell array format,
%              go over a list of keywords and check if their values
%              are equal to some specific strings or numbers.
% Input  : - This can be either an image header in a 3 column cell
%            array format, or any of the format acceptable by
%            images2sim.m
%          - Cell array of parameters to pass to images2sim.m.
%            default is {}.
%          * Abitrary number of pairs of ...,key,val,... arguments.
%            The keywords are header keywords strings, while the values
%            are either strings, cell array of strings, numbers or cell
%            array of numbers, which are compared to the the value of
%            the header keyword.
% Output :- A matrix in which the number of rows is equal to the number
%           of image headers while the number of columns is equal to
%           the number of input keywords.
%           For each image and input key,val pair, a single element is
%           returned which indicates if the value in the header is equal
%           to one of the values in the corresponding value argument.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example:
% IsKeyVal=is_head_keyval('lred*.fits',{},'SLITGRAB','deployed','DICHTRAN','deployed')
% Reliable: 2
%--------------------------------------------------------------------------


%ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

if (nargin==1),
    Im2SimPar = {};
end

Size=size(Header);
if (Size(2)==3),
    Sim.(HeaderField) = Header;
else
    Sim = images2sim(Header,Im2SimPar{:});
end
Nim = length(Sim);
    
Narg = length(varargin);

Keys = varargin(1:2:end-1);
IsKeyVal = false(Nim,Narg.*0.5);

for Iim=1:1:Nim,
    NewCellHead = cell_fitshead_getkey(Sim(Iim).(HeaderField),Keys,'NaN');

    Ind  = 0;
    for Iarg=1:2:Narg-1,
        Ind = Ind + 1;
        if (ischar(NewCellHead{Ind,2})),
            IsKeyVal(Iim,Ind) = any(strcmpi(spacedel(NewCellHead{Ind,2}),varargin{Iarg+1}));
        else
            IsKeyVal(Iim,Ind) = any(NewCellHead{Ind,2}==[varargin{Iarg+1}]);
        end
    end

end
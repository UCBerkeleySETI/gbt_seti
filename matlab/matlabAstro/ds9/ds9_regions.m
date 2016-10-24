function ds9_regions(Option,RegFileName)
%------------------------------------------------------------------------------
% ds9_regions function                                                     ds9
% Description: Load, save or delete ds9 regions file.
% Input  : - Option: load {'l'|'load'}, or save {'s'|'save'},
%            or delete {'d'|'delete'},
%            default is 's'.
%            For the 'delete' option no region file name is needed.
%          - Region file name, degault is 'ds9.reg'.
% Output : null
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Feb 2007
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: ds9_regions('s','try.reg')
% Reliable: 2
%------------------------------------------------------------------------------
DefOption       = 's';
DefRegFileName  = 'ds9.reg';

if (nargin==0),
   Option       = DefOption;
   RegFileName  = DefRegFileName;
elseif (nargin==1),
   RegFileName  = DefRegFileName;
elseif (nargin==2),
   % do nothing
else
   error('Illegal number of input arguments');
end

if (isempty(Option)==1),
   Option  = DefOption;
end


%------------------------
%--- Load/Save/Delete ---
%------------------------
switch lower(Option)
 case {'s','save'}
    ds9_system(sprintf('xpaset -p ds9 regions save %s',RegFileName));
 case {'l','load'}
    ds9_system(sprintf('xpaset -p ds9 regions load %s',RegFileName));
 case {'d','delete'}
    ds9_system(sprintf('xpaset -p ds9 regions delete all'));
 otherwise
    error('Unknown Option');
end

ds9_system(sprintf('xpaset -p ds9 mode pointer'));


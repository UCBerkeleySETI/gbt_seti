function CellHead=cell_fitshead_update(CellHead,varargin)
%--------------------------------------------------------------------------
% cell_fitshead_update function                                    ImBasic
% Description: Update keywords in fits header cell.
% Input  : - Cell array containing the FITS header information.
%          * Arbitrary number of arguments:
%            ...,{key,val,comment},{key,val},...
%            ...,key,val,comment,key,val,comment,...
% Output : - New cell arry of fits header.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: CellHead=cell_fitshead_update(CellHead,'EXPTIME',60,'');
%          CellHead=cell_fitshead_update(CellHead,{'EXPTIME',60,'Exposure time'});
% Reliable: 2
%--------------------------------------------------------------------------


Narg = numel(varargin);
if (Narg>0),
    if (iscell(varargin{1})),
        for Iarg=1:1:Narg,
            CellHead = update_keyword1(CellHead,varargin{Iarg}{:});
        end
    else
       
        for Iarg=1:3:Narg-2,
            CellHead = update_keyword1(CellHead,varargin{Iarg:Iarg+2});
        end
    end
end
CellHead = cell_fitshead_fix(CellHead);

function CellHead=update_keyword1(CellHead,Key,Val,Comment)
Nkey = size(CellHead,1);
if (nargin==3),
    Comment = '';
end
Ikey = find(strcmp(CellHead(:,1),Key));
if (isempty(Ikey)),
    % key doesn't exist - add
    % add key val at the end of the 
    CellHead(end+1,:) = {Key,Val,Comment};
else
    % update
    CellHead(Ikey(1),:) = {Key,Val,Comment};
    if (numel(Ikey)>1),
        CellHead = CellHead(setdiff((1:1:Nkey),Ikey(2:end)),:);
    end
end

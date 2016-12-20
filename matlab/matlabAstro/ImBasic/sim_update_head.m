function Sim1=sim_update_head(Sim,varargin)
%--------------------------------------------------------------------------
% sim_update_head function                                         ImBasic
% Description: 
% Input  : - A single structure array image in which to update the
%            header field.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'Comments' - A cell array of comments to add to header.
%            'History'  - A cell array of history comments to add to
%                         header.
%            'CopyHead' - Copy header from original image {'y' | 'n'}.
%                         Default is 'y'.
%            'AddHead'  - Cell array with 3 columns containing additional
%                         keywords to be add to the header.
%                         See cell_fitshead_addkey.m for header structure
%                         information. Default is empty matrix.
%            'DelDataSec' - Delete the 'DATASEC' header keyword
%                         {true|false}. Default is false.
% Output : - The original structure with an updated header.
% Tested : Matlab R2013a
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=sim_update_head(Sim(1),'Comments',{'A new comment'})
% Reliable: 2
%--------------------------------------------------------------------------


%ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';


DefV.CopyHead   = 'y';
DefV.Comments   = {};
DefV.History    = {};
DefV.AddHead    = {};
DefV.DelDataSec = false;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (~isfield(Sim,HeaderField)),
    HeaderInfo    = [];
    InPar.CopyHead = 'n';
else
    HeaderInfo = Sim.(HeaderField);
end

%--- Copy header ---
switch lower(InPar.CopyHead)
   case 'y'
        % do nothing - HeaderInfo = Sim.(HeaderField);
   case 'n'
       HeaderInfo = [];
   otherwise
       error('Unknown CopyHead option');
end
%--- Add to header comments regarding file creation ---
HeaderInfo = cell_fitshead_addcomment(HeaderInfo,'COMMENT',InPar.Comments);
HeaderInfo = cell_fitshead_addcomment(HeaderInfo,'HISTORY',InPar.History);
                                    
if (~isempty(InPar.AddHead)),
    %--- Add additional header keywords ---
    HeaderInfo = [HeaderInfo; InPar.AddHead];
end

if (InPar.DelDataSec),
    HeaderInfo = cell_fitshead_delkey(HeaderInfo,'DATASEC');
end

% fix header
HeaderInfo = cell_fitshead_fix(HeaderInfo);

Sim1 = Sim;
Sim1.(HeaderField) = HeaderInfo;


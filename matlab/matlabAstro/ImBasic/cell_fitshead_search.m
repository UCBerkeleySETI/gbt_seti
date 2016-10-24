function SubHead=cell_fitshead_search(Head,String,varargin)
%--------------------------------------------------------------------------
% cell_fitshead_search function                                    ImBasic
% Description: Search for substring in FITS header stored as a cell
%              array of 3 columns. Return and display all instances of
%              the substring.
% Input  : - Cell array containing 3 columns FITS header
%            {key, val, comment}.
%          - Sub string to search, or a regular expression patern to match.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'CaseSens' - case sensative {false|true}. Default is false.
%            'Col'      - Which column in the header to search:
%                         1 - keywords column.
%                         3 - comment column.
%                         You can specificy one or many options
%                         (e.g., [1 3] or 1). Default is [1 3].
%            'Verbose'  - Show search result {true|false}. Default is true.
% Output : - Cell array containing the lines which contain the requested
%            sub string.
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Apr 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: SubHead=cell_fitshead_search(Sim(1).Header,'FWHM');
% Reliable: 2
%--------------------------------------------------------------------------

DefV.CaseSens          = false;
DefV.Col               = [1 3];
DefV.Verbose           = true;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (InPar.CaseSens),
    FindFun = @regexp;
else
    FindFun = @regexpi;
end

Ncol = numel(InPar.Col);
AllLines = [];
for Icol=1:1:Ncol,
    ColInd = InPar.Col(Icol);
    AllLines = [AllLines; find(~isempty_cell(FindFun(Head(:,ColInd),String,'match')))];
end
AllLines = unique(AllLines);
SubHead = Head(AllLines,:);

if (InPar.Verbose),
    Nl = numel(AllLines);
    if (Nl>0),
        fprintf('\n');
        fprintf('Keyword      Value                   Comment\n');
        fprintf('----------   --------------------    ----------------------------------------\n');
    else
        fprintf('\n Substring not found \n');
    end
    for Il=1:1:Nl,
        if (isnumeric(SubHead{Il,2})),
            fprintf('%-10s   %20.9f    %-40s\n',SubHead{Il,1},SubHead{Il,2},SubHead{Il,3});
        else
            fprintf('%-10s   %-20s    %-40s\n',SubHead{Il,1},SubHead{Il,2},SubHead{Il,3});
        end
    end
end

    
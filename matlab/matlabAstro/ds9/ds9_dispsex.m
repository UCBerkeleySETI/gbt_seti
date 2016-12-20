function [Cat,Col]=ds9_dispsex(Cat,ColXY,SexPar,DispPar)
%------------------------------------------------------------------------------
% ds9_dispsex function                                                     ds9
% Description: Display markers around a list of sources on an image displayed
%              using ds9.
% Input  : - FITS image name on which to run SExtractor, or alternatively
%            A matrix returned by run_sextractor.m
%            or a FITS binary table containing the SExtractor catalog.
%          - Structure containing the columns description.
%            Alternatively, a two element vector containing the indices
%            of the columns corresponding to the X and Y position.
%            If empty matrix than use default. Default is [1 2].
%          - A cell array of additional parameters to pass to SExtractor.
%            Default is {}.
%          - A cell array of additional parameters to pass to ds9_plotregion.m
%            Default is {'Load','y','Coo','image','Type','circle','Color','red','Width',2,'Size',20}
% Output : Null
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
% Tested : Matlab 7.11
%     By : Eran O. Ofek                    Apr 2011
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 1
%------------------------------------------------------------------------------

Def.ColXY   = [1 2];
Def.SexPar  = {};
Def.DispPar = {'Load','y','Coo','image','Type','circle','Color','red','Width',2,'Size',20};
if (nargin==1),
    ColXY   = Def.ColXY;
    SexPar  = Def.SexPar;
    DispPar = Def.DispPar; 
elseif (nargin==2),
    SexPar  = Def.SexPar;
    DispPar = Def.DispPar; 
elseif (nargin==3),
    DispPar = Def.DispPar; 
elseif (nargin==4),
    % do nothing
else
    error('Illegal number of input arguments');
end

if (isempty(ColXY)),
    ColXY = Def.ColXY;
end


if (isnumeric(ColXY)),
    Col.XWIN_IMAGE = ColXY(1);
    Col.YWIN_IMAGE = ColXY(2);
end

if (isstr(Cat)),
    % cat is either a FITS image or a FITS binary table
    InfoHead = fitsinfo(Cat);
    if (sum(~isempty_cell(strfind(InfoHead.Contents,'Binary Table')))>0),
        % FITS binary table
        [~,~,ColCell]=get_fitstable_col(Cat,'BinTable');
        Col = cell2struct(num2cell(1:1:length(ColCell)),ColCell,2)
        Cat = cell2mat(fitsread(Cat,'BinTable'));
        
    elseif (sum(~isempty_cell(strfind(InfoHead.Contents,'Table')))>0),
        % FITS table
        [~,~,ColCell]=get_fitstable_col(Cat,'Table');
        Col = cell2struct(num2cell(1:1:length(ColCell)),ColCell,2)
        Cat = cell2mat(fitsread(Cat,'Table'));
        
    else
        % asume the input is a FITS image
	[CatS,~,Col] = run_sextractor(Cat,SexPar{:});
        Cat = CatS{1};
    end
    
end


% display extracted sources in ds9:
X = Cat(:,Col.XWIN_IMAGE);
Y = Cat(:,Col.YWIN_IMAGE);

ds9_plotregion(X,Y,DispPar{:});


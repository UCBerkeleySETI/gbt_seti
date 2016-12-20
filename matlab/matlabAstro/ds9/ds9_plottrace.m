function FileName=ds9_plottrace(Trace,varargin)
%------------------------------------------------------------------------------
% ds9_plottrace function                                                   ds9
% Description: Given a list of X and Y plot Y(X) on the ds9 display.
% Input  : - Two column matrix [X,Y].
%          * Arbitrary number of pairs of input arguments:...,keyword,value,...
%            Following keywords are available:
%            'FileName'   - Region file name, default is to create a temp file.
%            'Save'       - save region file {'y' | 'n'}, default is 'n'.
%            'Load'       - Load region file into ds9 display {'y'|'n'},
%                           default is 'y'.
%            'Append'     - Append to an existing region file {'y'|'n'},
%                           default is 'n'.
%                           If 'y' then will not write the region file header.
%            'Coo'        - Coordinate type {'image'|'fk5'},
%                           default is 'image'.
%            'Color'      - Symbol color, default is 'red'.
%            'Width'      - Marker line width, default is 1.
%            'Dist'       - If This parameter is >0, then the program will
%                           plot two lines at distance indicated by this
%                           parameter above and below the trace position.
%                           Where the 'DispAxis' parameter specify if the
%                           distance is measured relative to the X or Y
%                           axis. Default is 0.
%            'DispAxis'   - A parameter specifing the dispersion axis
%                           {'x'|'y'}. Default is 'x'.
%                           If dispersion axis is 'x' then the 'Dist'
%                           parameter is applied to the Y axis.
% Output : - Region file name.
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
% Tested : Matlab 7.11
%     By : Eran O. Ofek                    Feb 2011
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Reliable: 2
%------------------------------------------------------------------------------

DefV.FileName = tempname;
DefV.Save     = 'y';
DefV.Load     = 'y';
DefV.Append   = 'n';
DefV.Coo      = 'image';
DefV.Color    = 'red';
DefV.Width    = 1;
DefV.Dist     = 0;
DefV.DispAxis = 'x';

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

Type    = 'line';
Text    = '';
LinePar = [Trace(1:end-1,1),Trace(1:end-1,2),Trace(2:end,1),Trace(2:end,2)];


if (InPar.Dist==0),
   % do nothing
else
   switch lower(InPar.DispAxis)
    case 'x'
       LinePar = [[LinePar(:,1), LinePar(:,2) - InPar.Dist, LinePar(:,3), LinePar(:,4) - InPar.Dist]; [LinePar(:,1), LinePar(:,2) + InPar.Dist, LinePar(:,3), LinePar(:,4) + InPar.Dist]];

    case 'y'
       LinePar = [[LinePar(:,1) - InPar.Dist, LinePar(:,2), LinePar(:,3) - InPar.Dist, LinePar(:,4)]; [LinePar(:,1) + InPar.Dist, LinePar(:,2), LinePar(:,3) + InPar.Dist, LinePar(:,4)]];
    otherwise
       error('Unknown DispAxis option');
   end
end

RegPar = {'Save',InPar.Save,...
          'Load',InPar.Load,...
          'Append',InPar.Append,...
          'Coo',InPar.Coo,...
          'Color',InPar.Color,...
          'Width',InPar.Width,...
          'Text',Text,...
          'Type',Type,...
          'Size',LinePar};


FileName = ds9_plotregion([],[],RegPar{:});

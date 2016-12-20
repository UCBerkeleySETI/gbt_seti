function [LinkNavi,Image,Link]=get_sdss_finding_chart(RA,Dec,varargin);
%------------------------------------------------------------------------------
% get_sdss_finding_chart function                                         sdss
% Description: Get a link, JPG (and read it into matlab) of an SDSS finding
%              chart directly from the SDSS sky server (Version: SDSS-DR8).
% Input  : - List of RA [H M S] or [radians] or sexagesimal string.
%          - List of Dec [Sign D M S] or [radians] or sexagesimal string..
%          * Arbitrary pairs of arguments: keyword, value,...
%            'Scale'   - scale in arcse/pix, default is 0.4.
%            'Width'   - Image width in pixels, default is 600.
%            'Label'   - Show label {'y' | 'n'}, default is 'y'.
%            'Grid'    - Show Grid {'y' | 'n'}, default is 'y'.
%            'Invert'  - Invert image color {'y' | 'n'}, default is 'n'.
%            'Save'    - File (base only) name in which to save image in.
%                        If more then one image is retrieved
%                        then a serial number is appended.
%                        An extension '.jpg' is added to each file name.
%                        Default is [] - not to save a jpg copy.
% Output : - Cell array of links to navigation tool to each finding chart.
%          - Cell array of images matrix.
%          - Cell array of links to each finding chart.
% Tested : Matlab 5.3
%     By : Dovi Poznanski / Eran O. Ofek     June 2005
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: [Link,Image,LN]=get_sdss_finding_chart([10 0 0],[+1 0 0 0],'Save','Try');
% Reliable: 2
%------------------------------------------------------------------------------
RAD = 180./pi;
%FindingChartURL   = 'http://cas.sdss.org/dr7/en/tools/chart/chart.asp';
FindingChartURL = 'http://skyserver.sdss3.org/dr8/en/tools/chart/chart.asp';
%FindingChartNaviURL = 'http://cas.sdss.org/dr7/en/tools/chart/navi.asp';
%FindingChartNaviURL = 'http://skyserver.sdss3.org/dr9/en/tools/chart/navi.asp';
FindingChartNaviURL = 'http://skyserver.sdss.org/dr12/en/tools/chart/navi.aspx';
%FindingChartImURL = 'http://casjobs.sdss.org/ImgCutoutDR7/getjpeg.aspx';
FindingChartImURL = 'http://skyserver.sdss.org/dr12/en/tools/chart/image.aspx';
Apos            = '''';

RA  = convertdms(RA,'gH','r');
RA  = RA.*RAD;
Dec = convertdms(Dec,'gD','R');
Dec = Dec.*RAD;


%--- Default values ---
DefV.Scale       = 0.4;
DefV.Width       = 600;
DefV.Label       = 'y';
DefV.Grid        = 'y';
DefV.Invert      = 'n';
DefV.Save        = [];
InPar = set_varargin_keyval(DefV,'y','use',varargin{:});


%--- Options ---
Options = [];
switch lower(InPar.Label)
 case 'y'
    Options = [Options, 'L'];
 case 'n'
    % do nothing
 otherwise
    error('Unknown Label Option');
end

switch lower(InPar.Grid)
 case 'y'
    Options = [Options, 'G'];
 case 'n'
    % do nothing
 otherwise
    error('Unknown Grid Option');
end

switch lower(InPar.Invert)
 case 'y'
    Options = [Options, 'I'];
 case 'n'
    % do nothing
 otherwise
    error('Unknown Invert Option');
end


Nim = length(RA);

for Iim=1:1:Nim,

   %--- Manual link with control ---
   FullURL = sprintf('%s?ra=%f&dec=%f&opt=%s&scale=%f&width=%f&height=%f',FindingChartURL,RA(Iim),Dec(Iim),Options,InPar.Scale,InPar.Width,InPar.Width);
   FullNaviURL = sprintf('%s?ra=%f&dec=%f&opt=%s&scale=%f&width=%f&height=%f',FindingChartNaviURL,RA(Iim),Dec(Iim),Options,InPar.Scale,InPar.Width,InPar.Width);
   Link{Iim} = FullURL;
   LinkNavi{Iim} = FullNaviURL;

   %--- Image source URL ---
   ImURL   = sprintf('%s?ra=%f&dec=%f&opt=%s&scale=%f&width=%d&height=%d',FindingChartImURL,RA(Iim),Dec(Iim),Options,InPar.Scale,InPar.Width,InPar.Width);


   if (nargout>1),
      %--- wget images ---
      % sprintf('wget -q %c%s%c -O Try.jpg',Apos,ImURL,Apos)

      if (isempty(InPar.Save)==1),
         %--- don't save image ---
         ImageName = 'Tmp_SDSS_DR6_Image';
      else
         if (Nim==1),
            ImIndex = '';
         else
            ImIndex = sprintf('%d',Iim);
         end
         ImageName = sprintf('%s%s',InPar.Save,ImIndex);
      end
      ImageName = sprintf('%s.jpg',ImageName);

      eval(sprintf('!wget -q %c%s%c -O %s',Apos,ImURL,Apos,ImageName));

      CurIm      = imread(ImageName,'jpg');
      Image{Iim} =CurIm;

   end

end




function ds9_slit(RA,Dec,varargin)
%--------------------------------------------------------------------------
% ds9_slit function                                                    ds9
% Description: Download an image from a ds9 server and plot a slit in
%              the ds9 display.
%              profile in the image along the slit.
% Input  : - J2000 RA in radians, [H M S] or sexagesimal string.
%          - J2000 Dec in radians, [sign D M S] or sexagesimal string.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            Keywords and values can be one of the followings:
%            'PA'       - Slit Position Angle to plot [deg]. Default is 0.
%            'Width'    - Slit width [arcsec]. Default is 10.
%            'Length'   - Slit length [arcsec]. Default is 120.
%            'Plot'     - Plot intensity along the slit {true|false}.
%                         Default is true.
%            'StartDS9' - Open ds9 if not open {true|false}.
%                         Default is false.
%            'ImSize'   - Image size [width height]. Default is [5 5].
%            'SizeUnits'- Units of image size. Default is 'arcmin'.
%            'Frame'    - Frame into which to load the image.
%                         Default is 'new'.
%            'Save'     - Save loaded image {'yes'|'no'}. Default is 'no'.
%            'Survey'   - One of the following sky survey and image server
%                         name:
%                         {'2mass'|'dssstsci'|'dsseso'|'nvss'|'first'|'skyview'}
%                         Default is 'dssstsci'.
%            'Filter'   - Image filter name.
%                         {'j'|'h'|'k'} for Survey '2mass'.
%                         {'poss2ukstu_red'|'poss2ukstu_ir'|'poss2ukstu_blue'| 
%                          'poss1_blue|poss1_red'|'all'|'quickv'|
%                          'phase2_gsc2'|'phase2_gsc1'} for 'dssstsci'.
%                         {'DSS1'|'DSS2-red'|'DSS2-blue'|'DSS2-infrared'}
%                          for dsseso.
%                         {'sdssi'|'sdssr'|'sdssg'|'sdssu'|'sdssg'} for
%                         skyview.{'j'|'h'|'k'} for Survey '2mass'.
%                         {'poss2ukstu_red'|'poss2ukstu_ir'|'poss2ukstu_blue'| 
%                          'poss1_blue|poss1_red'|'all'|'quickv'|
%                          'phase2_gsc2'|'phase2_gsc1'} for 'dssstsci'.
%                         {'DSS1'|'DSS2-red'|'DSS2-blue'|'DSS2-infrared'}
%                          for dsseso.
%                         {'sdssi'|'sdssr'|'sdssg'|'sdssu'|'sdssg'} for
%                         skyview.
%                         Default is 'poss2ukstu_red'
%            'Scale'       - Scale function {'linear'|'log'|'pow'|'sqrt'
%                                            'squared'|'histequ'}
%                            default is 'linear'.
%            'ScaleLimits' - Scale limits [Low High], no default.
%            'ScaleMode'   - Scale mode {percentile |'minmax'|'zscale'|'zmax'},
%                            default is 'zscale'.
%            'CMap'        - Color map {'Grey'|'Red'|'Green'|'Blue'|'A'|'B'|
%                                       'BB'|'HE'|'I8'|'AIPS0'|'SLS'|'HSV'|
%                                       'Heat'|'Cool'|'Rainbow'|'Standard'|
%                                       'Staircase'|Color'},
%                            default is 'Grey'.
%             'InvertCM'   - Invert colormap {'yes'|'no'}, default is 'no'.
%             'Rotate'     - Image rotation [deg], default is to 0 deg.
%             'Orient'     - Image orientation (flip) {'none'|'x'|'y'|'xy'},
%                            default is 'none'.
%                            Note that Orient is applayed after rotation.
%             'Zoom'       - Zoom to 'fit' or [XY] or [X Y] zoom values,
%                            default is [1] (i.e., [1 1]).
% Output : null
% Required: XPA - http://hea-www.harvard.edu/saord/xpa/
% Reference: http://hea-www.harvard.edu/RD/ds9/ref/xpa.html
% Tested : Matlab 7.0
%     By : Eran O. Ofek                    Mar 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: ds9_slit(149.99871./RAD,40.98472./RAD,'survey','skyview','filter','sdssi','PA',30)
% Reliable: 2
%--------------------------------------------------------------------------
RAD = 180./pi;

DefV.PA           = 0;
DefV.Width        = 10;  % [arcsec]
DefV.Length       = 120; % [arcsec]
DefV.Plot         = true;
% ds9_imserver.m
DefV.StartDS9     = false;
DefV.ImSize       = [5 5];  % arcmin
DefV.SizeUnits    = 'arcmin';
DefV.Frame        = 'new';
DefV.Save         = 'no';
DefV.Survey       = 'dssstsci';
DefV.Filter       = 'poss2ukstu_red';
% display
DefV.Scale        = 'linear';
DefV.ScaleLimits  = [];
DefV.ScaleMode    = 'zscale';
DefV.CMap         = 'Grey';
DefV.InvertCM     = 'no';
DefV.Rotate       = 0;
DefV.Orient       = 'none';
DefV.Zoom         = 1;

InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

ds9_imserver(RA,Dec,varargin{:});
pause(3);

RA   = convertdms(RA,'gH','d');   % deg
Dec  = convertdms(Dec,'gD','d');  % deg


[Dec1,RA1]=reckon(Dec,RA,0.5.*InPar.Width./3600,InPar.PA+90,'degrees');
[Dec2,RA2]=reckon(Dec,RA,0.5.*InPar.Width./3600,InPar.PA-90,'degrees');
[Dec1s,RA1s]=reckon(Dec1,RA1,0.5.*InPar.Length./3600,InPar.PA,'degrees');
[Dec1e,RA1e]=reckon(Dec1,RA1,0.5.*InPar.Length./3600,InPar.PA+180,'degrees');
[Dec2s,RA2s]=reckon(Dec2,RA2,0.5.*InPar.Length./3600,InPar.PA,'degrees');
[Dec2e,RA2e]=reckon(Dec2,RA2,0.5.*InPar.Length./3600,InPar.PA+180,'degrees');


ds9_plotregion(RA,Dec,'coo','fk5','type','line','size',[RA1s, Dec1s, RA1e, Dec1e]);
ds9_plotregion(RA,Dec,'coo','fk5','type','line','size',[RA2s, Dec2s, RA2e, Dec2e]);

%[MatVal,MatX,MatY]=ds9_getbox(Coo,Method,CooType);
if (InPar.Plot),
    FileName=ds9_save;
    Image = fitsread(FileName);
    WCS=get_fits_wcs(FileName);
    PixScale = abs(WCS.CDELT1).*3600; % [arcsec/pix]


    RA_s  = mean([RA1s,RA2s])./RAD;
    Dec_s = mean([Dec1s,Dec2s])./RAD;
    RA_e  = mean([RA1e,RA2e])./RAD;
    Dec_e = mean([Dec1e,Dec2e])./RAD;


    [X,Y]=sky2xy_tan(FileName,[RA_s;RA_e],[Dec_s;Dec_e]);
    [Prof,VecR,VecX,VecY]=lineprof(Image,[X(1), Y(1)],[X(2), Y(2)]);

    plot(VecR.*PixScale,Prof,'k-');
    H=xlabel('Coordinates along the slit [arcsec]'); set(H,'FontSize',18);
    H=ylabel('Intensity'); set(H,'FontSize',18);

    delete(FileName);
end
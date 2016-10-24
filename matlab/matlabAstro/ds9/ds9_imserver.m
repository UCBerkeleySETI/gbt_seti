function ds9_imserver(RA,Dec,varargin)
%--------------------------------------------------------------------------
% ds9_imserver function                                                ds9
% Description: load an image from one of the ds9 image servers into the
%              ds9 display.
% Input  : - J2000 RA in radians, [H M S] or sexagesimal string.
%            Alternatively, if the second input argument is empty,
%            then this parameter is assumed to be an object name 
%            (e.g., 'm31');.
%          - J2000 Dec in radians, [sign D M S] or sexagesimal string.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            Keywords and values can be one of the followings:
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
% Example: ds9_imserver('10:00:00','+41:00:00','survey','skyview','filter','sdssi')
%          ds9_imserver('m51',[]);
% Reliable: 2
%--------------------------------------------------------------------------

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

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


if (InPar.StartDS9)
   ds9_start;
end

switch lower(InPar.Survey)
    case {'nvss','dsssao','first'}
        InPar.Filter = '';
    otherwise
        % do nothing
end


if (isempty(Dec)),
    % assume RA is object name
    Name = RA;
else
    RA   = convertdms(RA,'gH','d');   % deg
    Dec  = convertdms(Dec,'gD','d');  % deg
    Name = [];
end
        if (isempty(Name)),
            ds9_system(sprintf('xpaset -p ds9 %s coord %9.5f %9.5f degrees',InPar.Survey,RA,Dec));
        else
            ds9_system(sprintf('xpaset -p ds9 %s name %s',InPar.Survey,Name));
        end
        ds9_system(sprintf('xpaset -p ds9 %s size %d %d %s',InPar.Survey,InPar.ImSize(1),InPar.ImSize(2),InPar.SizeUnits));
        ds9_system(sprintf('xpaset -p ds9 %s save %s',InPar.Survey,InPar.Save));
        ds9_system(sprintf('xpaset -p ds9 %s frame %s',InPar.Survey,InPar.Frame));
        %ds9_system(sprintf('xpaset -p ds9 %s update frame',InPar.Survey));
        if (~isempty(InPar.Filter)),
            ds9_system(sprintf('xpaset -p ds9 %s survey %s',InPar.Survey,InPar.Filter));
        end
        ds9_system(sprintf('xpaset -p ds9 %s open',InPar.Survey));
        %eval(sprintf('xpaset -p ds9 2mass close'));


% change properties
ds9_disp([],[],varargin{:})


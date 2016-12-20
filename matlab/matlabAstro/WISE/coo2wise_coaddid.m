function [ID,JD,Table]=coo2wise_coaddid(RA,Dec,DR) 
%--------------------------------------------------------------------------
% coo2wise_coaddid function                                           WISE
% Description: Given celestial equatorial coordinates, find all WISE
%              coadd_id string ID that covers the coordinates.
% Input  : - J2000.0 RA in [rad] or [H M S] or sexagesimal string.
%          - J2000.0 Dec in [rad] or [Sign D M S] or sexagesimal string.
%          - WISE data relaese, default is 'allsky'.
%            If empty (i.e., []), then use default.
% Output : - A cell array of lists of coadd_id
%            that covers the given coordinates. Each cell for each
%            coordinate.
%          - An array of JD of frame for [band1] frame.
% Tested : Matlab R2014a
%     By : Ilia Labzovsky                  Apr 2012
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [ID,JD,Table]=coo2wise_coaddid(1.0124,-1.4469);
% Reliable: 2
%------------------------------------------------------------------------
RAD = 180./pi;

WISE_Server = 'http://irsa.ipac.caltech.edu/ibe';

DefDR = 'allsky';
if (nargin==2),
   DR     = DefDR;
elseif (nargin==3),
   % do nothing
else
   error('Illegal number of input arguments');
end

if (isempty(DR)),
   DR = DefDR;
end

RA  = convertdms(RA,'gH','r');
Dec = convertdms(Dec,'gD','R');

switch DR
case 'prelim_postcryo'
	error(fprintf('No Atlas images for prelim_postcryo \n'));
case 'prelim'
	table_name	= 'p3am_cdd';
case 'cryo_3band'
	table_name	= 'p3am_cdd';
case 'allsky'
	table_name	= '4band_p3am_cdd';
otherwise
	error(fprintf('DataRelease type not supported \n'));
end

Ind  = [1 2; 2 3; 3 4; 4 1];   % indecies of sides
N	= length(RA);
ID	= cell(N,1);
JD	= zeros(N,1);
%Dist = cell(N,1);

for I=1:1:N,
	URL			= sprintf('%s/search/wise/%s/%s?POS=%f,%f',WISE_Server,DR,table_name,RA(I)*RAD,Dec(I)*RAD);
    str			= urlread(URL);
    Table(I) = read_ipac_table(str,'str',false);
    RA      = Table(I).Cat(:,Table(I).Col.in_ra);
    Dec     = Table(I).Cat(:,Table(I).Col.in_dec);
    Band    = Table(I).Cat(:,Table(I).Col.band);
    RA1     = Table(I).Cat(:,Table(I).Col.ra1);
    RA2     = Table(I).Cat(:,Table(I).Col.ra2);
    RA3     = Table(I).Cat(:,Table(I).Col.ra3);
    RA4     = Table(I).Cat(:,Table(I).Col.ra4);
    Dec1    = Table(I).Cat(:,Table(I).Col.dec1);
    Dec2    = Table(I).Cat(:,Table(I).Col.dec2);
    Dec3    = Table(I).Cat(:,Table(I).Col.dec3);
    Dec4    = Table(I).Cat(:,Table(I).Col.dec4);
    CoaddID = Table(I).Cat(:,Table(I).Col.coadd_id);
    Date1   = Table(I).Cat(:,Table(I).Col.date_obs1);
    JD1     = julday(Date1);
    Date2   = Table(I).Cat(:,Table(I).Col.date_obs2);
    JD2     = julday(Date2);
    Nframe  = Table(I).Cat(:,Table(I).Col.numfrms);
    ZP      = Table(I).Cat(:,Table(I).Col.magzp);
    ID{I}   = spacedel(CoaddID{1});
    
    JD(I)   = (JD1(1)+JD2(1)).*0.5;
end
    
% 		ID{I} = [ID{I}; coadd_id{1}];
% 		MJD{I} = [MJD{I}; obs1];
% 		for Is=1:1:length(Ind),
% 			D0 = sphere_dist(ra(1,Ind(Is,1)), dec(1,Ind(Is,1)), ra(1,Ind(Is,2)), dec(1,Ind(Is,2)));
% 			D1 = sphere_dist(ra(1,Ind(Is,1)), dec(1,Ind(Is,1)), RA(I), Dec(I));
% 			D2 = sphere_dist(ra(1,Ind(Is,2)), dec(1,Ind(Is,2)), RA(I), Dec(I));
% 			
% 			S = 0.5.*(D0+D1+D2);    % semi-perimeter
% 			H = 2.*sqrt(S.*(S-D0).*(S-D1).*(S-D2))./D0;  % Height using Heron formula
% 			dist(:,Is) = H;
% 		end
% 		Dist{I} = [Dist{I}; dist];
% 	end
% end

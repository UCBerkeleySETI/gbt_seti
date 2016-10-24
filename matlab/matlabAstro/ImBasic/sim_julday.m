function [JD,ExpTime,Sim]=sim_julday(Sim,varargin)
%--------------------------------------------------------------------------
% sim_julday function                                              ImBasic
% Description: Get or calculate the Julian day of SIM images, based on
%              their header information. This function is looking for
%              the time and date in a set of header keywords and convert
%              it to JD. The program can also use the exposure time to
%              calculate the mid exposure time.
% Input  : - Structure array images or any of the input supported by
%            images2sim.m. Default '*.fits'.
%            The input must have an header.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'Dictionary' - Dictionary of header keywords to retrieve
%                           and how to intrepret them. This is a two column
%                           cell array. The first column is the header
%                           keyword name, and the second is its type.
%                           Possible types are: 'JD', 'MJD', 'date',
%                           'hour'.
%                           Default is {'JD','JD'; 'OBSJD','JD'; 'MJD','MJD'; 'OBSMJD', 'MJD';'UTC-OBS','date'; 'DATE-OBS','date'; 'DATE','date'; 'UTC','time'; 'OBS-DATE','date'; 'TIME-OBS','time'}
%                           The program will prioretize the keywords by
%                           the order. 'date' type keywords are either
%                           'YYYY-MM-DD HH:MM:SS.frac' like objects
%                           or 'YYYY-MM-DD'. In the later case, the program
%                           will look also for 'hour' object 'HH:MM:SS'
%            'AddDic'     - An additional user dicetionary to be appended
%                           at the begining of the dictionary.
%                           Default is {}.
%            'FunParKey'  - If the date/time is not found within the
%                           dictionary keywords, then the program will
%                           read the keywords specified in this cell
%                           array and use them to calculate the JD
%                           using some formula.
%                           e.g., {'a','b'}. Default is [].
%            'Fun'        - Function to calculate the JD.
%                           Default is @(a,b) (a+b)./86400;
%            'ExpTimeKey' - A cell array of header keywords that may
%                           contain the exposure time.
%                           Default is {'AEXPTIME','EXPTIME'}.
%                           The keywords will prioretize by their order.
%            'DefExpTime' - Default exposure time to use if ExpTimeKey
%                           is not found in the image header. Default is 0.
%                           If empty, then will return an error if exposure
%                           time is not found.
%            'ExpTimeUnits'- Cell array indicating the units of the exposure
%                           time keywords. Default is {'s','s'}.
%                           See convert_units.m for options.
%            'OutTime'    - The output time corresponding to:
%                           {'mid','start','end'} of the exposure.
%                           Default is 'mid'.
%            'OutType'    - Output time type {'JD','MJD'}. Default is 'JD'.
%            'UpdateHead' - Update/write JD to SIM header {true|false}.
%                           Default is true.
%            'UpdateKey'  - Name of header keyword in which to write the
%                           JD. Default is 'JD_MID'.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs: images2sim.m
% Output : - Vector of JDs per input image.
%          - Vector of ExpTime per input image.
%          - SIM array with the updated JD keyword in the header.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Oct 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: JD=sim_julday('*.fits');
% Reliable: 2
%--------------------------------------------------------------------------

if (nargin==0),
    Sim = '*.fits';
end
if (isempty(Sim)),
    Sim = '*.fits';
end

%ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

DefV.Dictionary   = {'JD','JD'; 'OBSJD','JD'; 'MJD','MJD'; 'OBSMJD', 'MJD';'UTC-OBS','date'; 'DATE-OBS','date'; 'DATE','date'; 'UTC','time'; 'OBS-DATE','date'; 'TIME-OBS','time'};
DefV.AddDic       = {}; %{'UTC-OBS','date'};
DefV.FunParKey    = [];
DefV.Fun          = @(a,b) (a+b)./86400;
DefV.ExpTimeKey   = {'AEXPTIME','EXPTIME'};
DefV.DefExpTime   = 0;
DefV.ExpTimeUnits = {'s','s'};
DefV.OutTime      = 'mid';   % {'mid','start','end'}
DefV.OutType      = 'jd';    % {'JD','MJD'}
DefV.UpdateHead   = true;
DefV.UpdateKey    = 'JD_MID';
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});


InPar.Dictionary = [InPar.AddDic; InPar.Dictionary];


if (isstruct(Sim) || issim(Sim)),
    % no need to read images
else
    Sim = images2sim(Sim,varargin{:});
end
Nsim = numel(Sim);

[CellTime] = sim_getkeyvals(Sim,InPar.Dictionary(:,1).','ConvNum',false);
[CellExpTime] = sim_getkeyvals(Sim,InPar.ExpTimeKey,'ConvNum',false);


JD      = zeros(size(Sim));
ExpTime = zeros(size(Sim)).*NaN;
for Isim=1:1:Nsim,
    % read all relevantr header keywords into a cell array
    CellTime1 = CellTime{Isim};
    % look for the first not NaN value
    Inn = find(~isnan_cell(CellTime1),1,'first');
    %Inn = find(cellfun(@isnan,CellTime,'UniformOutput',false)==false,1,'first');
    if (isempty(Inn)),
        % no time found in header based on the time dictionary
        % attempt using Fun
        if (~isempty(InPar.FunParKey)),
            ValFK    = sim_getkeyvals(Sim,InPar.FunParKey,'ConvNum',true);
            JD(Isim) = InPar.Fun(ValFK{1},ValFK{2});
        else
            % no time is available
            JD(Isim) = NaN;
        end
    else
        % time found in header
        switch lower(InPar.Dictionary{Inn,2})
            case 'jd'
                JD(Isim) = str2double_check(CellTime1{Inn});
            case 'mjd'
                JD(Isim) = str2double_check(CellTime1{Inn}) + 2400000.5;
            case 'date'
                DateVec = date_str2vec(CellTime1{Inn});
                if (length(DateVec)<6),
                    % need to read hour separtly
                    Itime=find(~isnan_cell(CellTime1) & strcmp(InPar.Dictionary(:,2),'time'),1,'first');
                    JD(Isim) = julday(DateVec(:,[3 2 1])) + hour_str2frac(CellTime1{Itime});
                    
                else
                    JD(Isim) = julday(DateVec(:,[3 2 1 4 5 6]));
                end
                
            case 'hour'
                % do nothing - treated in 'date'
            otherwise
                error('Not in time keywords dictionary');
        end
    end
    
    
    % calculate mid exposure time
    switch lower(InPar.OutTime)
        case 'start'
            % do nothing
        otherwise
            CellExpTime1 = CellExpTime{Isim};
            % look for the first not NaN value
            Inn = find(~isnan_cell(CellExpTime1),1,'first');
            if (isempty(Inn)),
                if (isempty(InPar.DefExpTime)),
                    error('Exposure time not found');
                else
                    ExpTime(Isim) = InPar.DefExpTime;
                    Inn = 1;
                end
            else
                ExpTime(Isim) = str2double_check(CellExpTime1{Inn});
            end
            
            ConvFactor = convert_units(InPar.ExpTimeUnits{Inn},'day');
            switch lower(InPar.OutTime)
                case 'mid'
                    JD(Isim) = JD(Isim) + 0.5.*ConvFactor.*ExpTime(Isim);
                case 'end'
                    JD(Isim) = JD(Isim) + ConvFactor.*ExpTime(Isim);
                otherwise
                    error('Unknown OutTime option');
            end
            
    end
    
    if (InPar.UpdateHead)
        [Sim(Isim).(HeaderField)]=cell_fitshead_addkey(Sim(Isim).(HeaderField),...
                                           sprintf('%s',InPar.UpdateKey), JD(Isim), 'JD for middle of exposure');
    end
    
 
end


switch lower(InPar.OutType)
    case 'jd'
        % do nothing
    case 'mjd'
        JD = JD - 2400000.5;
    otherwise
        error('Unknown OutType option');
end

        

        
        
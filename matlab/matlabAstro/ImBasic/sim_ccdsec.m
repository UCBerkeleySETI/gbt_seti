function [OutCCDSEC]=sim_ccdsec(Sim,CCDSEC,varargin)
%--------------------------------------------------------------------------
% sim_ccdsec function                                              ImBasic
% Description: Get CCDSEC keyword value from a Sim structure array.
%              If not available use image size.
% Input  : - Structure array of images or any image type readable
%            by images2sim.m.
%          - Either a CCDSEC header keyword name from which to exctract
%            the CCDSEC, or a CCDSEC vector that will override the
%            header keyword. If empty, or header keyword is not available
%            then use the image size and return CCDSEC for the entire
%            image size.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'UseNaN' - {true|false}. If true then will return NaN
%                       if header keyword is not availabl or NaN.
%                       Default is false.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs: images2sim.m
% Output : - A 4 column Matrix with CCDSEC [Xmin, Xmax, Ymin, Ymax]
%            values for all the images in Sim - row per image.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Oct 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: 
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';

if (nargin==1),
    CCDSEC = [];
end

DefV.UseNaN = false;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

Sim = images2sim(Sim);

Nim = numel(Sim);
OutCCDSEC = zeros(Nim,4);
for Iim=1:1:Nim,
    if (isempty(CCDSEC)),
        Size = size(Sim(Iim).(ImageField));
        OutCCDSEC(Iim,:) = [1 Size(2) 1 Size(1)];
    else
        if (ischar(CCDSEC)),
            % get CCDSEC from header
            [NewCellHead] = cell_fitshead_getkey(Sim(Iim).(HeaderField),CCDSEC,'NaN');
            CCDSEC_Val = NewCellHead{2};

            if (isnan(CCDSEC_Val)),
                if (InPar.UseNaN),
                    OutCCDSEC(Iim,:) = [NaN NaN NaN NaN];
                else
                    Size = size(Sim(Iim).(ImageField));
                    OutCCDSEC(Iim,:) = [1 Size(2) 1 Size(1)];
                end
            else
                Splitted   = regexp(CCDSEC_Val,'[:\[\],]','split');
                OutCCDSEC(Iim,:)  = [str2double(Splitted{2}), str2double(Splitted{3}), str2double(Splitted{4}), str2double(Splitted{5})];
            end
        elseif (isnumeric(CCDSEC)),
            OutCCDSEC(Iim,:) = CCDSEC;
        else
            error('Unknown CCDSEC type');
        end
    end
end

    
        
        
    


function [Gain,Sim]=sim_gain(Sim,varargin)
%--------------------------------------------------------------------------
% sim_gain function                                                ImBasic
% Description: Read the Gain keyword from the header of SIM or FITS images
%              and optionally multiply each image by its gain, and
%              set the new gain value to 1.
% Input:   - Set of images. See images2sim.m for options.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'Gain' - A string (e.g. 'GAIN'). In this case the gain value
%                     is obtained from the header.  
%                     Or, a scalar or vector of gain (per image).
%                     Default is 'GAIN'.     
%            'CorrectGain' - Multiply the image by its gain {true|false}.
%                     Default is true.
%            'UpdateHeader' - Update the gain value in the header
%                     {true|false}. Default is true.
%            'GainNanAbort' - {true|false}. Abort if Gain is NaN and user
%                     requested to correct by gain. Default is true.
%            'ImageField' - The field name in the structure image (Sim)
%                     containing the image to multiply. 
%                     Default is 'Im'.
%            'HeaderField' - The field name in the structure image (Sim)
%                     containing the header.
%                     Default is 'Header'.
%            'CopyHead' - Copy header from original image {'y' | 'n'}.
%                     Default is 'y'.
%            'AddHead' - Cell array with 3 columns containing additional
%                     keywords to be add to the header.
%                     See cell_fitshead_addkey.m for header structure
%                     information. Default is empty matrix.
%            'OrigGainKey' - An header keyword name in which to store
%                     the image original gain before it was corrected.
%                     Default is 'ORIGGAIN'.
%                     If empty then do not store original gain.
%            'CorrGainBack' - Correct background image gain {true|false}.
%                     Default is true.
%            'CorrGainErr' - Correct error image gain {true|false}.
%                     Default is true.
%            'OutSIM' - Force output to be a SIM class (true).
%                     {true|false}. Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m
% Output : - Vector of gain values.
%          - Structure array of updated images (SIM).
% Tested : Matlab R2013a
%     By : Tali Engel                      Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Gain,Sim]=sim_gain('lred0126.fits');
% Reliable: 2
%--------------------------------------------------------------------------

FunName = mfilename;

ImageField  = 'Im';
HeaderField = 'Header';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';


% --- default values ---
DefV.Gain             = 'GAIN';
DefV.CorrectGain      = true;
DefV.UpdateHeader     = true;
DefV.GainNanAbort     = true;
DefV.ImageField       = ImageField;
DefV.HeaderField      = HeaderField;
DefV.CopyHead         = 'y';
DefV.AddHead          = {};
DefV.OrigGainKey      = 'ORIGGAIN';
DefV.CorrGainBack     = true;
DefV.CorrGainErr      = true;
DefV.OutSIM           = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

ImageField  = InPar.ImageField;
HeaderField = InPar.HeaderField;

%--- read images ---
Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

if (InPar.OutSIM && ~issim(Sim)),
    Sim = struct2sim(Sim);            % output is of SIM class
end

%--- Go over images, obtain gain values ---
if (ischar(InPar.Gain)),
    %--- read data from header ---    
    [~,Header_struct] = sim_getkeyvals(Sim,InPar.Gain);
    Gain              = [Header_struct.(InPar.Gain)];   % If Nim>1, Gain is a vector.
    if (isnan(Gain)),
        fprintf('No gain in header - set gain to 1\n');
        Gain = 1;
    end
else
    Gain = InPar.Gain.*ones(Nim,1);             % Gain may be a vector or a scalar.
end

%--- multiply by the gain ---
if (InPar.CorrectGain && nargout>1),
    for Iim=1:1:Nim
        if (isnan(Gain(Iim))),
            if (InPar.GainNanAbort),
                error('Gain is NaN in image number=%d',Iim);
            else
                warning('sim_gain:gain_is_NaN','Gain is NaN in image number=%d',Iim);
            end
        end
                
            
        Sim(Iim).(ImageField) = Gain(Iim).*Sim(Iim).(ImageField);
        if (InPar.CorrGainBack),
            Sim(Iim).(BackImField) = Gain(Iim).*Sim(Iim).(BackImField);
        end
        if (InPar.CorrGainErr),
            Sim(Iim).(ErrImField) = Gain(Iim).*Sim(Iim).(ErrImField);
        end
        
        %--- update header ---
        Sim(Iim) = sim_update_head(Sim(Iim),'CopyHead',InPar.CopyHead,...
                                            'AddHead',InPar.AddHead,...
                                            'Comments',{sprintf('Updated by %s.m written by Tali Engel',FunName)});                        
        if (InPar.UpdateHeader)         
            Sim(Iim).(HeaderField) = cell_fitshead_update(Sim(Iim).(HeaderField),InPar.Gain,1.0,'CCD Gain (after correction by sim_gain.m)');
            if (~isempty(InPar.OrigGainKey)),
                Sim(Iim).(HeaderField) = cell_fitshead_update(Sim(Iim).(HeaderField),InPar.OrigGainKey,Gain(Iim),'Original CCD gain');
            end
        end                                
    end
end




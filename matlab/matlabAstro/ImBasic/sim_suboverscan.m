function Sim=sim_suboverscan(Sim,varargin)
%--------------------------------------------------------------------------
% sim_suboverscan function                                         ImBasic
% Description: Calculate and subtract overscan bias from a list of images.
% Input  : - Image from which to subtract overscan bias.
%            The following inputs are possible:
%            (1) Cell array of image names in string format.
%            (2) String containing wild cards (see create_list.m for
%                option). E.g., 'lred00[15-28].fits' or 'lred001*.fits'.
%            (3) Structure array of images (SIM).
%                The image should be stored in the 'Im' field.
%                This may contains also mask image (in the 'Mask' field),
%                and an error image (in the 'ErrIm' field).
%            (4) Cell array of matrices.
%            (5) A file contains a list of image (e.g., '@list').
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'BiasSec' - Bias Section. This can be either a vector of
%                        [xmin xmax ymin ymax] or a string containing
%                        the header keyword name containing the
%                        overscan bias section.
%                        This parameter must be provided.
%            'BiasAxis'- The axis of the bias over scan
%                        {'x'|'y'|'guess'}. Default is 'guess'.
%            'BiasMethod'- An handle to a bias estimatation function.
%                        The function, get the overscan region and
%                        the overscan dimension (1 for x | 2 for y).
%                        Default is @median.
%            --- Additional parameters
%            Any additional key,val, that are recognized by images2sim.m.
% Output : - Structure array of overscan bias subtracted images.
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Sim=sim_suboverscan(Sim);
% Reliable: 2
%-----------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
%BackImField = 'BackIm';
%ErrImField  = 'ErrIm';



DefV.BiasSec       = [];
DefV.BiasAxis      = 'guess';   % {'x'|'y'|'guess'}
DefV.BiasMethod    = @median;
%DefV.RejectN       = [0 1];     % low and high rejection

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

if (isempty(InPar.BiasSec)),
    error('Bias section must be provided');
end

MapBiasAxis = {'x','y'};

Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

for Iim=1:1:Nim,
    % get overscan section
    if (~isfield(Sim(Iim),HeaderField)),
        Sim(Iim).(HeaderField) = [];
    end
    BiasSec = get_ccdsec_head(Sim(Iim).(HeaderField),InPar.BiasSec);
    if (any(isnan(BiasSec))),
        error('Can not retrieve valid values for bias section');
    end
    
    % guess bias section direction
    ImSize = size(Sim(Iim).(ImageField));
    switch lower(InPar.BiasAxis)
        case 'guess'
            
            if (ImSize(2)==BiasSec(2) && BiasSec(1)==1),
                BiasAxis = 'x';
            elseif (ImSize(1)==BiasSec(4) && BiasSec(3)==1),
                BiasAxis = 'y';
            else
                error('Can not guess bias axis for image %d',Iim);
            end
            
        otherwise
            BiasAxis = InPar.BiasAxis;
    end
   
    BiasAxisDir = find(strcmpi(MapBiasAxis,BiasAxis));
    BiasVec = feval(InPar.BiasMethod,Sim(Iim).(ImageField)(BiasSec(3):BiasSec(4),BiasSec(1):BiasSec(2)),BiasAxisDir);
    Sim(Iim).(ImageField) = bsxfun(@minus,Sim(Iim).(ImageField),BiasVec);
    
end

    
    




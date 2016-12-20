function Sim=sim_filter(Sim,varargin)
%--------------------------------------------------------------------------
% sim_filter function                                              ImBasic
% Description: cross-correlate SIM images with filters.
% Input  : - Images to filter. Input can be FITS, SIM or other types of
%            images. For input options see images2sim.m.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'Filter' - Filters to use on each image.
%                       This can be a cell array of matrices,
%                       a single matrix, a singel cell or a SIM
%                       array of filters. Default is empty.
%                       However, the filter must be provided.
%                       Filter must contain odd by odd number of pixels.
%            'Back'    - If true, then will subtract the background using
%                        sim_background.m, you can pass parameters to
%                        sim_background.m.
%                        If empty, or false then no background will be
%                        subtracted (default).
%                        alternatively will use images2sim.m to read
%                        the background images. 
%            'FiltNorm'- Normalize filters such their sum equal to this
%                        number. Default is 1. If empty, do not normalize.
%                        Default is 1.
%            'MaskFilter' - A flag indicating if the mask image is needed
%                        to be filter {true | false}. Default is true.
%                        If true, then the filter for the mask image
%                        will be constructed of ones where the filter is
%                        not equal zero. NOT OPERATIONAL.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m, image2sim.m, sim_background.m
% Output : - The input SIM images, but in which the ImageField contained
%            the filtered image.
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Apr 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: S=sim_filter(A,'Filter',B)
% Reliable: 2
%--------------------------------------------------------------------------

ImageField     = 'Im';
MaskField      = 'Mask';

DefV.Filter            = {};
%DefV.FiltPrep          = 'shift';
DefV.Back              = [];
DefV.FiltNorm          = 1;
%DefV.MaskFilter        = true;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% read images
Sim = images2sim(Sim,varargin{:});
Nim = numel(Sim);

% prepare Filter
if (isnumeric(InPar.Filter)),
    InPar.Filter = {InPar.Filter};
else
    if (issim(InPar.Filter) || isstruct(InPar.Filter)),
        InPar.Filter = {InPar.Filter.(ImageField)};
    end
end
Nfilt = numel(InPar.Filter);

if (isempty(InPar.Filter)),
    error('Filter must be provided');
end

% prepare background
if (islogical(InPar.Back)),
    if (InPar.Back),
        Sim = sim_background(Sim,varargin{:},'SubBack',true);
    else
        % do not subtract background
        InPar.Back = [];
    end
else
    if (isempty(InPar.Back)),
        % do not subtract background
    else
        % Subtract background
        SimBack = images2sim(InPar.Back,varargin{:});
        Sim     = Sim - SimBack;
    end
end




for Iim=1:1:Nim,
    Ifilt  = min(Iim,Nfilt);  % select index of filter to use
    Filter = InPar.Filter{Ifilt};
    Image  = Sim(Iim).(ImageField);
    
    % normalize filter
    if (~isempty(InPar.FiltNorm)),
        Filter = Filter.*InPar.FiltNorm./sum(Filter(:));
    end 
    
    Sim(Iim).(ImageField) = filter2smart(Image,Filter);
   
end



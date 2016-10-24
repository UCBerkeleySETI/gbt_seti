function [SimMos]=sim_mosaic(Sim,CCDpos,varargin)
%--------------------------------------------------------------------------
% sim_mosaic function                                              ImBasic
% Description: Mosaicing (tiling) a set of images into a single image.
% Input  : - Images to mosaic. This is either FITS or SIM images or
%            any input acceptable by images2sim.m.
%          - A matrix indicating the CCDSEC position [Xmin Xmax Ymin Ymax]
%            in each line. Each line corresponds to one image in the
%            order they are being read.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m, image2sim.m
% Output : - SIM image containing the mosaic.
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Apr 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [SimMos]=sim_mosaic({'A.fits','B.fits'},[1 100 1 100; 101 200 1 100]);
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
%HeaderField = 'Header';
%FileField   = 'ImageFileName';
MaskField   = 'Mask';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';
WeightImField = 'WeightIm';

%InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% read images
Sim = images2sim(Sim,varargin{:});
%Nim = numel(Sim);

%MinX = min(CCDpos(:,1));
MaxX = max(CCDpos(:,2));
%MinY = min(CCDpos(:,3));
MaxY = max(CCDpos(:,4));
Npos = size(CCDpos,1);

SimMos = SIM;
SimMos.(ImageField) = zeros(MaxY,MaxX);

for Ipos=1:1:Npos,
    if (isfield_notempty(Sim(Ipos),ImageField)),
        SimMos.(ImageField)(CCDpos(Ipos,3):CCDpos(Ipos,4),CCDpos(Ipos,1):CCDpos(Ipos,2)) = Sim(Ipos).(ImageField);
    end
    if (isfield_notempty(Sim(Ipos),MaskField)),
        SimMos.(MaskField)(CCDpos(Ipos,3):CCDpos(Ipos,4),CCDpos(Ipos,1):CCDpos(Ipos,2)) = Sim(Ipos).(MaskField);
    end
    if (isfield_notempty(Sim(Ipos),BackImField)),
        SimMos.(BackImField)(CCDpos(Ipos,3):CCDpos(Ipos,4),CCDpos(Ipos,1):CCDpos(Ipos,2)) = Sim(Ipos).(BackImField);
    end
    if (isfield_notempty(Sim(Ipos),ErrImField)),
        SimMos.(ErrImField)(CCDpos(Ipos,3):CCDpos(Ipos,4),CCDpos(Ipos,1):CCDpos(Ipos,2)) = Sim(Ipos).(ErrImField);
    end
    if (isfield_notempty(Sim(Ipos),WeightImField)),
        SimMos.(WeightImField)(CCDpos(Ipos,3):CCDpos(Ipos,4),CCDpos(Ipos,1):CCDpos(Ipos,2)) = Sim(Ipos).(WeightImField);
    end
end

            


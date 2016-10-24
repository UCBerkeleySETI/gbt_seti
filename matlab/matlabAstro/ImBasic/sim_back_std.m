function [Sim,BackStd]=sim_back_std(Sim,varargin)
%--------------------------------------------------------------------------
% sim_back_std function                                            ImBasic
% Description: Estimate background and std of an image. The background and
%              std are calculated in local blocks (bins), and than
%              interpolated to each pixel.
% Input  : - Images. See images2sim.m for possible formats.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'BlockN' - How many blocks (bins) to use in each dimension
%                       [y,x]. Default is [4 4].
%                       If one of the elements is zero than will treat
%                       the image as a single block.
%            'BackStdAlgo' - Background and std algorith.
%                       'mode_fit': fit Gaussian to background
%                                  using mode_fit.m.
%                       'median' : estimate background using median
%                                  and std usinf robust std (nanrstd.m).
%                       Default is 'mode_fit'
%            'InterpMethod' - Interpolation method. See interp2.m for options.
%                       Default is '*cubic'.
%            'SubBack' - Subtract background from image {true|false}.
%                       Default is true.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m, image2sim.m, 
% Output : - SIM images with the background and error images.
% See also: sim_background.m, sim_std.m
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Sep 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Sim,Back,StD]=sim_back_std(Sim);
% Reliable: 2
%--------------------------------------------------------------------------


ImageField  = 'Im';
HeaderField = 'Header';
%FileField   = 'ImageFileName';
%MaskField   = 'Mask';
BackImField = 'BackIm';
ErrImField  = 'ErrIm';


DefV.BlockN       = [0 0]; %[2 2];   % [y, x]
DefV.BackStdAlgo  = 'mode_fit';
DefV.InterpMethod = '*cubic';
DefV.SubBack      = true;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% read images
Sim = images2sim(Sim);
Nim = numel(Sim);

Nby = InPar.BlockN(1);
Nbx = InPar.BlockN(2);

if (Nbx<3 || Nby<3),
    InPar.InterpMethod = 'linear';
end

% for each image
for Iim=1:1:Nim,
    % prepare blocks
    ImSize = size(Sim(Iim).(ImageField));
    [FullMatX, FullMatY] = meshgrid((1:1:ImSize(2)),(1:1:ImSize(1)));
    
    BlockSizeY = ImSize(1)./Nby;
    BlockSizeX = ImSize(2)./Nbx;
    
    % for each block
    Back = zeros(Nby+1,Nbx+1);
    StD  = zeros(Nby+1,Nbx+1);
    MatX = zeros(Nby+1,Nbx+1);
    MatY = zeros(Nby+1,Nbx+1);
    
    % for each block 
    for Iby=1:1:Nby+1,
        for Ibx=1:1:Nbx+1,
            if (Nbx==0 || Nby==0),
                % treat the image as a single block
                CenterX = ImSize(2)./2;
                CenterY = ImSize(1)./2;
                BX      = (1:1:ImSize(2));
                BY      = (1:1:ImSize(1));
            else
                CenterY = (Iby-1).*BlockSizeY+1;
                CenterX = (Ibx-1).*BlockSizeX+1;
                BY      = (CenterY-floor(BlockSizeY.*0.5):1:CenterY+ceil(BlockSizeY.*0.5))';
                BX      = (CenterX-floor(BlockSizeX.*0.5):1:CenterX+ceil(BlockSizeX.*0.5))';
                BY      = BY(BY>0 & BY<ImSize(1));
                BX      = BX(BX>0 & BX<ImSize(2));
            end
            MatY(Iby,Ibx) = CenterY;
            MatX(Iby,Ibx) = CenterX;
            switch lower(InPar.BackStdAlgo)
                case {'mode_fit'}
                    [Back(Iby,Ibx),StD(Iby,Ibx)] = mode_fit(Sim(Iim).(ImageField)(BY,BX));
                    
                case 'median'
                    Tmp = Sim(Iim).(ImageField)(BY,BX);
                    Back(Iby,Ibx) = nanmedian(Tmp(:));
                    StD(Iby,Ibx)  = nanrstd(Tmp(:));
                otherwise
                    error('Unknown background/std method option');
            end
        end
    end
    BackStd(Iim).Back = Back;
    BackStd(Iim).StD  = StD;
    
    if (Nbx==0 || Nby==0),
        Sim(Iim).(BackImField) = Back;
        Sim(Iim).(ErrImField)  = StD;
    else
        Sim(Iim).(BackImField) = interp2(MatX,MatY,Back,FullMatX,FullMatY,InPar.InterpMethod);    
        Sim(Iim).(ErrImField)  = interp2(MatX,MatY,StD,FullMatX,FullMatY,InPar.InterpMethod);
    end
    % subtract background from image
    if (InPar.SubBack),
        Sim(Iim).(ImageField) = Sim(Iim).(ImageField) - Sim(Iim).(BackImField);
    end
end

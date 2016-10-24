function Bit=def_bitmask_photpipeline(Field)
%--------------------------------------------------------------------------
% def_bitmask_photpipeline function                                 ImPhot
% Description: The spectroscopic pipeline bit mask definition.
%              Given the Bit mask name return the bit mask index.
% Input  : - Bit mask name.
% Output : - Bit index.
% License: GNU general public license version 3

% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Feb 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% See also: maskflag_check.m, maskflag_set.m
% Example: Bit=def_bitmask_photpipeline('Bit_ImSaturated');
% Reliable: 2
%--------------------------------------------------------------------------


Ind = 0;
Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImDeadPix';        %       % pixel bias response is close to 0

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImHotPix';         %       % pixel bias response is high

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImNoisyPix';       %       % pixel bias std is high

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImLowNoisePix';    %       % pixel bias std is low

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImSaturated';      %       % pixel is saturated

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImNonLinear';      %       % pixel is in non-linear regime

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImFlatNaN';        %       % pixel has no flat value

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImFlatLowNim';     %       % pixel has small numeber of input images

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImFlatLow';        %       % pixel has low flat response

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImFlatHigh';       %       % pixel has very high flat response (maybe a problem)

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImFlatNoisy';      %       % pixel has high flat std

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImDivide0';        %       % pixel is affected by division by 0

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImCR';             %       % pixel is likely affected by CR

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImSrc';            %       % pixel is likely to contain a source

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImSrcSN';          %       % pixel is likely to contain a source above a given S/N

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImBckHigh';        %       % pixel is likely to have background level higher than image best fir background

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImSpike';          %       % pixel is near possible bright star diffraction spike

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImHalo';           %       % pixel is near possible bright star halo

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImGhost';          %       % pixel is near possible image ghost

Ind = Ind + 1;
Map(Ind).Ind  = Ind;
Map(Ind).Name = 'Bit_ImNearEdge';       %       % pixel is near image edge


Ind = find(strcmp({Map.Name},Field));
Bit = Map(Ind).Ind;

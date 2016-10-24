function Out=convert_flux(In,InUnits,OutUnits,Freq,FreqUnits)
%------------------------------------------------------------------------------
% convert_flux function                                              AstroSpec
% Description: Convert between different flux units
% Input  : - Flux
%          - Input units - options are:
%            'cgs/A'    [erg/s/cm^2/A]
%            'cgs/Hz'   [erg/s/cm^2/Hz]
%            'mJy'      [1e-26 erg/s/cm^2/Hz]
%            'AB'       [AB magnitude]
%            'ph/A'     [ph/s/cm^2/A]
%            'ph/Hz'    [ph/s/cm^2/Hz]
%          - Output units (see input units).
%          - Frequency at which flux is measured.
%          - Units of frequency - options are:
%            'Hz'
%            'A'
%            'cm'
%            'nm'
%            'eV'
%            'keV'...
%            see convert_energy.m for more options
% Output : - Flux in output units
% Tested : Matlab 7.11
%     By : Eran O. Ofek                  March 2011
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Out=convert_flux(1,'mJy','AB')
% Reliable: 2
%------------------------------------------------------------------------------

c = get_constant('c');  % speed of light [cm/s]
h = get_constant('h');  % planck constant [cgs]


Freq_Hz = convert_energy(Freq,FreqUnits,'Hz');

switch lower(InUnits)
    case 'cgs/a'
       % convert to mJy
       Flux_mJy = In.*(c.*1e8./(Freq_Hz.^2))./1e-26;
    case 'cgs/hz'
       % convert to mJy
       Flux_mJy = In./1e-26;
    case 'mjy'
       % already in mJy
       Flux_mJy = In;
    case 'ab'
       % convert to mJy
       Flux_mJy = 1e3.*10.^(23-(In+48.6)./2.5);
    case 'ph/a'
       Flux_mJy = In.*h.*Freq_Hz  .*(c.*1e8./(Freq_Hz.^2))./1e-26;
    case 'ph/hz'
       Flux_mJy = In.*h.*Freq_Hz    ./1e-26;
    otherwise
        error('Unknown InUnits option');
end

switch lower(OutUnits)
    case 'cgs/a'
       % convert mJy to cgs/A
       Out = Flux_mJy./((c.*1e8./(Freq_Hz.^2))./1e-26);
    case 'cgs/hz'
       % convert mJy to cgs/Hz
       Out = Flux_mJy.*1e-26;
    case 'mjy'
       Out = Flux_mJy;
    case 'ab'
       Out = 2.5.*(23-log10(Flux_mJy.*1e-3))-48.6;
    case 'ph/a'
       Out = Flux_mJy./(h.*Freq_Hz  .*(c.*1e8./(Freq_Hz.^2))./1e-26);
    case 'ph/hz'
       Out = Flux_mJy./(h.*Freq_Hz    ./1e-26);
    otherwise
        error('Unknown OutUnits option');
end 

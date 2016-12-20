function OutSim=struct2sim(InSim)
%--------------------------------------------------------------------------
% struct2sim function                                              ImBasic
% Description: Convert a structure array to SIM object.
% Input  : - A structure image array.
%          * Arbitrary number of pairs or arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
% Output : - A structure image array converted to SIM object.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Oct 2014
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: S=struct2sim(S);
% Reliable: 2
%--------------------------------------------------------------------------

if (issim(InSim)),
    OutSim = InSim;
else
    FN     = fieldnames(InSim);
    Nfn    = numel(FN);

    OutSim = SIM;
    Nsim   = numel(InSim);

    for Isim=1:1:Nsim,
        for Ifn=1:1:Nfn,
            OutSim(Isim).(FN{Ifn}) = InSim(Isim).(FN{Ifn});
        end
    end
end

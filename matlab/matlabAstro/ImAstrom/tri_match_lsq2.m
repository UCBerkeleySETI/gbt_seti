function [ParBest,Isel]=tri_match_lsq2(ListRef,ListCat,varargin)
%--------------------------------------------------------------------------
% tri_match_lsq2 function                                         ImAstrom
% Description: Given two matrices of [X,Y, Mag] columns, where the Mag
%              column is optional, attempt to find a shift+scale+rotation
%              transformation between the two lists using triangle pattren
%              matching.
% Input  : - Three columns, reference list [X, Y].
%          - Three columns catalog list [X,Y] to match to reference
%            list.
%          * Arbitrary number of pairs of ...,key,val,... arguments.
%            The following keywords are available:
% 
% Tested : Matlab R2011b
%     By : Eran O. Ofek                    Sep 2013
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [ParBest,Isel]=tri_match_lsq2;   % run in simulation mode
%          [ParBest,Isel]=tri_match_lsq2(ListRef,ListCat);
% Reliable: 2
%--------------------------------------------------------------------------
RAD           = 180./pi;
%ARCSEC_IN_DEG = 3600;

%tic;
if (nargin==0),
    % simulation mode
    Nstar = 500;
    ListRef = rand(Nstar,2).*2048;
    Noverlap = 50;
    ListCat = [ListRef(1:Noverlap,1), ListRef(1:Noverlap,2)];
    ListCat = [ListCat; rand(Nstar-Noverlap,2).*2048];
    ListCat = ListCat + 20 + randn(Nstar,2).*0.3;
end

DefV.TriCombBaseFileName = 'TriComb';
DefV.TriCombFilePath     = '/home/eran/matlab/fun/ImAstrom/';
DefV.SaveTriCombFile     = true;  % {true|false}
DefV.MinTriDist          = 100;
DefV.MaxTriDist          = 1000;
%DefV.RevX                = false;
%DefV.RevY                = false;
%DefV.TestRev             = true;
DefV.RevTable            = [1 1; 1 -1; -1 1;-1 -1];  % [1 1]
DefV.NcombList           = 1e5;
DefV.MaxDistCenter       = 100; %5000;
DefV.RotationRange       = [-3 3]./RAD;
DefV.NrandTri            = 1000;
DefV.TriMaxRatio         = 3;
DefV.TriMaxResid         = 0.3;   % [arcsec]
DefV.NsigRatioErr        = 1;
DefV.Scale               = 1;
DefV.ScaleErr            = 0.05;  
DefV.RatioErr            = [];
DefV.StopNumber          = 20;
DefV.MinFracMatch        = 0.15;
DefV.MaxRMS              = 0.4;
DefV.MinNmatch           = 50;
DefV.TestMatchDist       = 2;
DefV.SelectMethod        = 'minRMS'; % {'minRMS'|'maxNmatch'|'maxFracMatch'}
                
InPar = set_varargin_keyval(DefV,'y','use',varargin{:});

if (isempty(InPar.RatioErr)),
    InPar.RatioErr = InPar.NsigRatioErr.*InPar.TriMaxResid./InPar.MinTriDist;
end

% number of elements in Ref and Cat
Nref = size(ListRef,1);
Ncat = size(ListCat,1);

Xref = ListRef(:,1);
Yref = ListRef(:,2);
Xcat = ListCat(:,1);
Ycat = ListCat(:,2);

tic;
Dx = bsxfun(@minus,Xref,Xcat.');
Dy = bsxfun(@minus,Yref,Ycat.');
[H,VX,VY] = hist2d(Dx(:),Dy(:),[-100 100],1,[-100 100],1);
toc


%--- all possible distances ---
Dref = tril(sqrt(bsxfun(@minus,Xref,Xref.').^2 + bsxfun(@minus,Yref,Yref.').^2));
Dcat = tril(sqrt(bsxfun(@minus,Xcat,Xcat.').^2 + bsxfun(@minus,Ycat,Ycat.').^2));
[Jref,Iref] = meshgrid((1:1:Nref),(1:1:Nref));
[Jcat,Icat] = meshgrid((1:1:Ncat),(1:1:Ncat));

[SDref,SIref] = sort(Dref(:));
[SDcat,SIcat] = sort(Dcat(:));
Jref          = Jref(SIref);
Iref          = Iref(SIref);
Jcat          = Jref(SIcat);
Icat          = Iref(SIcat);

tic;
for J1=1:1:Ncat-2,
    for I1=J1+1:1:Ncat-1,
        for I2=I1+1:1:Ncat,
            %J2   = 
            %TriI = [I1 I2 Iref(I2,
            %TriJ = [J1 
        end
    end
end
toc




% Ir0 = find(SDref>0);
% Ic0 = find(SDcat>0);
% Im1=mfind_bin(SDcat(Ic0),SDref(Ir0).'-1);
% Im2=mfind_bin(SDcat(Ic0),SDref(Ir0).'+1);


L_Dcat           = find(SDcat>20 & SDcat<200);
[I_Dcat, J_Dcat] = ind2sun(size(SDcat),L_Dcat);
for Istar=1:1:numel(L_Dcat),





Kdc = find(Dcat>20 & Dcat<200);
[Idc,Jdc] = ind2sub(size(Dcat),Kdc);
for Istar=1:1:numel(Kdc),
    % look for pair in Ref with simlar dist:
    % Kdr are the linear indices of all the pairs in Ref
    % which have distance similar to Dcat(Kdc(Istar))
    Kdr = find(abs(Dcat(Kdc(Istar))-Dref(:))<1.5);
    
    % look for third star:
    L   = find(Dcat(1:Jdc(Istar)-1,Jdc(Istar))<200);   % possible matches are at [L, Jdc(Istar)]
    [Istar, length(L)];
    if (numel(L)>1),
        % remove from L the base star
        L = setdiff(L,Idc(Istar));
        % possible third star to work with found
        % look fot the third star in Ref
        % Stars which have similar dist to: Dcat(L,Jdc(Istar))
        [Idr,Jdr] = ind2sub(size(Dref),Kdr);
        for Il=1:1:numel(L),
            LR = find(abs(Dcat(L(Il),Jdc(Istar))-Dref(:))<1.5);
            [LIdr,LJdr] = ind2sub(size(Dref),LR);
        end
    end
end

            
[Idr,Jdr] = ind2sub(size(Dref),Kdr);



%--- indices of all triplet combinations in ListRef ---
% this may take time so try to see if exist.

TriCombFileName = sprintf('%s_%04d.mat',InPar.TriCombBaseFileName,Nref);
if (exist(TriCombFileName,'file')>0),
    % combination file found in path - try to upload
    CIref = load2(TriCombFileName);
else
    % combination file was not found
    % calculate combination
    %CIref = combnk((1:1:Nref).',3);    % triangle combinations
    CIref = combinator(Nref,3,'c');
    if (InPar.SaveTriCombFile)
        save(sprintf('%s%s',InPar.TriCombFilePath,TriCombFileName),'CIref');
    end
end

%CIA = combnk((1:1:Nref).',3);    % triangle combinations
%CIA = combnk((1:1:NA).',3);     % list A

    
%--- Ref list properties ---
% distances between verteces [d12, d23, d31] ---
Dref = [sqrt((Xref(CIref(:,1))-Xref(CIref(:,2))).^2 + (Yref(CIref(:,1))-Yref(CIref(:,2))).^2),...
        sqrt((Xref(CIref(:,2))-Xref(CIref(:,3))).^2 + (Yref(CIref(:,2))-Yref(CIref(:,3))).^2),...
        sqrt((Xref(CIref(:,3))-Xref(CIref(:,1))).^2 + (Yref(CIref(:,3))-Yref(CIref(:,1))).^2)];

% select triangles which sides are not too small or too large 
% units are radians
Igood_tri = find(min(Dref,[],2)>InPar.MinTriDist & max(Dref,[],2)<InPar.MaxTriDist & max(Dref,[],2)./min(Dref,[],2)<InPar.TriMaxRatio);
Dref      = Dref(Igood_tri,:);
CIref     = CIref(Igood_tri,:);

% sort traingle verteces by length of sides
[SDref,SIref] = sort(Dref,2);
% build the sorted vertecs list
IndMat = repmat((1:1:size(SIref,1)).',1,3);
SCIref   = reshape(CIref(sub2ind([size(SIref,1),3],IndMat(:),SIref(:))),size(SIref,1),3);
RefRatio12 = SDref(:,1)./SDref(:,2);
RefRatio23 = SDref(:,2)./SDref(:,3);
   
% [X1, X2, X3] matrix of sorted triangles (by distances)
%XtRef = round(0.5-real(InPar.RevX)).*Xref(SCIref);
%YtRef = round(0.5-real(InPar.RevY)).*Yref(SCIref);
XtRef = Xref(SCIref);
YtRef = Yref(SCIref);

%--- Random triplets in ListCat ---
CIcat = zeros(InPar.NcombList,3);
for I=1:1:InPar.NcombList,
   CIcat(I,:) = randperm(Ncat,3);
end

Dcat = [sqrt((Xcat(CIcat(:,1))-Xcat(CIcat(:,2))).^2 + (Ycat(CIcat(:,1))-Ycat(CIcat(:,2))).^2),...
        sqrt((Xcat(CIcat(:,2))-Xcat(CIcat(:,3))).^2 + (Ycat(CIcat(:,2))-Ycat(CIcat(:,3))).^2),...
        sqrt((Xcat(CIcat(:,3))-Xcat(CIcat(:,1))).^2 + (Ycat(CIcat(:,3))-Ycat(CIcat(:,1))).^2)];

Igood_tri = find(min(Dcat,[],2)>InPar.MinTriDist & max(Dcat,[],2)<InPar.MaxTriDist & max(Dcat,[],2)./min(Dcat,[],2)<InPar.TriMaxRatio);
Dcat      = Dcat(Igood_tri,:);
CIcat     = CIcat(Igood_tri,:);
 
% sort traingle verteces by length of sides
[SDcat,SIcat] = sort(Dcat,2);
% build the sorted vertecs list
IndMat = repmat((1:1:size(SIcat,1)).',1,3);
SCIcat   = reshape(CIcat(sub2ind([size(SIcat,1),3],IndMat(:),SIcat(:))),size(SIcat,1),3);

XtCat = Xcat(SCIcat);
YtCat = Ycat(SCIcat);
NtriCat = size(XtCat,1);

    

%  matrix in which each column is the [x1;x2;x3;y1;y2;y3]
% of a triangle in the reference catalog
Ylsq = [XtRef.';YtRef.']; 

% for each random triangle in ListCat
H          = zeros(6,4);
H(1:3,1)   = 1;
H(4:6,2)   = 1;

%InPar.TriMaxResid = InPar.TriMaxResid./(RAD.*ARCSEC_IN_DEG);
%toc

%tic;
Ibest = 0;
%Counter = zeros(InPar.NrandTri,1);

ParBest = struct('Resid',{},'RMS',{},'MaxResid',{},'Par',{});
SearchCont = true;
Irand = 0;
Nrev  = size(InPar.RevTable,1);
while (SearchCont),
    Irand = Irand + 1;

   CatRatio12 = SDcat(Irand,1)./SDcat(Irand,2);
   CatRatio23 = SDcat(Irand,2)./SDcat(Irand,3);
   Iratio  = find(abs(RefRatio12-CatRatio12)<InPar.RatioErr & ...
                  abs(RefRatio23-CatRatio23)<InPar.RatioErr);
              
   %FlagRatio  = abs(RefRatio12-CatRatio12)<InPar.RatioErr & ...
   %             abs(RefRatio23-CatRatio23)<InPar.RatioErr;
                  
              
   H(:,3:4) = [XtCat(Irand,:).', -YtCat(Irand,:).' ; YtCat(Irand,:).', XtCat(Irand,:).'];

   for Irev=1:1:Nrev,
       Hr = H;
       Hr(1:3,3) = H(1:3,3).*InPar.RevTable(Irev,1);
       Hr(4:6,3) = H(4:6,3).*InPar.RevTable(Irev,2);
       %Hr(1:3,3:4) = InPar.RevTable(Irev,1).*H(1:3,3:4);
       %Hr(4:6,3:4) = InPar.RevTable(Irev,2).*H(4:6,3:4);
       
       
       % NOTE: use of inv here is faster than "\"
       %       use of find is faster than logical
       %       probably due to the typical size of the array
       Par = inv(Hr.'*Hr)*Hr.'*Ylsq(:,Iratio);
       Resid = Ylsq(:,Iratio) - Hr*Par;

       IcandTri = find(max(abs(Resid))<InPar.TriMaxResid);
       Icrit = IcandTri;
       if (~isempty(IcandTri)),
           Par    = Par(:,IcandTri);
           Resid = Resid(:,IcandTri);
           Theta  = atan2(Par(4,:),Par(3,:));
           Scale  = sqrt(Par(3,:).^2 + Par(4,:).^2);
           ShiftX = Par(1,:);
           ShiftY = Par(2,:);
           Icrit  = find(Theta>InPar.RotationRange(1) & ...
                         Theta<InPar.RotationRange(2) & ...
                         abs(Scale - InPar.Scale)<InPar.ScaleErr & ...
                         (ShiftX.^2 + ShiftY.^2)<InPar.MaxDistCenter.^2);

           for Ic=1:1:length(Icrit),      
               Ibest = Ibest + 1;
               ParBest(Ibest).Resid    = Resid(:,Icrit(Ic));
               ParBest(Ibest).RMS      = std(ParBest(Ibest).Resid);
               ParBest(Ibest).MaxResid = max(ParBest(Ibest).Resid);
               ParBest(Ibest).Par      = Par(:,Icrit(Ic));
               ParBest(Ibest).Reverse  = InPar.RevTable(Irev,:);
               Rot = [Par(3), -Par(4); Par(4), Par(3)].*diag(ParBest(Ibest).Reverse);
               ParBest(Ibest).T        = [Rot, [Par(1);Par(2)];0, 0, 1];
               %[Par(3), -Par(4), Par(1); Par(4), Par(3), Par(2); 0, 0, 1];

               % match all
               ParBest(Ibest).Match     = test_match(ListRef,ListCat,ParBest(Ibest).T,InPar.TestMatchDist);
               ParBest(Ibest).Nmatch    = length(ParBest(Ibest).Match);
               ParBest(Ibest).FracMatch = ParBest(Ibest).Nmatch./Ncat;
               ParBest(Ibest).stdX      = std([ParBest(Ibest).Match.DX]);
               ParBest(Ibest).stdY      = std([ParBest(Ibest).Match.DY]);

           end
       end
   end

   
   SearchCont = length(ParBest)<InPar.StopNumber && ...
                Irand<InPar.NrandTri && ...
                Irand<NtriCat;
     
end
% toc
   
if (~isempty(ParBest)),
    Igood = find([ParBest.FracMatch]>InPar.MinFracMatch & ...
                 [ParBest.RMS]<InPar.MaxRMS & ...
                 [ParBest.Nmatch]>InPar.MinNmatch);
    if (~isempty(Igood)),
        switch lower(InPar.SelectMethod)
            case 'minrms'
               [~,Isel] = min([ParBest(Igood).stdX].^2+[ParBest(Igood).stdY].^2);
               Isel = Igood(Isel);
            case 'maxnmatch'
               [~,Isel] = max([ParBest(Igood).Nmatch]);
               Isel = Igood(Isel); 
            case 'maxfracmatch'
               [~,Isel] = max([ParBest(Igood).FracMatch]);
               Isel = Igood(Isel);  
            
            otherwise
                error('Unknown SelectMethod option');
        end

    else
        Isel = [];
    end
else
    Igood = [];
    Isel  = [];
end

    
%ParBest
%AA=[ParBest.Par];
%Theta = atan2(AA(4,:),AA(3,:));
%Scale = sqrt(AA(3,:).^2 + AA(4,:).^2); % .*RAD.*3600;
%ShiftX = AA(1,:);
%ShiftY = AA(2,:);

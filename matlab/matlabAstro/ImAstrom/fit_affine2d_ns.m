function [FitRes]=fit_affine2d_ns(Cat,Ref)
%--------------------------------------------------------------------------
% fit_affine2d_ns function                                        ImAstrom
% Description: Fit an 2D affine transformation, non-simultanously to both
%              axes, to a set of control points.
%              The fit is of the form:
%              Xref = A*X - B*Y + C
%              Yref = D*X + E*Y + F
% Input  : - Matrix of the first set of control points (the "catalog")
%            [X, Y, ErrX, ErrY]. If ErrY is not provided, then
%            will assume ErrY=ErrX. If ErrX is not provided then
%            ErrY=ErrX=1.
%          - Matrix of the second set of control points (the "reference")
%            [X, Y, ErrX, ErrY]. If ErrY is not provided, then
%            will assume ErrY=ErrX. If ErrX is not provided then
%            ErrY=ErrX=1.
%            The reference coordinates are used as the "independent"
%            variable in the least square problem.
% Output : - Structure containing the best fit information.
%            Note that the RMS is in units of the reference image
%            coordinate units.
% See also: fit_affine2d_s.m
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Ref = rand(100,2).*1024;
%          Cat(:,1) =  Ref(:,1).*cosd(1)+Ref(:,2).*sind(1) + 10;
%          Cat(:,2) = -Ref(:,1).*sind(1)+Ref(:,2).*cosd(1) + 20;
%          [FitRes]=fit_affine2d_ns(Cat,Ref);
% Reliable: 2
%--------------------------------------------------------------------------

[Ncat,Ncatcol] = size(Cat);
[Nref,Nrefcol] = size(Ref);

if (Ncat~=Nref),
    error('Number of points in Ref and Cat must be identical');
end
N = Ncat;

% make sure Ref and Cat has four column each [X,Y,ErrX, ErrY]
if (Ncatcol==2),
    Cat = [Cat, ones(size(Cat))];
elseif (Ncatcol==3),
    Cat = [Cat, Cat(:,3)];
else
    % do nothing
end

if (Nrefcol==2),
    Ref = [Ref, ones(size(Ref))];
elseif (Nrefcol==3),
    Ref = [Ref, Ref(:,3)];
else
    % do nothing
end

    
% control points
Xcat     = Cat(:,1);
Ycat     = Cat(:,2);
Xref     = Ref(:,1);
Yref     = Ref(:,2);
ErrXcat  = Cat(:,3);
ErrYcat  = Cat(:,4);
ErrXref  = Ref(:,3);
ErrYref  = Ref(:,4);

% combine errors of Ref and Cat
ErrX     = sqrt(ErrXcat.^2 + ErrXref.^2);
ErrY     = sqrt(ErrYcat.^2 + ErrYref.^2);


% non simultanous fit
% fit X and Y separately
% design matrix
Hx = [Xcat, -Ycat, ones(N,1)];
Hy = [Xcat,  Ycat, ones(N,1)];
% fit
[ParX,ParErrX] = lscov(Hx,Xref,ErrX.^-2);
[ParY,ParErrY] = lscov(Hy,Yref,ErrY.^-2);
    
% residuals
FitRes.ResidX = Xref - Hx*ParX;
FitRes.ResidY = Yref - Hy*ParY;
FitRes.Resid  = [FitRes.ResidX; FitRes.ResidY];

% rms
FitRes.Xrms   = std(FitRes.ResidX);
FitRes.Yrms   = std(FitRes.ResidY);
FitRes.rms    = std(FitRes.Resid);
% chi2
FitRes.Chi2   = sum(FitRes.ResidX./ErrX) + sum(FitRes.ResidY./ErrY);
FitRes.N      = 2.*N;
FitRes.Nsrc   = N;
FitRes.Npar   = numel([ParX;ParY]);
FitRes.Ndof   = FitRes.N - FitRes.Npar;

FitRes.tdata.ParX = ParX;
FitRes.tdata.ParY = ParY;
FitRes.tdata.ParErrX = ParErrX;
FitRes.tdata.ParErrY = ParErrY;

% transformation
FitRes.Tran          = affine2d([[ParX(1) -ParX(2) ParX(3)]; [ParY(1), +ParY(2), ParY(3)]; 0 0 1].');


    
    
    
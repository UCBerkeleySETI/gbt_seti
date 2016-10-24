function [FitRes]=fit_affine2d_s(Cat,Ref)
%--------------------------------------------------------------------------
% fit_affine2d_s function                                         ImAstrom
% Description: Fit an 2D affine transformation, simultanously to both
%              axes, to a set of control points.
%              The fit is of the form:
%              Xref = A*X - B*Y + C
%              Yref = A*Y + B*X + D
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
% See also: fit_affine2d_ns.m
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    May 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: Ref = rand(100,2).*1024;
%          Cat(:,1) =  Ref(:,1).*cosd(1)+Ref(:,2).*sind(1) + 10;
%          Cat(:,2) = -Ref(:,1).*sind(1)+Ref(:,2).*cosd(1) + 20;
%          [FitRes]=fit_affine2d_s(Cat,Ref);
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
Hx = [Xcat, -Ycat, ones(N,1), zeros(N,1)];
Hy = [Ycat,  Xcat, zeros(N,1), ones(N,1)];
H  = [Hx;Hy];
Zref = [Xref;Yref];
Err = [ErrX; ErrY];
% fit
[Par,ParErr] = lscov(H,Zref,Err.^-2);
    
% residuals
FitRes.Resid  = Zref - H*Par;
FitRes.ResidX = FitRes.Resid(1:N);
FitRes.ResidY = FitRes.Resid(N+1:end);

% rms
FitRes.Xrms   = std(FitRes.ResidX);
FitRes.Yrms   = std(FitRes.ResidY);
FitRes.rms    = std(FitRes.Resid);
% chi2
FitRes.Chi2   = sum(FitRes.Resid./Err);
FitRes.N      = 2.*N;
FitRes.Nsrc   = N;
FitRes.Npar   = numel(Par);
FitRes.Ndof   = FitRes.N - FitRes.Npar;

FitRes.tdata.Par    = Par;
FitRes.tdata.ParErr = ParErr;

% transformation
FitRes.Tran          = affine2d([[Par(1) -Par(2) Par(3)]; [Par(2), +Par(1), Par(4)]; 0 0 1].');


    
    
    
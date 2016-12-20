function [TranRes]=fit_tran2d(Cat,Ref,varargin)
%--------------------------------------------------------------------------
% fit_tran2d function                                             ImAstrom
% Description: Fit a general transformation between two sets of
%              control points (catalog and reference).
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
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'FunTrans' - Handle for transformation function. Options are:
%                       @fit_affine2d_ns - fit affine 2D transformation
%                               non simultanously for both axes (default).
%            'FunPar' - Cell array of transformation function parameters.
%                       Default is {}.
%            'ErrCat' - A scalar value to multiply (or to use) as the
%                       catalog error. Default is 1.
%            'ErrRef' - A scalar value to multiply (or to use) as the
%                       reference error. Default is 1.
%            'MaxIter'- Maximum number of sigma clipping iterations.
%                       Default is 1.
%            'SigClipPar' - Cell array of parameters to pass to
%                       clip_resid.m. Default is
%                       {'Method','perc','Mean','median','Clip',[0.02 0.02]}.
%            'GoodRMS' - Maximum RMS below for which to do sigma cliping.
%                       Default is Inf.
% Output : - A structure array containing the best fit transformation.
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Jan 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [TranRes]=fit_tran2d(CP(:,1:2),CP(:,3:4));
%          Ref = rand(100,2).*1024;
%          Cat(:,1) =  Ref(:,1).*cosd(1)+Ref(:,2).*sind(1) + 10;
%          Cat(:,2) = -Ref(:,1).*sind(1)+Ref(:,2).*cosd(1) + 20;
%          [FitRes]=fit_tran2d(Cat,Ref,'FunTrans',@fit_general2d_ns);
% Reliable: 2 
%--------------------------------------------------------------------------

DefV.FunTrans    = @fit_affine2d_ns;
DefV.FunPar      = {};
DefV.ErrCat      = 1;    % scalar error
DefV.ErrRef      = 1;
DefV.MaxIter     = 1;    % 0 no iterations
DefV.SigClipPar  = {'Method','perc','Mean','median','Clip',[0.02 0.02]};
DefV.GoodRMS     = Inf;

InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

% size of Cat and Ref
[Ncat,Ncatcol] = size(Cat);
[Nref,Nrefcol] = size(Ref);

if (Ncat~=Nref),
    error('Number of points in Ref and Cat must be identical');
end
N = Ncat;

% make sure Ref and Cat has four column each [X,Y,ErrX, ErrY]
if (Ncatcol==2),
    Cat = [Cat, ones(size(Cat)).*InPar.ErrCat];
elseif (Ncatcol==3),
    Cat = [Cat, Cat(:,3)];
else
    % do nothing
end

if (Nrefcol==2),
    Ref = [Ref, ones(size(Ref)).*InPar.ErrRef];
elseif (Nrefcol==3),
    Ref = [Ref, Ref(:,3)];
else
    % do nothing
end


% assume the reference is the independent variable
TranRes = InPar.FunTrans(Cat,Ref,InPar.FunPar{:});
Iter = 0;
while (Iter<InPar.MaxIter && TranRes.Xrms>InPar.GoodRMS && TranRes.Yrms>InPar.GoodRMS),
    Iter = Iter + 1;
    [~,~,FlagX]=clip_resid(TranRes.ResidX,InPar.SigClipPar{:});
    [~,~,FlagY]=clip_resid(TranRes.ResidY,InPar.SigClipPar{:});
    Flag = FlagX & FlagY;
    Cat = Cat(Flag,:);
    Ref = Ref(Flag,:);
    TranRes = InPar.FunTrans(Cat,Ref,InPar.FunPar{:});
end

TranRes.Niter = Iter;






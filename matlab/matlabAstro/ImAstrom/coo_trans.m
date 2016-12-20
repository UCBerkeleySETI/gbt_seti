function [XYt,TranS]=coo_trans(XY,Fwd,varargin)
%--------------------------------------------------------------------------
% coo_trans function                                              ImAstrom
% Description: Coordinates transformation given a general transformation.
% Input  : - Two columns matrix of [X,Y] coordinates.
%          - A flag indicating if too applay forward transformation (true)
%            or invrese transformation (false). Default is true.
%          * A structure with the following fields (keywords) or
%            a cell array of pairs of arguments or pairs of
%            input arguments (...,key,val,...):
%            The following keywords are available:
%            'Shift' - [X, Y] shift. Default is [0 0].
%            'Rot'   - Rotation matrix. Default is [1 0;0 1].
%            'ParAng1' - [ParAng(rad), amplitudeX, amplitudeY].
%                      If two columns then amplitudeY=amplitudeX.
%            'ParAng2' - [ParAng(rad), amplitudeX, amplitudeY].
%                      If two columns then amplitudeY=amplitudeX.
%            'PolyX'  - Polynomial transformation for the X equation:
%                       [Coef DegX, DegY, X0, Y0, NormX, NormY].
%                       Line per polynomial.
%                       Default is [].
%            'PolyY'  - Like 'PolyX', but for the Y transformation.
%            'RadPolyX' - Radial polynomial transformation for the X equation:
%                       [Coef DegR, X0, Y0, NormR].
%                       Line per polynomial.
%                       Default is [].
%            'RadPolyY' - Like 'RadPolyX', but for the Y transformation.
%            'FunX'    - General function for the X transformation
%                       (linear in the parameters).
%                       The value is a cell array in which the first element
%                       is a function of the form alpha*FunX(X,Y,Pars).
%                       The rest of the elemnts are the function
%                       parameters. E.g., {@FunHandle,alpha,FunPars...}.
%            'FunY'   - Like 'FunX' but for the Y transformation.
% Output : - Two column matrix of transformed [X,Y] coordinates.
%          - Structure with transformation data.
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    May 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [XYt,T]=coo_trans([1,1],true,'Shift',[0 1],'PolyX',[0.5 2 1 500 500 1000 1000])
% Reliable: 2
%--------------------------------------------------------------------------

[~,X,Y]=fit_general2d_ns(XY,ones(size(XY)),varargin{:});
XYt = [X,Y];

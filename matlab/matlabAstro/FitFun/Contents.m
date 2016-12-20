%
% Contents file for package: FitFun
% Created: 29-Dec-2015
%---------
% bessel_icoef.m :  Calculate the Bessel interpolation coefficiant.
% bin_sear.m :  Binary search for a value in a sorted vector. If the value does not exist, return the closes index.
% bin_sear2.m :  Binary search for a value in a sorted vector. If the value does not exist, return the closes index.
% calc_hessian.m :  Calculate the Hessian (second derivative) matrix of a multivariable function.
% centermass2d.m :  Calculate the center of mass and second moments of 2-dimensional matrix.
% chi2_bivar_gauss.m :  Calculate the \chi^2 of a bivariate Gaussian with a data and error matrices: Chi2 = sumnd(((bivar_gauss(X,Y,Pars)-Data)/Error)^2))
% chi2_nonsym.m :  Given measurments with non-symetric Gaussian errors (i.e., upper and lower errors are not equal), calculate the \chi^{2} relative to a given model.
% chi2fit_nonlin.m :  Perform a non-linear \chi^2 fit to dataset.
% chi2fit_nonlin_stab.m :  Stab function for chi2fit_nonlin.m
% clip_resid.m :  Clip residuals using various methods including sigma clipping min/max, etc.
% conrange.m :  Given two vectors of Y and X, calculate the range in X that satisfies the constrain Y<=(min(Y)+Offset). This is useful for calculating likelihood/chi^2 errors for 1-D data.
% find_local_extramum.m :  Given table of equally spaced data, use Stirling interpolation formula to find the local extramums of the tabulated data. The program find all the local extramums between X(Deg/2) to X(end-Deg/2), where Deg is the degree of interpolation.
% find_local_zeros.m :  Given table of equally spaced data, use Stirling interpolation formula to find the local zeros of the tabulated data. The program find all the local zeros between X(Deg/2) to X(end-Deg/2), where Deg is the degree of interpolation.
% fit_2d_polysurface.m :  Fit a 2-D polynomial surface to z(x,y) data. e.g., Z= a + b.*X + c.*X.^3 + d.*X.*Y + e.*Y.^2.
% fit_circle.m :  Fit points, on a plane or a sphere, to a circle. Calculate the best fit radius and the center of the circle.
% fit_gauss1d.m :  Fit a 1-D Gaussian without background using fminsearch. The Gaussian form is: Y=A/(Sigma*sqrt(2*pi))*exp(-(X-X0).^2./(2.*Sigma^2))
% fit_gauss1da.m :  Fit a 1-D Gaussian without background using fminsearch. The Gaussian form is: Y=A/(Sigma*sqrt(2*pi))*exp(-(X-X0).^2./(2.*Sigma^2))
% fit_gauss2d.m :  Non-linear fitting of a 2D elliptical Gaussian with background.
% fit_gauss2dl.m :  Fit a 2-D Gaussian to data of the form f(x,y). f(x,y) = A*exp(-(a*(x-x0)^2+2*b*(x-x0)*(y-y0)+c*(y-y0)^2))
% fit_lin.m :  Fit a general function which linear in its free parameters.
% fit_pm_parallax.m :  Fit low-accuracy parallax and proper motion to a set of of celestial positions.
% fit_pow.m :  Fit power-law function of the form Y=A*X^Alpha to data using non linear lesat squares.
% fit_sn_rise.m :  Fit various functions appropriate for the rise of SN light curve. The fitted functions are: L_max*(1-((t-t_max)/t_rise)^2) L_max.*(1-exp(-(t-t_start)./t_rise)) L_max.*erfc(sqrt(t_diff./(2.*(t-t_start))))
% fitexp.m :  Fit an exponential function to data. The function has the form: Y = A * exp(-X./Tau).
% fitgauss.m :  Fit a Gaussian function to data, where the Gaussian has the form: Y = A * exp(-0.5.*((X-X0)./s).^2).
% fitgenpoly.m :  Fit general polynomials (e.g., normal, Legendre, etc) to data. The fitted function has the form: Y = P(N+1) + P(N)*f_1(X) + ... + P(1)*f_N(X). Where f_i(X) are the polynomial functions. For example, in the case of normal polynomials these are f_1(X) = X, f_2(X) = X^2, etc.
% fitharmo.m :  Fit trignometric functions with their harmonies to data. The fitted function is of the form: Y= a_1*sin(w1*t) + b_1*cos(w1*t) + a_2*sin(2*w1*t) + b_2*cos(2*w1*t) + ... a_n*sin(n_1*w1*t) + b_n*cos(n_1*w1*t) + ... c_1*sin(w2*t) + d_1*cos(w2*t) + ... s_0 + s_1*t + ... + s_n.*t.^n_s Note that w is angular frequncy, w=2*pi*f, while the use should indicate the frequency.
% fitlegen.m :  Fit Legendre polynomials to data, where the fitted function has the form: Y= a_0*L_0(X) + a_1*L_1(X) +...+ a_n*L_n(X) This function is replaced by fitgenpoly.m
% fitpoly.m :  Linear least squares polynomial fitting. Fit polynomial of th form: Y= a_0 + a_1*X + a_2*X^2 +...+ a_n*X^n to set of data points. Return the parameters, thir errors, the \chi^2 squars, and the covariance matrix. This function is replaced by fitgenpoly.m
% fitpow.m :  Fit a power-law function of the form: Y = A * X ^(Gamma), to data set.
% fitslope.m :  Linear least squares polynomial fitting, without the constant term. Fit polynomial of the form: Y= a_1*X + a_2*X^2 +...+ a_n*X^n to a set of data points. Return the parameters, their errors, the \chi^2, and the covariance matrix. This function is replaced by fitgenpoly.m
% fminsearch_chi2.m :  \chi^2 fitting using fminsearch.m and given a model function.
% fminsearch_my.m :  A version of the built in fminsearch.m function in which it is possible to pass additional parameters to the function, with no need for nested functions.
% fun_binsearch.m :  Given a monotonic function, Y=Fun(X), and Y, search for X that satisfy Y=F(X). The search is done using a binary search between the values stated at X range.
% get_fwhm.m :  Given a 1-D vector estimate the FWHM by measuring the distance between the two extreme points with height of 0.5 of the maximum height.
% interp_diff.m :  Interpolation of equally spaced data using high-order differences.
% interp_diff_ang.m :  Given a vector of time and a vector of coordinate on a sphere interpolate the spherical coordinate in a list of times. This function is talking into account [0..2pi] angels discontinuity.
% lin_fun.m :  Evaluate a cell array of functions, which are linear in the free parameters: i.e. Y=A(1)*Fun{1} + A(2)*Fun{2} + ...
% ls_conjgrad.m :  Solve a linear least squares problem using the conjugate gradient method.
% mfind_bin.m :  Binary search on a vector running simolutnously on multiple values. A feature of this program is that it you need to add 1 to the index in order to make sure the found value is larger than the searched value.
% polyconf_cov.m :  Estimate the 1-sigma confidence interval of a polynomial given the polynomial best fit parameters and its covariance matrix. REQUIRE FURTHER TESTING.
% polyfit_sc.m :  Fitting a polynomial, of the form Y=P(1)*X^N + ... + P(N)*X + P(N+1) to data with no errors, but with sigma clipping: 
% polysubstitution.m :  Given a polynomial [a_n*X^n+...+a_1*x+a_0] coefficients [a_n, a_n-1,..., a_1, a_0] and a the coefficients of a linear transformation of the form X=A_1*Z+A_0 [A_1, A_0], substitute the second polynomial into the first and find the new coefficients. This function is being used by fitgenpoly.m to change the variables in polynomials.
% polysurface_fit.m :  Fit a surface using a 2-D polynomials.
% polysval.m :  Evaluate multiple polynomials at multiple points. Similar to polyval.m but allows a matrix of polynomials coefficients in which each rows is a different polynomial coefficients. The first column in P corresponds to the coefficient of the highest power and the last column for the power of zero.

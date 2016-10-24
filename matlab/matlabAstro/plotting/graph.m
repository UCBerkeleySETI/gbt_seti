function H=graph(X,varargin);
%------------------------------------------------------------------------------
% graph function                                                      plotting
% Description: Given a two column matrix, plot the second column as a
%              function of the first column. The function may get additional
%              argument like the plot command.
% Input  : - Matrix containing at least two columns.
%          * Arbitrary number of parameters to send to the plot command.
% Output : - Handle for the objects plotted.
% Plot   : A plot of the second column of the input matrix as a
%          function of the first column.
% Tested : Matlab 3.5
%     By : Eran O. Ofek                  December 1993
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
% Example: graph(rand(10,2));
% Reliable: 1
%------------------------------------------------------------------------------

H=plot(X(:,1),X(:,2),varargin{:});

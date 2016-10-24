%
% Contents file for package: plotting
% Created: 29-Dec-2015
%---------
% axis01.m :  Create an axis that have only an the 0 X and Y axes.
% cell_contourc.m :  A contourc-like program that return a cell array of the contour levels.
% contour_percentile.m :  Given a matrix, generate a contour plot in which the contours represent the percentiles of the total matrix sum. I.e., the region encompassed within the contour contains the given percentile of the matrix total sum.
% date_axis.m :  Given a graph in which the lower x-axis shows the Julian day, add a date-axis in the upper x-axis that corresponds to the Julian day.
% draw_iline.m :  Draw line interactively. Click the mouse right bottom in start point and at end point. Click left mouse button to exit this program.
% errorxy.m :  Plot graphs with error bars in both axes. 
% generate_colors.m :  Generate equally spaced N colors from a given color map.
% ginput_multi.m :  Get multiple ginput coordinates, for arbitrary number of points, using the mouse. Use left click to select multiple coordinates, and right click to abort.
% gno.m :  Get the nearest object handle. The user click near a a figure objects, and the function return their handles and coordinates. Left click for object selection. Right click for exiting program.
% graph.m :  Given a two column matrix, plot the second column as a function of the first column. The function may get additional argument like the plot command.
% hist_ofplot.m :  Given an existing plot, calculate histograms of the X and Y axis of the data in current figure axis. The x-axis histogram is displayed below the current plot, and the y-axis histogram is dispalyed on the right side.
% insert_image.m :  Insert an image to a matlab figure in a given position.
% invy.m :  Invert the y-axis of the current axis. This is a shortcut command to axis('ij') and set(gca,'YDir','Inverse').
% multi_axis.m :  Create additional x or y axis which is related to the exsiting axis by a given function.
% patch_band.m :  Given X and Y positions describing a line, plot a band (i.e., using the patch command), around the line, where the patch is defined to have a given height below and above the line.
% plot_compass.m :  Plon a compass on a plot or image.
% plot_corner.m :  Plot corners which lines are parallel to the axes.
% plot_cs.m :  Plot colored symbols.
% plot_ellipse.m :  Plot an ellipse with given properties.
% plot_int.m :  Plot a 2-dimensional graph and interactively remove and restore data points from the plot.
% plot_int1.m :  Given a plot wait for the use to click a keyboard key or a mouse click on the plot, and return the key and click position.
% plot_invchildren.m :  Invert the order of the childrens under a given handle.
% plot_lineprof.m :  Clicking on two points in current image the script return the intensity as function of position along the line between the two points.
% plot_rm.m :  Plot a 2-dimensional graph and interactively remove and restore data points from the plot. This function is being replaced by plot_int.m
% plot_scale.m :  Add a scale bar on a plot or image.
% plot_select_points.m :  Given an open plot, and X/Y vectors, let the use select using the left-mouse-click arbitrary number of points. For each points return the clicked position, and the nearest in the [X,Y] vectors. Right-mouse click to terminate. Use middle click to remove a point.
% plot_slit.m :  Plot a slit (box) on a plot or image.
% plotline.m :  Plot a line given the position of its start point, length, and angle as measured in respect to the x-axis.
% quiver1.m :  An improved quiver function. Allows to control the arrows properties and shape.
% subplot1.m :  An improved subplot function. Allows to control the gaps between sub plots and to define common axes.
% subplot1c.m :  Given a matrix with N-columns show subplots, of all the combinations of one column plotted against another column. Also return the correlations matrix between all the columns.
% textrel.m :  Write a text in current axis, where the position of the text is specified as a relative position in respect to the axis limits.

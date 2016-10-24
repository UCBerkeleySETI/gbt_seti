%
% Contents file for package: General
% Created: 29-Dec-2015
%---------
% and_mat.m :  Perform logical and operation between all the columns or rows of a matrix.
% and_nan.m :  Logical function "and" for NaNs. This function is similar to "and" logical function, but NaNs are regarded as no information using the following logical table: M1 M2 Result 1 1 1 1 0 0 0 1 0 0 0 0 NaN 1 1 NaN 0 0 1 NaN 1 0 NaN 0 NaN NaN 1
% array_select.m :  Given a matrix, select lines which their column fullfill a specific criteria. For example, the values in the second columns are in some range or follow a specific criterion.
% assoc_range.m :  Given a vector of data points and a vector of edges, for each value in the vector,return the index of the bin (defined by the edges) to which it belongs. 
% bivar_gauss.m :  Return the value of a normalized bivariate (2-dim) Gaussian: F = Const + Norm/(2*pi*SigmaX*SigmaY*sqrt(1-Rho^2)) * exp(-1/(2*(1-Rho^2))*((X-X0)^2/SigmaX^2 + (Y-Y0)^2/SigmaY^2 - 2*Rho*(X-X0)*(Y-Y0)/ (SigmaX*SigmaY)))
% catfile.m :  Concatenate files into a single file. Add a carriage return at the end of each Concatenated file.
% chebyshev_poly.m :  Evaluate Chebyshev polynomials
% check_range.m :  Given the size of an N-dimensional array, and minimum and amximum indices per each dimension, check if the indices are out of bounds. If the minimum or maximum indices are out of bounds return a new minimum and maximum indices that are in bound and nearest to the original indices.
% circ_2d.m :  Calculate a circule in 2-D (i.e., cylinder).
% construct_fullpath.m :  Construct a full path string to a program name, given its name, path or relative path and base path.
% construct_keyval_string.m :  Construct a ...,keyword, value,... string from pairs of parameters.
% convert_energy.m :  Convert between different energy units.
% convert_temp.m :  Convert between temperature systems.
% convert_units.m :  Unit conversion function. Given an input and output strings containing unit names, return the conversion multiplication factor needed for converting the input units to the output units. The user is responsible for the balance of the transformation. Type of units: Length: 'mm' ; 'cm' ; 'inch' ; 'feet' ; 'm' - meter ; 'km' ; 'mile' ; 'erad' - Earth radius ; 'au' ; 'ly' - light year ; 'pc'; 'yard' Time: 's' ; 'min' ; 'hour' ; 'day'; 'sday' - sidereal day ; week ; 'year'; 'cen' - century Mass: 'gr'; 'kg'; 'emass' - Earth mass; 'jmass' - Jupiter mass; 'smass' - Solar mass; 'mp' - proton mass; 'me' - electron mass; 'libra';'pound' Energy: (see also convert_energy.m) 'erg'; 'J' Angle: 'rad' ; 'deg' ; 'amin' (or 'arcmin') - arcmin ; 'asec' (or 'arcsec') - arcsec Solid Angle: 'ster' ; 'sdeg' - square degree ; 'smin' - square arcmin ; 'ssec' - square arcsec
% copy_files_from_dirtree.m :  Given a location (or the present working directory), look for all files in the subdirectories and copy them to the main directory.
% create_list.m :  Create a file and a cell array containing a list of files. The list is created from a cell array, or file name with wildcards.
% cross1_fast.m :  cross product of two 3-elements vectors. This is a fast version of the cross.m function. This function will work only between two vectors.
% cross_fast.m :  cross product of two 3-columns matrices. This is a fast version of the cross.m function. 
% date_str2vec.m :  Convert a string or a cell array of string containing date and time in the format 'YYYY-MM-DD HH:MM:SS.frac' or 'YYYY-MM-DD', to a matrix of dates with the following columns [Y M D H M S].
% delete_cell.m :  Delete a list of files listed in a cell array.
% delete_ind.m :  Delete a column/s or row/s from a specific position in a matrix.
% derivative.m :  Numerical derivative of a row vector.
% epar.m :  Allow the user to edit key/val parameters of functions. All the functions which use the set_varargin_keyval.m function will save their parameters into a file: ~matlab/.FunPars/<FunName>_PAR.mat. The user then can change the default values of these parameters by "epar <FunName>".
% eq_sampling.m :  Given two lists, each contains [X,Y], equalize the sampling frequency of the two lists by interpolating both lists to at a specified X.
% fermi_fun.m :  Calculate the Fermi function (Dermi-Dirac statistics) of the form 1/(1+exp((x-x0)/DX)).
% file2str.m :  Read the content of a file into a string or cell vector (line per element).
% filter_fft1.m :  Filter a 1D equally spaced series using FFT.
% find_peak.m :  Given a tabulated function [X,Y], find the maximum near a given X0.
% find_peak_center.m : NaN
% find_ranges.m :  Given a vector and several ranges, return the indices of values in the vector which are found within one of the ranges.
% find_ranges_flag.m :  Given a vector and several ranges, return the a vector indicating if a given position in the input vector is included in one of the ranges.
% find_strcmpi.m :  find(strcmpi(varargin{:})) function. I.e., like strcmpi.m but returning the indices of true.
% findmany.m :  Find all values in a vector in another vector or matrix.
% flag2regions.m :  Given a column vector flags (true/false) returns pairs of indices of the positions of continuus regions in which the flag is true.
% for_each_file.m :  Given a file name containing list of files, load each file into a matrix and execute a function with the loaded matrix as a paramter.
% fpf.m :  Easy to use fprintf, with automatic formatting. This function is similar to fprintf, but (i) open and close the file automaticaly; (ii) in case that format string is not given then the function try to select a format string automaticaly based on the precision of each number.
% fprintf_cell.m :  An fprintf command for a cell vector.
% fun_gauss2d.m :  Calculate bivariate Gaussian in a 2-D grid. This function is appropriate for minimization as the first input argument is the vector of free parameters.
% fun_template.m :  Generate a functionm template with help section and basic optional commands.
% gauss_2d.m :  Calculate bivariate Gaussian in a 2-D grid.
% get_constant.m :  Get the value of an astronomical/physical constant.
% get_constant1.m :  Get the value of an astronomical/physical constant (old version). See also get_constant.m
% group_cellstr.m :  Given a cell array of strings create a cell array of indices of distinct strings (see example).
% hour_str2frac.m :  Convert a string or cell array of strings containing the hour in format HH:MM:SS.frac' to fraction of day.
% image2avi.m :  Create avi file from list of images
% ind_cell.m :  Given a cell vector in which each element contains a vector of the same length and a vecor of indices, return a new cell array of the same size in which each element contains a vecor of only the elements which indices are specified (see example).
% index_outofbound.m :  Given a vector of indices and the allowed size of the array will remove from the vector of indices any indices which are out of bound.
% insert_ind.m :  Insert a column/s or row/s to a specific position in a matrix.
% int2d.m :  Numerically interagte a 2-D matrix.
% integral_percentile.m :  Given a numerically tabulated function (X,Y) and a percentile P, find the limits (a,b) such that int_{a}^{b}(Y(X)dx)=P (i.e., the integral of Y(X) from a to b equal P.
% interp1_nan.m :  Interpolate over NaNs in 1-D vector.
% interp1_sinc.m :  Interpolation of a 1-D array using the Whittaker–Shannon interpolation formula (i.e., sinc interpolation).
% interp2fast.m :  A faster version of the interp2.m built in function. This function is faster than interp2.m when the XI and YI vectors span over a small range in the VecX and VecY space.
% is_evenint.m :  Check for each integer number in array if even.
% isempty_cell.m :  Given a cell array, return a matrix of flags indicating if each one of the cells is empty.
% isfield_notempty.m :  Check if a field exist in a structure and if it is not empty.
% isnan_cell.m :  Given a cell array, return a matrix of flags indicating if each one of the cells is nan.
% lanczos_2d.m :  Calculate lanczos function in 2-D.
% latex_table.m :  Create a latex table from a data given in a cell array.
% list2vec.m :  Given an arbitrary number of arguments, each containing a vector, concatenate all vectors to a single vector.
% load2.m :  load a mat file containing a single variable to a variable name (rather than a structure, like load.m). If multiple variables are returned then will behave like load.m
% load_check.m :  Load a matlab variable or file from disk (similar to the load.m command). However, before the variable is loaded the function checks if the variable with name identical to the file name is already exist in the matlab main workspace. If it is exist it will copy the variable from the workspace. If variable does not exist it will load it in the usual way and store it in the main workspace. This is usefull when you want to load big variables in multiple function calles.
% loadh.m :  load a matrix from HDF5 file. If dataset name is not provided than will read all datasets into a structure. This function doesn't support groups. This is becoming faster than matlab (2014a) for matices with more than ~10^4 elements.
% lpar.m :  List user default parameters for a function (see epar.m).
% maskflag_check.m :  Given a matrix or vector of bit masks and a list of bits to test, return true for indices in the matrix in which one of the bits specified in the list of bits is open.
% maskflag_set.m :  Given a matrix or vector of bit masks, set specific bits of specific indices.
% mat2vec.m :  Convert matrix to vector.
% match_lists_index.m :  Match the lines in two matrices according to the values in one of the columns in each matrix.
% median_sigclip.m :  A robust median calculation, by removing lower and upper percentiles prior to the median calculation.
% nangetind.m :  Get elements from an array by its indices. However, unlike A(I,J) operation in matlab if I or J are out of bound then return NaN.
% nanmedfilt1.m :  One dimensional median filter that ignores NaNs.
% nickel56_decay.m :  Calculate the energy production of Nicel56->Cobalt->Iron radioactive decay as a function of time.
% nlinfit_my.m :  A version of the built in nlinfit.m function in which it is possible to pass additional parameters to the function, with no need for nested functions.
% openb.m :  open a matlab function in matlab editor (like the open command). Before the file is opened it is backuped in a backup (default is 'old') directory in the file directory under the name FileName.current_date
% or_nan.m :  Logical function "or" for NaNs. This function is similar to "or" logical function, but NaNs are regarded as no information using the following logical table: M1 M2 Result 1 1 1 1 0 1 0 1 1 0 0 0 NaN 1 1 NaN 0 0 1 NaN 1 0 NaN 0 NaN NaN 1
% prctile1.m :  calculate the percenile of a sample in a vector. This is similar to the builtin function prctile.m but limited to vector inputs and somewhat faster.
% q_fifo.m :  Implementation of a first-in-first-out queue (FIFO). The queue is implemented using a cell array and each element in the queue can be of any matlab type.
% quad_my.m :  A version of the built in quad.m function in which it is possible to pass additional parameters to the function, with no need for nested functions.
% read_formatted.m :  Read text/data file with constant format (i.e., columns at specfied locations). The start and end column for each entry should be specified.
% read_ipac_table.m :  
% read_str_formatted.m :  Read text/data string with constant format (i.e., columns at specfied locations). The start and end column for each entry should be specified.
% remove_cell_element.m :  Remove a list of indices from a cell vector.
% saveh.m :  Save a matrix into HDF5 file. If file exist then will add it as a new dataset to the same file. This is becoming faster than matlab (2014a) for matices with more than ~10^5 elements.
% set_varargin_keyval.m :  The purpose of this program is to handle a list of pairs of keywords and values input arguments. The program is responsible to check if the keywords are valid, and if a specific keyword is not supplied by the user, then the program will set it to its default value.
% sn_calc.m :  A signal-to-noise calculator for astronomical telescopes. Calculate S/N or limiting magnitude and field of view properties.
% sn_cooling_rw.m :  Calculate the shock cooling light curve (following the shock breakout) of a supernova, based on the Rabinak & Waxman (2011) model.
% sort_numeric_cell.m :  sort each row or columns in a cell array of numbers. see also: sortrows_numeric_cell.m
% sort_struct.m :  Sort in ascending order all the elements in a structure by one fields in the structure. 
% sp_powerlaw_int.m :  Calculate the value the spherical integral and the line integral of a broken power law of the form: rho = K R^(-W1) R<=R0 rho = K R0^(W2-W1) R^(-W2) R>R0.
% spacedel.m :  Given a string, recursively delete all spaces.
% spacetrim.m :  Given a string, recursively replace any occurance of two spaces with a single space, such that the final product is a string with a single spaces between words.
% sprintf2cell.m :  Generate a cell array of strings using the sprintf function, where the sprintf arguments, per string in cell is taken from a row in a matrix.
% star_ang_rad.m :  Empirical angular radii of stars based on their magnitude and colors.
% stellar_imf.m :  Return the stellar initial mass function in a given mass range.
% str2double_check.m :  Convert strings to doubles, but unlike str2double if the input is a number will return the number (instead of NaN).
% str2num_nan.m :  Convert string to number, and return NaN if not a number or empty.
% str_duplicate.m :  Duplicate a string multiple times.
% strcmp_cell.m :  Given two cell arrays of strings, check if each one of the strings in the first cell array exist in the second cell array.
% strlines2cell.m :  Given a string with end of line characters, break the string into a cell array in which each cell contains a line.
% struct2keyvalcell.m :  Given a structure convert the field names and their values to a cell array of ...,key,val,... arguments.
% struct2varargin.m :  Given a structure, prepare a cell array of all the field_names, field_values,... This is useful for converting InPar to varargin input.
% struct_def.m :  Define a structure array of a specific size with fields specified in a cell array of names.
% structcon.m :  Concatenate two structures into one. Example: S1.A S1.B, and S2.A, S2.B: S=structcon(S1,S2,1); will return a structure S, with fields A and B which contains the content of [S1.A;S2.A] and [S1.B;S2.B], respectively. The concatantion can be done along the 1st or 2nd dimensions.
% structcut.m :  Given a structure and a vector of indices, select from each field in the structure only the rows in each field which are specified by the vector of indices.
% subtract_back1d.m :  Subtract background level from a 1-D vector.
% sum_bitor.m :  Given a 2D array of integers, perform a bitor operation on all lines or rows and return a vector ofbit-wise or.
% summatlevel.m :  Given a matrix and a level value, return the sum of all the values in the matrix which are larger than the level.
% superdir.m :  A version of the matlab 'dir' function that can deal with more sophisticated types of wild cards. For example searching for: 'l*00[19-21].fits'.
% system_list.m :  Run the system command for a list of files.
% systemarg.m :  Running the UNIX system command.
% trapzmat.m :  Trapezoidal numerical integration on columns or rows of matrices. Contrary to trapz.m, the X input for this function can be a matrix.
% triangle_2d.m :  Calculate triangle in 2-D (i.e., cone).
% unique_cell_grouping.m :  Given a cell matrix containing eithr strings or numeric values, find unique lines in the cell matrix.
% unique_count.m :  Select unique values in numeric vector and count the number of apperances of each value.
% user_name.m :  Get the current user name.
% wcl.m :  Count the number of lines in a file.
% which_dir.m :  Return the directory in which a matlab program resides. This program is a version of "which.m" that trim the program name from the full path.
% wmedian.m :  Weighted median for a vector. Calculates the weighted median of a vector given the error on each value in the vector.
% xcorr_fft.m :  cross correlation of two 1-D serieses using FFT. The shortest vector will be padded by zeros (at the end).
% xcorr_fft_multi.m :  cross correlation of a 1-D series with multiple 1-D serieses using FFT. The shortest vector will be padded by zeros (at the end). This function return only the best correlation (and its shift) among all the columns in the multiple 1-D serieses.

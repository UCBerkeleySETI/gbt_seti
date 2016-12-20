%
% Contents file for package: www
% Created: 29-Dec-2015
%---------
% cgibin_parse_query_str.m :  Break a URL parameters query string to parameter names and values.
% find_urls.m :  Given a URL, read the URL content and extract all the links within the URL and return a cell array of all the links. Optionaly, the program can filter URL using regular expressions.
% ftp_dir_list.m :  Given an FTP URL that contains only a file listing, return a cell array of all the files URLs.
% html_page.m :  Create an HTML file. The file contains the necessery header and footer and a supplied content.
% html_table.m :  Given a matlab matrix or cell array create an html page with the matrix in an html table.
% install_astro_matlab.m :  Install or update the astronomy and astrophysics package for matlab.
% install_eran_matlab.m :  install or update the astronomy and astrophysics packages for matlab.
% mwget.m :  A wrapper around the wget command. Retrieve a URL using the wget command. See also pwget.m, rftpget.m.
% parse_html_table.m :  Parse columns from an HTML table into matlab. The program can not parse tables with colspan parameter different than 1.
% pwget.m :  Parallel wget function designed to retrieve multiple files using parallel wget commands. If fast communication is available, running several wget commands in parallel allows almost linear increase in the download speed. After exceuting pwget.m it is difficult to kill it. In order to stop the execuation while it is running you have to create a file name 'kill_pwget' in the directory in which pwget is running (e.g., "touch kill_pwget").
% rftpget.m :  A wrapper around the wget command designed to recursively retrieve the entire directory tree in an FTP site. See also: mwget.m, pwget.m, find_urls.m

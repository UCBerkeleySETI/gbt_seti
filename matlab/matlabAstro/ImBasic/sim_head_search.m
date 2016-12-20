function [SubHead,Sim]=sim_head_search(Sim,String,varargin)
%--------------------------------------------------------------------------
% sim_head_search function                                         ImBasic
% Description: Search for substring in Sim/FITS image headers.
% Input  : - List of FITS/SIM images - see images2sim.m for options.
%          - Substring or regular expression to search in header keyword
%            or comment columns.
%          * Arbitrary number of pairs of arguments: ...,keyword,value,...
%            where keyword are one of the followings:
%            'ReadImage' - Read image from FITS file. Default is false.
%            --- Additional parameters
%            Any additional key,val, that are recognized by one of the
%            following programs:
%            images2sim.m, image2sim.m
% Output : - Strructure array containing for each image the header lines
%            that contain the substring.
%          - SIM images.
% See als: cell_fitshead_search.m
% License: GNU general public license version 3
% Tested : Matlab R2014a
%     By : Eran O. Ofek                    Apr 2015
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: SubHead=sim_head_search('*.fits','GAIN')
% Reliable: 2
%--------------------------------------------------------------------------

HeaderField     = 'Header';
NameField       = 'ImageFileName';

DefV.ReadImage          = false;
InPar = set_varargin_keyval(DefV,'n','use',varargin{:});

Sim = images2sim(Sim,varargin{:},InPar.ReadImage);
Nim = numel(Sim);

SubHead = struct_def({HeaderField},Nim,1);
for Iim=1:1:Nim,
    fprintf('\n Search image number %d out of %d \n FileName: %s \n',Iim,Nim,Sim(Iim).(NameField));
    SubHead(Iim).(HeaderField) = cell_fitshead_search(Sim(Iim).(HeaderField),String,varargin{:});
end




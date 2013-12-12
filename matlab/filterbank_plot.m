function output=filterbank_plot(file, chanstart, chanend, tstart, tend)

fcenter = 1420;


[status, result] = unix(['header ',file, ' | grep -v Name']);

    
%sscanf(result, ['%s' ': ' '%s'])
%pause
out = (regexp(result, ': .*', 'match', 'dotexceptnewline'))

fil.file = out{1}(3:end)
fil.header = str2num(out{2}(3:end))
fil.size = str2num(out{3}(3:end))
fil.type = out{4}(3:end)
fil.telescope = out{5}(3:end)
fil.machine = out{6}(3:end)
%fil.name = out{7}(3:end)
fil.ra = out{7}(3:end)
fil.dec = out{8}(3:end)
fil.freq = str2num(out{9}(3:end))
fil.bw = str2num(out{10}(3:end))
fil.nchan = str2num(out{11}(3:end))
fil.nbeam = str2num(out{12}(3:end))
fil.beamnum = str2num(out{13}(3:end))
fil.mjd = str2num(out{14}(3:end))
fil.date=out{15}(3:end)
fil.tsamp=str2num(out{16}(3:end))
fil.nsamp=str2num(out{17}(3:end))
fil.tobs=str2num(out{18}(3:end))
fil.nbits=str2num(out{19}(3:end))
fil.ifs=str2num(out{20}(3:end))


if (~exist('chanstart', 'var'))
    chanstart = 1;
end

if (~exist('chanend', 'var'))
    chanend = fil.nchan;
end

if (~exist('tstart', 'var'))
    tstart = 1;
end

if (~exist('tend', 'var'))
    tend = fil.nsamp;
end

if abs(fil.bw) < 0.001
    fil.bw = -1.0/fil.tsamp*3;
end

fil.freq = fil.freq + (fil.bw * (chanstart-1))

chans = chanend-(chanstart-1);
samps = (tend - (tstart - 1));

fid = fopen(file);
header = fread(fid, fil.header, '*uint8');

if fil.nbits == 8
	
	
	if (samps ~= fil.nsamp)
		fseek(fid, fil.nchan * (tstart - 1), 0);	
	end
	
	if chanstart > 0
		data = fread(fid, chanstart - 1, '*uint8');
	end
	
	[num2str(chans),'*uint8']
	nskip = fil.nchan - chans;
	data = fread(fid, [chans, samps], [num2str(chans),'*uint8'], nskip);


elseif fil.nbits == 32


	if (samps ~= fil.nsamp)
		fseek(fid, fil.nchan * (tstart - 1) * 4, 0);	
	end
	
	if chanstart > 0
		data = fread(fid, chanstart - 1, '*single');
	end
	
	[num2str(chans),'*single']
	nskip = fil.nchan - chans;
	data = fread(fid, [chans, samps], [num2str(chans),'*single'], nskip*4);

end
fclose(fid);

fil.data = fliplr(data');



seti_freq_time_plot(fil)

output=fil;


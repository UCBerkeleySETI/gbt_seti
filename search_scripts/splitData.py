from math import ceil
from sys import argv
import os, sys, re

args = { 'chunk' : 10, 'template' : None}

if __name__ == "__main__":

	# Process command-line arguments
	if len(sys.argv) < 2:
		print "Not enough arguments!"
		print "python SplitData.py filename template chunk=x"
		sys.exit(0)

	elif not os.path.exists(argv[1]):
		print "Input file does not exist"
		sys.exit(0)

	args['filename'] = sys.argv[1]
	args['template'] = sys.argv[2]

	for item in sys.argv[3:]:
		ind = item.find('=')
		if ind > 0:
			args[item[:ind]] = eval(item[ind + 1:])

	for k, v in args.iteritems():
		globals()[k] = v

	# Read associated inf file
	infPath = filename[:-3] + 'inf'
	if not os.path.exists(infPath):
		print "%s does not exist" % infPath
		sys.exit(0)

	# Extract required information
	inffile  = open(infPath, 'r')
	inf      = inffile.read()
	tsamp    = float(inf.split('\n')[10].split('=')[1])
	nsamp    = int(inf.split('\n')[9].split('=')[1])
	mjd      = float(inf.split('\n')[7].split('=')[1])
	chunkLen = int(chunk / tsamp)
        if chunkLen % 2 == 1:
            chunkLen = chunkLen + 1

	# Perform the split
	os.system("split -d -b %d %s %s" % (chunkLen * 4, filename, template))

	# Rename files and create inf files for each chunk
	for i in range(0, int(ceil(nsamp / chunkLen))):

		# Rename file
		if i <= 9:
			name = template + '0' + str(i)
		else:
			name = template + str(i)
		os.system('mv %s %s' % (name, name + '.dat') )

		# Create inf file
		text = inf.replace(str(nsamp), str(chunkLen))
		text = text.replace(str(mjd), str(mjd + (i * chunkLen * tsamp) / (24.0 * 60 * 60)))
		f = open(name + '.inf', 'w')
		f.write(text)
		f.close()

	# Process last file
#	i = int(ceil(nsamp / chunkLen))
#	if i < 9:
#		name = template + '0' + str(i)
#	else:
#		name = template + str(i)

#	os.system('mv %s %s' % (name, name + '.dat') )
#	text = inf.replace(str(nsamp), str(nsamp - i * chunkLen))
#	text = text.replace(str(mjd), str(mjd + (i * chunkLen * tsamp) / (24.0 * 60 * 60)))
#	f = open(name + '.inf', 'w')
#	f.write(text)
#	f.close()


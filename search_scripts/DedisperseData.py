from subprocess import Popen, PIPE, STDOUT
from datetime import datetime
from sys import argv
from os import path
import os, re

# Executable locations
mdsmPipeline = "/home/lessju/Code/MDSM/release/pelican-mdsm/pipelines/sigprocPipeline"
decimate     = "/home/lessju/Code/SETI/decimate"
sheader      = "/home/lessju/Code/sigproc_orig/header"
logfile      = "logfile.txt"

# Define search parameters here
startDM  = [0,    107.52, 187.52, 353.920, 833.920, 1537.920, 3073.920]		
numDMs   = [5376, 1600,   1664,   1600,    1408,    1536,     512]
dmStep   = [0.02, 0.05,   0.1,    0.3,     0.5,     1,        2]
downsamp = [1,    2,      4,      8,       16,      32,       64]

performDownsamples = 7
nsamp              = 65536

def createPelicanConfigFile(configPath, inputFile, mdsmConfig, outputPrefix, tsamp):
	""" Creatre pelican configuration file"""

	string = """<?xml version="1.0" encoding="UTF-8"?>\n\
	<!DOCTYPE pelican>\n\n\
	<configuration version="1.0">\n\
	<pipeline>\n\
	\t<buffers>\n\
	\t\t<SpectrumDataSetStokes>\n\
	\t\t\t<buffer maxSize="10000000" maxChunkSize="10000000"/>\n\
	\t\t</SpectrumDataSetStokes>\n\
	\t</buffers>\n\n\
	\t<adapters>\n\
	\t\t<SigprocAdapter>\n\
	\t\t\t<sampleSize bits="8"/>\n\
	\t\t\t<subbands number="2048" />\n\
	\t\t\t<samplesPerRead number="1024" />\n\
	\t\t</SigprocAdapter>\n\
	\t</adapters>\n\n\
	\t<clients>\n\
	\t\t<FileDataClient>\n\
	\t\t\t<data type="SpectrumDataSetStokes" adapter="SigprocAdapter" file="%s"/>\n\
	\t\t</FileDataClient>\n\
	\t</clients>\n\n\
	\t<modules>\n\
    \t\t<RFI_Clipper active="active" channelRejectionRMS="10.0" spectrumRejectionRMS="6.0">\n\
	\t\t\t<zeroDMing active="true" />\n\
	\t\t\t<BandPassData file="/home/lessju/Code/SETI/GBTBandPass.bp" />\n\
	\t\t\t<Band matching="true" startFrequency="1899.804688" endFrequency="1100"/>\n\
	\t\t\t<History maximum="10000"/>\n\
   	\t\t</RFI_Clipper>\n\n\
	\t\t<MdsmModule>\n\
	\t\t\t<observationfile filepath="%s" />\n\
	\t\t\t<createOutputBlob value="1" />\n\
	\t\t\t<invertChannels value="0" />\n\
	\t\t</MdsmModule>\n\
	\t</modules>\n\n\
	\t<output>\n\
	\t\t<dataStreams>\n\
	\t\t\t<stream name="DedispersedTimeSeriesF32" listeners="DedispersedDataWriter"/>\n\
	\t\t</dataStreams>\n\
	\t\t<streamers>\n\
	\t\t\t<DedispersedDataWriter>\n\
	\t\t\t\t<sigprocFile filename="%s" />\n\
	\t\t\t\t<file prefix="%s" postfix=".dat"/>\n\
	\t\t\t\t<topChannelFrequency value="1899.804688" />\n\
	\t\t\t\t<frequencyOffset value="800" />\n\
	\t\t\t\t<samplingTime value="%fe-06"/>\n\
	\t\t\t\t<DMs values="all" />\n\
	\t\t\t\t<headerless value="1" />\n\
	\t\t\t</DedispersedDataWriter>\n\
	\t\t</streamers>\n\
	\t</output>\n\
	</pipeline>\n\
	</configuration> """ %  (inputFile, mdsmConfig, inputFile, outputPrefix, tsamp)
	
	f = open(configPath, 'w')
	f.write(string)
	f.close()

def createMDSMConfigFile(configPath, filePrefix, baseDir, startDM, numDMs, dmStep, tsamp):
	""" Create MDSM Configuration file"""

	string = '<observation>\n\
    <frequencies top="1899.804688" offset="-0.390625" />\n\
    <dm lowDM="%f" numDMs="%d" dmStep="%f" useBruteForce="1" useL1Cache="1" />\n\
    <samples number="%d" />\n\
    <channels number="2048" />\n\
    <timing tsamp="%fe-06" />\n\
    <detection threshold="10" />\n\
    <output filePrefix="%s" baseDirectory="%s" secondsPerFile="180" \n\
            usePCTime="0" singleFileMode="1" useKurtosis="0" />\n\
    <gpus ids="0" />\n\
	</observation>' % (startDM, numDMs, dmStep, nsamp, tsamp, filePrefix, baseDir)

	f = open(configPath, 'w')
	f.write(string)
	f.close()

def processSigprocHeader(filepath):
	""" Process sigproc files """

	p = Popen("%s %s" % (sheader, filepath), 
			   shell = True, stdin = PIPE, stdout = PIPE, stderr = STDOUT, close_fds = True)
	string = p.stdout.read()

	values = {}
	values['telescope'] = re.search("Telescope\W*:\W*(?P<telescope>.*)\W*", string).groupdict()['telescope']
	values['ra'] = re.search("Source RA \(J2000\)\W*:\W*(?P<ra>.*)\W*", string).groupdict()['ra']
	values['dec'] = re.search("Source DEC \(J2000\)\W*:\W*(?P<dec>.*)\W*", string).groupdict()['dec']
	values['time'] = float(re.search("Time stamp of first sample \(MJD\)\W*:\W*(?P<time>.*)\W*", string).groupdict()['time'])
	values['tsamp'] = float(re.search("Sample time \(us\)\W*:\W*(?P<tsamp>.*)\W*", string).groupdict()['tsamp'])
	values['nsamp'] = int(re.search("Number of samples\W*:\W*(?P<nsamp>.*)\W*", string).groupdict()['nsamp'])
	return values	

def createInfFiles(infPath, filename, values, samples):
	""" Create inf files for presto post-processing"""

	string = ' Data file name without suffix          =  %s\n\
 Telescope used                         =  GBT\n\
 Instrument used                        =  GUPPI\n\
 Object being observed                  =  OBJECTa\n\
 J2000 Right Ascension (hh:mm:ss.ssss)  =  %s\n\
 J2000 Declination     (dd:mm:ss.ssss)  =  %s\n\
 Data observed by                       =  Andrew Siemion\n\
 Epoch of observation (MJD)             =  %f\n\
 Barycentered?           (1=yes, 0=no)  =  0\n\
 Number of bins in the time series      =  %d\n\
 Width of each time series bin (sec)    =  %fe-06\n\
 Any breaks in the data? (1=yes, 0=no)  =  0\n\
 Type of observation (EM band)          =  Radio\n\
 Beam diameter (arcsec)                 =  391\n\
 Dispersion measure (cm-3 pc)           =  %f\n\
 Central freq of low channel (Mhz)      =  1500\n\
 Total bandwidth (Mhz)                  =  800\n\
 Number of channels                     =  1\n\
 Channel bandwidth (Mhz)                =  800\n\
 Data analyzed by                       =  Alessio Magro\n\
 Any additional notes:\n\
    Input filterbank samples have 32 bits.' \
	% (filename, values['ra'], values['dec'], values['time'], samples, values['tsamp'], dm )

	f = open(infPath, 'w')
	f.write(string)
	f.close()

if __name__ == "__main__":

	# Read input arguments
	if len(argv) < 3:
		print "Two directories required as arguments"
		exit(0)

	elif not (path.exists(argv[1]) and path.isdir(argv[1])):
		print "Invalid input directory"
		exit(0)

	inputDir  = argv[1]
	outputDir = argv[2]

	# Log file
	logfile = open(logfile, 'w')

	# Check if output directory exists
	if not path.exists(outputDir):
		os.mkdir(outputDir)

	dedisp = lambda f1, f2, dm, tsamp: 4148.741601 * (f1**-2 - f2**-2) * dm / tsamp

	# Loop over all	 files in input directory
	for item in os.listdir(inputDir):
		if path.isfile(inputDir + '/' + item):

			if path.exists(outputDir + '/' + item) or item not in ["Row_00_8bit.fil"]:
				print "Skipping file %s" % (inputDir + '/' + item)
				continue

			logfile.write("%s: Started processing %s\n" % (str(datetime.now()), item))
			
			# Create associated output directory
			currentOutputDir = outputDir + '/' + item
			os.mkdir(currentOutputDir)

			for i in range(performDownsamples):

				# Decimate input file to required
				if downsamp[i] != 1:
					logfile.write("%s: Decimating %s (%d)\n" % (str(datetime.now()), item, downsamp[i]))

					inputFile  = inputDir + '/' + item
					outputFile = inputDir + '/Decimated/' + item[:-4] + "_%d.fil" % downsamp[i]
	
					# Don't recreate the file if it exists	
					if not os.path.exists(outputFile):
						os.system("%s %s -c 1 -t %d > %s" % (decimate, inputFile, downsamp[i], outputFile))
					inputFile = outputFile
				else:
					inputFile = inputDir + '/' + item

				# Extract info from input file
				header = processSigprocHeader(inputFile)

				if header['nsamp'] < 3e6 and downsamp[i] > 8:
					nsamp = 65536 / 2

				# Create configuration files			
				createMDSMConfigFile(currentOutputDir + '/mdsmObs.xml', item[:-4] + '_%d_mdsm' % downsamp[i], 
									 currentOutputDir, startDM[i], numDMs[i], dmStep[i], header['tsamp'])
				createPelicanConfigFile(currentOutputDir + '/pelicanConfig.xml', \
										inputFile, \
										currentOutputDir + '/mdsmObs.xml',
										currentOutputDir + '/' + item,
										header['tsamp'])

				# Call MDSM pipeline, start dedispersing
				logfile.write("%s: Dedispersing %s\n" % (str(datetime.now()), inputFile))
				os.system("%s --config=%s/pelicanConfig.xml" % (mdsmPipeline, currentOutputDir))

				# Write inf files
				for j in range(numDMs[i]):
					dm = startDM[i] + j * dmStep[i]
					if (int(dm)) == dm:
						dmStr = str(int(dm)) + ".00"
					elif len(str(dm).split('.')[1]) <= 1:
						dmStr = str(dm) + "0"
					else:
						dmStr = str(dm)

					createInfFiles(currentOutputDir + '/' + item + '_' + dmStr + '.inf', 
								   item + '_' + dmStr + '.dat', header, 
								   header['nsamp'] - dedisp(1100, 1900, startDM[i] + numDMs[i] * dmStep[i], 
								    						header['tsamp'] / 1e6) )
				logfile.flush()

			logfile.write("%s: Finished processing %s\n\n" % (str(datetime.now()), item))

	logfile.close()

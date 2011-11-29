from numpy import *
from math import *

def valueForKeyword(line, keyword):
	## valueForKeyword
	## Returns the value for the specified keyword in the line
	## Assuming keyword/value are stored in 80 character cards.
	
	index = line.find(keyword)
	if index == -1:
		return -1
	else:
		card = line[index:index+80]		# 80 char card from substring
		eqSignIndex = card.find("=")
		value = card[eqSignIndex + 1:]
		value = value.strip(' \'')		# Strip out surrounding whitespace and ' from value	
		
		return value

### readRaw
### Extracts information from raw file
## Open file for reading
rawFile = open('/queencow/kepler/B2021+51/gpu1/guppi_55689_PSRB2021+51_C_0034.0000.raw','rb')
line = rawFile.readline()

## Header information
nchan = int(valueForKeyword(line, 'OBSNCHAN'))
print 'nchan = ' + str(nchan)

blocSize = int(valueForKeyword(line, 'BLOCSIZE'))
print 'blocSize = ' + str(blocSize)

## Data

endIndex = line.rfind("END")
print 'endIndex = ' + str(endIndex)

rawFile.seek(endIndex + 80)	# Put the file reader at the beginning of the data part of the file

quantizationLookup = [3.3358750, 1.0, -1.0, -3.3358750]

timeSize = blocSize/nchan

requantizedList = empty((nchan*2,timeSize), dtype=complex)	# Complex array storing data
# Format of requantizedList:
# (Max Channels: c, Max Time: n)
# [	[(channel 1, polarization 1, time 1), (channel 1, polarization 1, time 2), ..., (channel 1, polarization 1, time n)],
# 	[(channel 1, polarization 2, time 1), (channel 1, polarization 2, time 2), ..., (channel 1, polarization 2, time n)],
# 	[(channel 2, polarization 1, time 1), (channel 2, polarization 1, time 2), ..., (channel 2, polarization 1, time n)],
# 	[(channel 2, polarization 2, time 1), (channel 2, polarization 2, time 2), ..., (channel 2, polarization 2, time n)],
# 	...,
# 	[(channel c, polarization 1, time 1), (channel c, polarization 1, time 2), ..., (channel c, polarization 1, time n)],
# 	[(channel c, polarization 2, time 1), (channel c, polarization 2, time 2), ..., (channel c, polarization 2, time n)], 
# ]

print '\n\n'

for c in range(0, nchan):
	print "Current channel: " + str(c + 1)
	for t in range(0, timeSize):
		aChar = rawFile.read(1)	# Read next character
		byteValue = ord(aChar)	# Convert character to decimal value, representing one byte of data
		
		for a in range(0, 4):
			value = quantizationLookup[(byteValue >> (a * 2)) & 3]
			
			# print 'Value: ' + str(value)	# Debugging
			
			if a%2 == 0:
				requantizedList[c*2 + (a/2)][t] = complex(value, requantizedList[c*2 + (a/2)][t].imag)
			else:
				requantizedList[c*2 + (a/2)][t] = complex(requantizedList[c*2 + (a/2)][t].real, value)


rawFile.close()


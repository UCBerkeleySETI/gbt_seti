from sys import argv
from os import path
import numpy as np

class RawFile:

    def __init__(self, filename):
        """ Class constructor """

        # Initialise values and open file
        self._rawFile = open(filename, 'rb')
        self._offset  = 0
        self._nchans = 0
        self._blocSize = 0

        # Extract file size
        self._rawFile.seek(0, 2)
        self._filesize = self._rawFile.tell()
        self._rawFile.seek(0, 0)


    def __del__(self):
        """ Destructor """
        self._rawFile.close()


    def valueForKeyword(self, line, keyword):
	    """ Returns the value for the specified keyword in the line
	        Assuming keyword/value are stored in 80 character cards. """
	
	    index = line.find(keyword)
	    if index == -1:
		    return -1
	    else:
		    card = line[index:index + 80]		# 80 char card from substring
		    eqSignIndex = card.find("=")
		    value = card[eqSignIndex + 1:]
		    value = value.strip(' \'')		# Strip out surrounding whitespace and ' from value	
		
	    return value


    def readHeader(self):
        """ Read data header for current file section """
        
        # Read section header
        self._rawFile.seek(self._offset)
        line = self._rawFile.readline(1024*7)

        # Extract required header information
        self._nchans = int(self.valueForKeyword(line, 'OBSNCHAN'))
        self._blocSize = int(self.valueForKeyword(line, 'BLOCSIZE'))

        # Update file offset
        endIndex = line.rfind("END")
        self._offset += endIndex + 80


    def processFileSegment(self):
        """ Process file """

        quantLookup = [3.3358750, 1.0, -1.0, -3.3358750]
        
        # Process all data sections in file
        if self._offset < self._filesize:

            # Extract section header and seek to correct position
            self.readHeader()
            self._rawFile.seek(self._offset)

            timeSize = self._blocSize / self._nchans

            # Data object needs to be resized for new data to fit
            data = np.zeros([self._nchans * 2, timeSize], dtype=complex)

            for c in range(0, 2):
            
                self._rawFile.seek(self._offset + c * timeSize)
                values = self._rawFile.read(timeSize)

                for t in range(0, timeSize):
                    
                    # Convert character to decimal value, representing one byte of data
                    byteValue = ord(values[t])

                    data[c * 2, t]     = complex(quantLookup[(byteValue >> 0) & 3], 
                                                 quantLookup[(byteValue >> 2) & 3])
                    data[c * 2 + 1, t] = complex(quantLookup[(byteValue >> 4) & 3], 
                                                 quantLookup[(byteValue >> 6) & 3])  
  
            # Update offset
            self._offset += self._blocSize

            # We are done, return data
            return data

        # No more data in file, return None    
        else:
            return None


# Script entry point
if __name__ == "__main__":

    if len(argv) < 2:
        print "Filepath required as an argument"
        exit(0)

    elif not (path.exists(argv[1]) and path.isfile(argv[1])):
        print "Provided filepath is invalid"
        exit(0)

    # Create class instance for file
    f = RawFile(argv[1])

    # Loop through file segments in file:
    iterations = 0;

    series = np.array([])
    valsx = valsy = []
    while True:

        requantizedData = f.processFileSegment()
        iterations += 1

        print "Read file segment %d" % iterations

        if requantizedData == None:
            print "Reached end of file"
            exit();

        # Process data here
		# ........... PROCESSING ...................


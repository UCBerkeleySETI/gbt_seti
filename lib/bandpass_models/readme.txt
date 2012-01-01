The files in this directory are empirically derived models of the bandpass
shape of polyphase filterbank channels produced by the Green Bank Ultimate
Pulsar Processor (GUPPI).  Each file is labeled based on the number of points
in the model, fifteen.bin: 2^15 points, twenty.bin: 2^20 points, and so on.
The file format is a simple sequence of IEEE 32bit floating point values.

Some example commands for reading and plotting these data using the MATLAB
clone Octave are:
 
octave:1> file = fopen("fifteen.bin")
file =  3
octave:2> model=fread(file, 32768, "single");
octave:3> plot(model)



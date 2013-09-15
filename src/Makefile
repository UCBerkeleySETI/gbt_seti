OPT64 = /home/pulsar64
FC = gfortran
FFLAGS = -g -fPIC
OS = Linux
BINDIR = $(SETI_GBT)/bin
# Linux is the first choice
ifeq ($(OS),Linux)
        LIBSUFFIX = .so
        LIBCMD = -shared
# else assume Darwin (i.e. OSX)
else
        LIBSUFFIX = .dylib
        LIBCMD = -dynamiclib
endif

CFLAGS = -O3 -Wall -DFOLD_USE_INTRINSICS -I$(OPT64)/include -I/usr/include/cfitsio -I/usr/include/mysql 
PROGS = test_psrfits_read \
	raw_quant raw_read rawdopplersearch \
	treedop obs_db_update raw_head two2eight rawfilterbank rawfilterbank_singleband_xy raw_ascii2
OBJS  = guppi_status.o guppi_databuf.o guppi_udp.o guppi_error.o \
	guppi_params.o guppi_time.o median.o setimysql.o\
	write_psrfits.o read_psrfits.o misc_utils.o \
	fold.o polyco.o hget.o hput.o sla.o downsample.o barycenter.o chkio.o
LIBS = -L$(OPT64)/lib -lfftw3f_threads -lcfitsio -L$(SETI_GBT)/lib -lsla -lm -lgsl -lgslcblas -lpthread -I/usr/include/cfitsio -lfftw3f -L/usr/lib64/mysql -lmysqlclient 
all: slalib $(PROGS)
clean:
	rm -f $(PROGS) *~ *.o 
INSTALL_DIR = ../bin
install: $(PROGS)
	mkdir -p $(INSTALL_DIR) && \
	cp -f $(PROGS) $(INSTALL_DIR)
slalib: libsla$(LIBSUFFIX)
	cd slalib ; $(FC) -o sla_test sla_test.f -fno-second-underscore -L$(SETI_GBT)/lib -lsla
	slalib/sla_test
libsla$(LIBSUFFIX):
	cd slalib ; $(FC) $(FFLAGS) -fno-second-underscore -c -I. *.f *.F
	rm slalib/sla_test.o
	cd slalib ; $(FC) $(LIBCMD) -o $(SETI_GBT)/lib/libsla$(LIBSUFFIX) -fno-second-underscore *.o

.SECONDEXPANSION:
$(PROGS): $$@.c $(OBJS)
	$(CC) $(CFLAGS) $< -o $(BINDIR)/$@ $(OBJS) $(LIBS) $(THREAD_LIBS)

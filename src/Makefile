OPT64 = /opt/vegas
FC = gfortran
FFLAGS = -g -fPIC
OS = Linux
# Default to installing in /usr/local/listen
SETI_GBT ?= /usr/local/listen
prefix ?= $(SETI_GBT)
BINDIR = $(prefix)/bin
LIBDIR = $(prefix)/lib
CUDA = /usr/local/cuda
CUDAC = $(CUDA)/bin/nvcc
# Linux is the first choice
ifeq ($(OS),Linux)
        LIBSUFFIX = .so
        LIBCMD = -shared
# else assume Darwin (i.e. OSX)
else
        LIBSUFFIX = .dylib
        LIBCMD = -dynamiclib
endif

CUDAFLAGS = --std=c++11 
CFLAGS = -Wno-unused-result -O3 -Wall -Werror=return-local-addr -DFOLD_USE_INTRINSICS -I$(OPT64)/include -I/usr/include/cfitsio -I/usr/include/mysql -I$(CUDA)/include -I/usr/local/include -I/usr/local/include/libbson-1.0 -I/usr/local/include/libmongoc-1.0
PROGS = test_psrfits_read \
	raw_quant raw_read massage_vpm \
	treedop raw_head two2eight filterbanksearch s3test filterbanksearchdoppler baryacc
GPU_PROGS = gpuspec gpuspec2 raw2wu test_guppi2spectra
OBJS  = guppi_status.o guppi_databuf.o guppi_udp.o guppi_error.o \
	guppi_params.o guppi_time.o median.o setimysql.o imswap.o\
	write_psrfits.o read_psrfits.o misc_utils.o read_filterbank_header.o filterbankutil.o \
	fold.o polyco.o hget.o hput.o sla.o downsample.o barycenter.o chkio.o s3util.o
LIBS = -L$(OPT64)/lib -I$(OPT64)/include -L$(CUDA)/lib64 -lfftw3f_threads -lcfitsio -lm -lgsl -lgslcblas -lpthread -I/usr/include/cfitsio -lfftw3f -L/usr/lib64/mysql -lmysqlclient -lcufft -lcudart -Lslalib -ls3 -lmongoc-1.0 -lbson-1.0 -lstdc++ 

ifeq ($(DEBUG),1)
  CFLAGS += -ggdb -O1
endif

all: $(PROGS) $(GPU_PROGS)

guppi2spectra_gpu.o: guppi2spectra_gpu.cu
	$(CUDAC) -c guppi2spectra_gpu.cu -o guppi2spectra_gpu.o $(CUDAFLAGS)

test_guppi2spectra: test_guppi2spectra.c guppi2spectra_gpu.o
	$(CC) $(CFLAGS) -fno-strict-aliasing test_guppi2spectra.c -o $@ guppi2spectra_gpu.o $(LIBS) $(THREAD_LIBS) -lstdc++

gpuspec: guppi2spectra.c guppi2spectra_gpu.o $(OBJS)
	$(CC) $(CFLAGS) -fno-strict-aliasing guppi2spectra.c -o gpuspec $(OBJS) guppi2spectra_gpu.o $(LIBS) $(THREAD_LIBS) -lsla -lstdc++

gpuspec2: guppi2spectra2.c guppi2spectra_gpu.o $(OBJS)
	$(CC) $(CFLAGS) -fno-strict-aliasing guppi2spectra2.c -o gpuspec2 $(OBJS) guppi2spectra_gpu.o $(LIBS) $(THREAD_LIBS) -lsla -lstdc++
	
raw2wu: raw2wu.c guppi2spectra_gpu.o $(OBJS)
	$(CC) $(CFLAGS) -fno-strict-aliasing raw2wu.c -o raw2wu $(OBJS) guppi2spectra_gpu.o $(LIBS) $(THREAD_LIBS) -lsla -lstdc++


gpu: gpuspec raw2wu

check: 

install: install-listen_sh install-progs install-libs

install-listen_sh:
	mkdir -p $(prefix)
	echo 'test -e $(CUDA)/cuda.sh && source $(CUDA)/cuda.sh'       > $(prefix)/listen.sh
	echo 'export LD_LIBRARY_PATH="$${LD_LIBRARY_PATH}:$(LIBDIR)"' >> $(prefix)/listen.sh
	echo 'export PATH="$${PATH}:$(BINDIR)"'                       >> $(prefix)/listen.sh
	echo 'source $$(dirname $${BASH_SOURCE[0]})/setkeys.sh'       >> $(prefix)/listen.sh
	
install-progs: $(PROGS) $(GPU_PROGS)
	mkdir -p $(BINDIR)
	cp -f $^ $(BINDIR)

install-libs: 
	mkdir -p $(LIBDIR)
	cp -f $^ $(LIBDIR)

tags:
	ctags -R .

clean:
	rm -f $(PROGS) $(GPU_PROGS) *~ *.o 

.PHONY: all gpu check install install-listen_sh install-progs install-libs tags clean

.SECONDEXPANSION:
$(PROGS): $$@.c $(OBJS) 
	$(CC) $(CFLAGS) $< -o $@ $(OBJS) $(LIBS) -lsla $(THREAD_LIBS)

# vi: set ts=8 :

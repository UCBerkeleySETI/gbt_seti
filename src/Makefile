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

SLAOBJS = slalib/addet.o slalib/afin.o slalib/airmas.o slalib/altaz.o                \
	slalib/amp.o slalib/ampqk.o slalib/aop.o slalib/aoppa.o slalib/aoppat.o      \
	slalib/aopqk.o slalib/atmdsp.o slalib/atms.o slalib/atmt.o slalib/av2m.o     \
	slalib/bear.o slalib/caf2r.o slalib/caldj.o slalib/calyd.o slalib/cc2s.o     \
	slalib/cc62s.o slalib/cd2tf.o slalib/cldj.o slalib/clyd.o slalib/combn.o     \
	slalib/cr2af.o slalib/cr2tf.o slalib/cs2c.o slalib/cs2c6.o slalib/ctf2d.o    \
	slalib/ctf2r.o slalib/daf2r.o slalib/dafin.o slalib/dat.o slalib/dav2m.o     \
	slalib/dbear.o slalib/dbjin.o slalib/dc62s.o slalib/dcc2s.o slalib/dcmpf.o   \
	slalib/dcs2c.o slalib/dd2tf.o slalib/de2h.o slalib/deuler.o slalib/dfltin.o  \
	slalib/dh2e.o slalib/dimxv.o slalib/djcal.o slalib/djcl.o slalib/dm2av.o     \
	slalib/dmat.o slalib/dmoon.o slalib/dmxm.o slalib/dmxv.o slalib/dpav.o       \
	slalib/dr2af.o slalib/dr2tf.o slalib/drange.o slalib/dranrm.o                \
	slalib/ds2c6.o slalib/ds2tp.o slalib/dsep.o slalib/dsepv.o slalib/dt.o       \
	slalib/dtf2d.o slalib/dtf2r.o slalib/dtp2s.o slalib/dtp2v.o                  \
	slalib/dtps2c.o slalib/dtpv2c.o slalib/dtt.o slalib/dv2tp.o                  \
	slalib/dvdv.o slalib/dvn.o slalib/dvxv.o slalib/e2h.o slalib/earth.o         \
	slalib/ecleq.o slalib/ecmat.o slalib/ecor.o slalib/eg50.o                    \
	slalib/el2ue.o slalib/epb.o slalib/epb2d.o slalib/epco.o slalib/epj.o        \
	slalib/epj2d.o slalib/epv.o slalib/eqecl.o slalib/eqeqx.o                    \
	slalib/eqgal.o slalib/etrms.o slalib/euler.o slalib/evp.o                    \
	slalib/fitxy.o slalib/fk425.o slalib/fk45z.o slalib/fk524.o                  \
	slalib/fk52h.o slalib/fk54z.o slalib/fk5hz.o slalib/flotin.o                 \
	slalib/galeq.o slalib/galsup.o slalib/ge50.o slalib/geoc.o                   \
	slalib/gmst.o slalib/gmsta.o slalib/gresid.o slalib/h2e.o                    \
	slalib/h2fk5.o slalib/hfk5z.o slalib/idchf.o slalib/idchi.o                  \
	slalib/imxv.o slalib/intin.o slalib/invf.o slalib/kbj.o slalib/m2av.o        \
	slalib/map.o slalib/mappa.o slalib/mapqk.o slalib/mapqkz.o                   \
	slalib/moon.o slalib/mxm.o slalib/mxv.o slalib/nut.o slalib/nutc.o           \
	slalib/nutc80.o slalib/oap.o slalib/oapqk.o slalib/obs.o slalib/pa.o         \
	slalib/pav.o slalib/pcd.o slalib/pda2h.o slalib/pdq2h.o slalib/permut.o      \
	slalib/pertel.o slalib/pertue.o slalib/planel.o slalib/planet.o              \
	slalib/plante.o slalib/plantu.o slalib/pm.o slalib/polmo.o                   \
	slalib/prebn.o slalib/prec.o slalib/preces.o slalib/precl.o                  \
	slalib/prenut.o slalib/pv2el.o slalib/pv2ue.o slalib/pvobs.o                 \
	slalib/pxy.o slalib/random.o slalib/range.o slalib/ranorm.o                  \
	slalib/rcc.o slalib/rdplan.o slalib/refco.o slalib/refcoq.o                  \
	slalib/refro.o slalib/refv.o slalib/refz.o slalib/rverot.o                   \
	slalib/rvgalc.o slalib/rvlg.o slalib/rvlsrd.o slalib/rvlsrk.o                \
	slalib/s2tp.o slalib/sep.o slalib/sepv.o slalib/smat.o slalib/subet.o        \
	slalib/supgal.o slalib/svd.o slalib/svdcov.o slalib/svdsol.o                 \
	slalib/tp2s.o slalib/tp2v.o slalib/tps2c.o slalib/tpv2c.o                    \
	slalib/ue2el.o slalib/ue2pv.o slalib/unpcd.o slalib/v2tp.o slalib/vdv.o      \
	slalib/veri.o slalib/vers.o slalib/vn.o slalib/vxv.o slalib/wait.o           \
	slalib/xy2xy.o slalib/zd.o

ifeq ($(DEBUG),1)
  CFLAGS += -ggdb -O1
endif

SLALIB = slalib/libsla$(LIBSUFFIX)

all: $(SLALIB) $(PROGS) $(GPU_PROGS)

slalib/%.o : slalib/%.f
	$(FC) $(FFLAGS) -fno-second-underscore -c -I slalib -o $@ $^

slalib/%.o : slalib/%.F
	$(FC) $(FFLAGS) -fno-second-underscore -c -I slalib -o $@ $^

$(SLALIB): $(SLAOBJS)
	cd slalib ; $(FC) $(FFLAGS) -fno-second-underscore -c -I. *.f *.F
	rm slalib/sla_test.o
	cd slalib ; $(FC) $(LIBCMD) -o $(notdir $(SLALIB)) -fno-second-underscore *.o

guppi2spectra_gpu.o: guppi2spectra_gpu.cu
	$(CUDAC) -c guppi2spectra_gpu.cu -o guppi2spectra_gpu.o $(CUDAFLAGS)

test_guppi2spectra: test_guppi2spectra.c guppi2spectra_gpu.o
	$(CC) $(CFLAGS) -fno-strict-aliasing test_guppi2spectra.c -o $@ guppi2spectra_gpu.o $(LIBS) $(THREAD_LIBS) -lstdc++

gpuspec: guppi2spectra.c guppi2spectra_gpu.o $(SLALIB) $(OBJS)
	$(CC) $(CFLAGS) -fno-strict-aliasing guppi2spectra.c -o gpuspec $(OBJS) guppi2spectra_gpu.o $(LIBS) $(THREAD_LIBS) -lsla -lstdc++

gpuspec2: guppi2spectra2.c guppi2spectra_gpu.o $(SLALIB) $(OBJS)
	$(CC) $(CFLAGS) -fno-strict-aliasing guppi2spectra2.c -o gpuspec2 $(OBJS) guppi2spectra_gpu.o $(LIBS) $(THREAD_LIBS) -lsla -lstdc++
	
raw2wu: raw2wu.c guppi2spectra_gpu.o $(SLALIB) $(OBJS)
	$(CC) $(CFLAGS) -fno-strict-aliasing raw2wu.c -o raw2wu $(OBJS) guppi2spectra_gpu.o $(LIBS) $(THREAD_LIBS) -lsla -lstdc++

slalib: $(SLALIB)

gpu: gpuspec raw2wu

check: slalib-check

slalib-check: slalib
	cd slalib ; $(FC) -o sla_test sla_test.f -fno-second-underscore -L. -lsla
	cd slalib; ./sla_test

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

install-libs: $(SLALIB)
	mkdir -p $(LIBDIR)
	cp -f $^ $(LIBDIR)

tags:
	ctags -R .

clean:
	rm -f $(PROGS) $(GPU_PROGS) *~ *.o 
	rm -f $(SLALIB) $(SLAOBJS) slalib/sla_test

.PHONY: all slalib gpu check slalib-check install install-listen_sh install-progs install-libs tags clean

.SECONDEXPANSION:
$(PROGS): $$@.c $(OBJS) $(SLALIB)
	$(CC) $(CFLAGS) $< -o $@ $(OBJS) $(LIBS) -lsla $(THREAD_LIBS)

# vi: set ts=8 :

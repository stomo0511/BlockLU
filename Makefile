CXX = /usr/local/bin/g++-9
CXXFLAGS = -m64 -fopenmp -O2
LDFLAGS = -lpthread -lm -ldl

MKL_ROOT =  /opt/intel/compilers_and_libraries/mac/mkl
MKL_INC_DIR = $(MKL_ROOT)/include
MKL_LIB_DIR = $(MKL_ROOT)/lib
MKL_LIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

OBJS =		BlockLU.o
#OBJS =		BlockLU.o trace.o

TARGET =	BlockLU

all:	$(TARGET)

$(TARGET):	$(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS) -L$(MKL_LIB_DIR) $(MKL_LIBS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -I$(MKL_INC_DIR) -o $@ $<

%.o: %.c
	$(CXX) -c $(CXXFLAGS) -I$(MKL_INC_DIR) -o $@ $<

trace.o: trace.c
	$(CXX) -O2 -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

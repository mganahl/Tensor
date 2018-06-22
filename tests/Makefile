#LDFLAGS= -static -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group -liomp5 -lboost_serialization
LDFLAGS=  -lhptt -ltblis -lopenblas -ltcl /usr/lib/atlas-base/atlas/liblapack.so
CPP=g++
CPPFLAGS= -O3 -m64 -fopenmp  -fPIC -std=c++17
OBJ = ../lib/linalg/tblisroutines.o ../lib/linalg/blasroutines.o ../lib/linalg/lapackroutines.o

#CPPFLAGS= -O3 -m64 -fopenmp -DNDEBUG -fPIC -std=c++14   
#INCLUDES=-I/home/maga/software/hptt-master/include -I/home/maga/software/tblis-master/src/external/tci/
#INCLUDES=-I/home/maga/workspace/MPSTools/C++/tensorlib/lib/linalg/tblisroutines.cpp
#LINKS=-L/home/maga/software/hptt-master/lib (I have put a symlink to libhptt.so and tlibtblis into /usr/lib)
INCLUDES=


all: test.exe testSlice.exe testconj.exe testlapack.exe testtblis.exe testtcl.exe testgemm.exe testgemm_mv.exe testTensorTranspose.exe testcontractionall.exe testlapack.exe testArithmetic.exe testConstructors.exe tblisvsgemm.exe 
test.exe: test.cpp
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $^ -o $@ $(OBJ)  $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '
tblisvsgemm.exe: tblisvsgemm.cpp
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $^ -o $@ $(OBJ)  $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '
testTensordot.exe: testTensordot.cpp
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $^ -o $@ $(OBJ)  $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '

testconj.exe: testconj.cpp
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $^ -o $@ $(OBJ)  $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '
testConstructors.exe: testConstructors.cpp
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $^ -o $@ $(OBJ)  $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '
testCast.exe: testCast.cpp 
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $< -o $@ $(OBJ) $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '

testlapack.exe: testlapack.cpp 
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $< -o $@ $(OBJ) $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '
testcontractionall.exe: testcontractionall.cpp 
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $< -o $@  $(OBJ) $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '

testtblis.exe: testtblis.cpp 
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $< -o $@  $(OBJ) $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '
testtcl.exe: testtcl.cpp 
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $< -o $@  $(OBJ) $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '
testgemm.exe: testgemm.cpp 
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $< -o $@  $(OBJ) $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '
testgemm_mv.exe: testgemm_mv.cpp 
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $< -o $@  $(OBJ) $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '

testTensorTranspose.exe: testTensorTranspose.cpp
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $< -o $@  $(OBJ) $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '
testTensor.exe: testTensor.cpp
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $< -o $@  $(OBJ)  $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '

testTranspose.exe: testTranspose.cpp
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $< -o $@  $(OBJ)  $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '
testSlice.exe: testSlice.cpp
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $< -o $@  $(OBJ)  $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '
testArithmetic.exe: testArithmetic.cpp
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $< -o $@  $(OBJ)  $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '
testblas.exe: testblas.cpp
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS) $(INCLUDES)  $< -o $@  $(OBJ)  $(LDFLAGS) 
	@echo 'Finished building $@'
	@echo ' '

clean:
	rm -f  *.exe

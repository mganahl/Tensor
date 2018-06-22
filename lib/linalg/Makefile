LDFLAGS=   -lpthread -lblas -ltblis -ltcl -llapack
CPP=g++
CPPFLAGS= -O3 -m64 -fopenmp -DNDEBUG -fPIC -std=c++17
all:blasroutines.o tblisroutines.o lapackroutines.o testblas.exe testlapack.exe
%.o: %.cpp 
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS)   -c $< -o $@ $(LDFLAGS)
	@echo 'Finished building $@'
	@echo ' '

blasroutines.o: blasroutines.cpp 
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS)  -c $< -o $@ 
	@echo 'Finished building $@'
	@echo ' '
lapackroutines.o: lapackroutines.cpp 
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS)  -c $< -o $@ 
	@echo 'Finished building $@'
	@echo ' '

tblisroutines.o: tblisroutines.cpp 
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS)  -c $< -o $@ 
	@echo 'Finished building $@'
	@echo ' '

testblas.exe: testblas.cpp blasroutines.o
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS)  $^ -o $@ $(LDFLAGS)
	@echo 'Finished building $@'
	@echo ' '
testlapack.exe: testlapack.cpp  lapackroutines.o blasroutines.o
	@echo 'Building $<'
	$(CPP) $(CPPFLAGS)  $^ -o $@ $(LDFLAGS)
	@echo 'Finished building $@'
	@echo ' '

clean:
	rm -f  *.o
	rm -f *.exe

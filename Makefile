# Usage of the makefile
# make <action> 
# make clean && make <action> to recompile up to date files

CXXFLAGS=-O3
EXTRAFLAGS=-arch=sm_20

# This variable is used as the action name. It's the exact name of the source file solving the first question.
#FILE=src/first

action: 
	nvcc -lcublas -lcurand -o bin/execute $(CXXFLAGS) $(EXTRAFLAGS) src/$(FILE).cu
clean:
	#deletes executables
	$(RM) bin/execute

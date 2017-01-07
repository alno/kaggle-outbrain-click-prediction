CXX = g++
CXXFLAGS = -Wall -O3 -std=c++14 -march=native -mrdrnd

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

all: bin/prepare-leak bin/prepare-similarity bin/prepare-counts bin/prepare-viewed-docs bin/prepare-viewed-ads bin/export-vw-data bin/export-ffm-data bin/ffm bin/export-ffm-data-2

bin/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DFLAG) -MMD -c -o $@ $<

bin/%: bin/%.o bin/ffm-io.o
	$(CXX) $(CXXFLAGS) -o $@ $< bin/ffm-io.o -lboost_iostreams -lboost_program_options

-include bin/*.d

.PHONY: clean
clean:
	rm bin/*

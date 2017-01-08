CXX = g++
CXXFLAGS = -Wall -O3 -std=c++14 -march=native

# comment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp

TARGETS = bin/prepare-leak bin/prepare-similarity bin/prepare-counts
TARGETS += bin/prepare-viewed-ads
TARGETS += bin/prepare-viewed-docs bin/prepare-group-viewed-docs
TARGETS += bin/export-vw-data bin/export-ffm-data bin/ffm bin/export-bin-data-p1

all: $(TARGETS)

bin/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DFLAG) -MMD -c -o $@ $<

bin/%: bin/%.o bin/ffm-io.o bin/ffm-model.o
	$(CXX) $(CXXFLAGS) -o $@ $< bin/ffm-io.o bin/ffm-model.o -lboost_iostreams -lboost_program_options

-include bin/*.d

.PHONY: clean
clean:
	rm bin/*

CXX = g++
CXXFLAGS = -Wall -O3 -std=c++14 -march=native -fopenmp

TARGETS = bin/prepare-leak bin/prepare-counts bin/prepare-rivals
TARGETS += bin/prepare-viewed-ads bin/prepare-viewed-docs
TARGETS += bin/export-bin-data-f4
TARGETS += bin/ffm


all: $(TARGETS)

bin/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DFLAG) -MMD -c -o $@ $<

bin/%: bin/%.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lboost_iostreams -lboost_program_options


bin/ffm: bin/ffm-io.o bin/ffm-model.o
bin/export-bin-data-f4: bin/ffm-io.o

-include bin/*.d

.PHONY: clean
clean:
	rm bin/*

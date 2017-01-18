CXX = g++
CXXFLAGS = -Wall -O3 -std=c++14 -march=native -fopenmp

TARGETS = bin/prepare-leak bin/prepare-similarity bin/prepare-counts bin/prepare-rivals
TARGETS += bin/prepare-viewed-ads bin/prepare-viewed-docs bin/prepare-group-viewed-docs
TARGETS += bin/export-vw-data bin/export-ffm-data bin/export-bin-data-p1 bin/export-bin-data-f1 bin/export-bin-data-f2 bin/export-bin-data-f3 bin/export-bin-data-f4 bin/export-bin-data-f5
TARGETS += bin/ffm


all: $(TARGETS)

bin/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DFLAG) -MMD -c -o $@ $<

bin/%: bin/%.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -lboost_iostreams -lboost_program_options


bin/ffm: bin/ffm-io.o bin/ffm-model.o bin/ffm-nn-model.o bin/ftrl-model.o bin/nn-model.o
bin/export-bin-data-p1: bin/ffm-io.o
bin/export-bin-data-f1: bin/ffm-io.o
bin/export-bin-data-f2: bin/ffm-io.o
bin/export-bin-data-f3: bin/ffm-io.o
bin/export-bin-data-f4: bin/ffm-io.o
bin/export-bin-data-f5: bin/ffm-io.o

-include bin/*.d

.PHONY: clean
clean:
	rm bin/*

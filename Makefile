default: bin/export-vw-data bin/export-ffm-data bin/prepare-leak bin/prepare-similarity

bin/%: %.cpp
	g++ -O3 -std=c++14 $< -lboost_iostreams -MMD -o $@

-include bin/*.d

.PHONY: clean
clean:
	rm bin/*

default: bin/export-vw-data bin/export-ffm-data

bin/%: %.cpp
	g++ -std=c++14 $< -lboost_iostreams -MMD -o $@

-include *.d

.PHONY: clean
clean:
	rm bin/*

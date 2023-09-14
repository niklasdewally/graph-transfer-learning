TARGETS=$(patsubst %.dot,%.png,$(wildcard *.dot))

.PHONY: all
all: $(TARGETS)

%.png: %.dot
	dot -Tpng $< > $@

.PHONY: clean
clean: 
	rm -f $(TARGETS)

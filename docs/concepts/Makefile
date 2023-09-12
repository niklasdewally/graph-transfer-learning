
PNGS=$(patsubst %.dot,%.png,$(wildcard *.dot))

all: $(PNGS)

%.png: %.dot
	dot -Tpng $< > $@
	


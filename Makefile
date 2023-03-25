SRC = $(wildcard notebooks/*.ipynb)
DEST = $(patsubst notebooks/%.ipynb,documents/%.html,$(SRC))
.PHONY: all
all: $(DEST)

documents/%.html: notebooks/%.ipynb
	jupyter nbconvert --to html_embed --output "$*" --output-dir documents "$<"

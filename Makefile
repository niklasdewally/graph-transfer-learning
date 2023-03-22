.PHONY: all
all: documents/EGI_Framework_Original_Code_Overview.html

documents/%.html: notebooks/%.ipynb
	jupyter nbconvert --to html --output "$*" --output-dir documents "$<"

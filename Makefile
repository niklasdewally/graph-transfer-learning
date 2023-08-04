#MAKEFLAGS += -j2 # do things in parallel

#################################################
#                    GLOBALS                    #
#################################################

PYTHON := poetry run python3

########################################################
#                    PHONY COMMANDS                    #
########################################################

.PHONY: all
## Run all experiment perparation tasks
all: dl prepare-triangle-detection generate-data

.PHONY: aug23-generate-data
## Generate synthetic graphs and sample negative triangles dataset for August 23 experiments. 
## Use the pregenerated dataset (dl-aug23) if possible for reproducibility.
generate-data: data/generated/2023-08-clustered

.PHONY: dl-coauthor
## Download raw coauthor datasets
dl-coauthor: data/raw/coauthor-cs.npz data/raw/coauthor-phy.npz

.PHONY: dl-aug23
## Download input synthetic graphs for August 2023 experiments.
dl-aug23: data/2023-08-clustered

.PHONY: clean
## Clean all data directories
clean:
	rm -rf data/processed/*
	rm -rf data/raw/*
	rm -rf data/2023-08-clustered
	rm -rf data/generated/*


###############################################
#                    RULES                    #
###############################################

# graph generation is for creating .tar.gz stored on the website only.
# most of the time, use the tar on the website for reproducability.
# use this to regenerate the data inside this tar
data/generated/2023-08-clustered:
	mkdir -p $@
	# disable limits on cpu-time; memory
	ulimit -t hard &&\
	ulimit -v hard &&\
	$(PYTHON) scripts/generate-data/generate_2023_08_clustered.py $@ &&\
	$(PYTHON) scripts/pre-processing/negative-triangles.py $@ $@


AUG_TAR_URL = https://niklas.dewally.com/files/staris-gtl/2023-08-03-clustered-graphs.tar.gz
data/2023-08-clustered:
	mkdir -p $@
	curl -L "$(AUG_TAR_URL)" | tar -xzf - -C $@

# Download raw coauthor datasets from
# https://github.com/shchur/gnn-benchmark

data/raw/coauthor-cs.npz: 
	mkdir -p data/raw
	curl -o $@ -L https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_cs.npz

data/raw/coauthor-phy.npz: 
	mkdir -p data/raw
	curl -o $@ -L https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_phy.npz

############################################################
#                    SELF DOCUMENTATION                    #
############################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

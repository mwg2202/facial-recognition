SHELL := /bin/bash

define conda_run
	-conda env create -f src/$(1).yml
	source activate facial-recognition-$(1) && python src/$(1).py;
endef

%: src/%.py
	$(call conda_run,$@)

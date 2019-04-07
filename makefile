include makefiles/d55.mk
include makefiles/d91.mk
include makefiles/pm.mk

.DEFAULT_GOAL := all
.PHONY: all D91 D55 PM clean

all: D91 D55 PM

clean:
	@echo "Cleaning up checkpoints"
	rm -rf checkpoints/

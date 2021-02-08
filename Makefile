PLATFORM = third_party/brezel

# Makes sure the Research Platform gets pulled as a submodule into $(PLATFORM)
# Installs hooks to automate submodule updates on pull
# WARNING: uncommitted modifications in $(PLATFORM) will be lost!
.PHONY: init
init:
	git submodule sync --recursive
	git submodule update --init --recursive --force
	git config core.hooksPath $(PLATFORM)/scripts/githooks/project_repo

# Upgrade brezel
# Running `make upgrade` will fast-forward folder third_party/brezel into the
# current master commit of the brezel repository.
# You usually need to run this rule if you want to use new features from the Research Platform.
# WARNING: running this command will create a commit in the present project.
# WARNING: uncommitted modifications in $(PLATFORM) will be lost!
.PHONY: upgrade
upgrade:
	git submodule foreach git fetch origin
	git submodule foreach git checkout origin/master --force
	git commit $(PLATFORM) --message 'RESEARCH PLATFORM: upgrade submodule'

# Call rule from brezel's Makefile
# If the rule is not defined in the present Makefile, we forward it the the Makefile
# located in $(PLATFORM).
# Typical usage: build, run, run-list
.PHONY: phony_explicit
phony_explicit:
%: phony_explicit
	@$(MAKE) --no-print-directory --makefile=$(PLATFORM)/Makefile $@ ROOT=$(PWD)/$(PLATFORM)


##
# Makefile rules specific to this project
##

# Path to the bazel sandbox
BAZELBIN = $(shell bazel info bazel-bin 2>/dev/null)
bazel-run:
	@ln -s $(BAZELBIN)/run $@

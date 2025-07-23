ROSWELL_PREFIX=$(HOME)/.local
ROSWELL=$(ROSWELL_PREFIX)/bin/ros
INSTALLED_SYSTEMS=$(HOME)/.roswell/lisp/quicklisp/dists/quicklisp/software
INSTALLED_LOCAL=$(HOME)/.roswell/local-projects
TESTS_FILE="bianet-neuron-tests.lisp"
TEST_WORK_FILE="bianet-neuron-test-work.lisp"
REPORTER=list

APT_PACKAGES=automake \
             build-essential \
             curl \
             git \
             gnupg \
             libcurl4-openssl-dev \
             zlib1g-dev

CL_PACKAGES=cl-ppcre \
            yason \
            ironclad \
            trivial-utf-8 \
            cl-csv \
						zpng \
						png-read \
						transducers \
            prove \
            macnod/dc-dlist \
            macnod/dc-eclectic \
            macnod/bianet-mesh

.PHONY: all setup install-apt-packages install-roswell install-dependencies test clean

all: setup test

setup: install-apt-packages install-roswell install-dependencies

install-apt-packages: $(APT_PACKAGES)

$(APT_PACKAGES):
	@dpkg-query -l --no-pager $@ >/dev/null || sudo apt install $@ -y

install-roswell:
	if ! [ -f $(ROSWELL) ]; then \
		git clone https://github.com/roswell/roswell.git roswell; \
		cd roswell; \
		sh bootstrap; \
		./configure --prefix=$(ROSWELL_PREFIX); \
		make; \
		make install; \
		cd ..; \
		rm -rf roswell; \
		$(ROSWELL) setup; \
	fi

install-dependencies: $(CL_PACKAGES)

$(CL_PACKAGES):
	@if ! [ -f "$(echo $@ | tr "/" "-").installed" ]; then \
		$(ROSWELL) install $@; \
		echo "Writing $(echo $@ | tr '/' '-').installed"; \
		touch "$(echo $@ | tr '/' '-').installed"; \
	else \
		echo "$@ is already installed"; \
	fi

test:
	$(ROSWELL) run -- \
		--eval "(ql:quickload :prove)" \
		--eval "(require :prove)" \
		--eval "(prove:run #P\"$(TESTS_FILE)\" :reporter :$(REPORTER))" \
		--non-interactive

work:
	$(ROSWELL) run -- \
		--eval "(ql:quickload :prove)" \
		--eval "(require :prove)" \
		--eval "(prove:run #P\"$(TEST_WORK_FILE)\" :reporter :$(REPORTER))" \
		--non-interactive

# TESTS_FILE="$(HOME)/common-lisp/bianet-neuron/bianet-neuron-tests.lisp"
# TEST_WORK_FILE="$(HOME)/common-lisp/bianet-neuron/bianet-neuron-test-work.lisp"
# LISP=/usr/bin/sbcl
# # Reporter can be list dot tap or fiveam.
# REPORTER=list
# test:
# 	$(LISP) --eval "(ql:quickload :prove)" \
# 	  --eval "(require :prove)" \
# 	  --eval "(prove:run #P\"$(TESTS_FILE)\" :reporter :$(REPORTER))" \
# 	  --non-interactive
# work:
# 	$(LISP) --eval "(ql:quickload :prove)" \
# 	  --eval "(require :prove)" \
# 	  --eval "(prove:run #P\"$(TEST_WORK_FILE)\" :reporter :$(REPORTER))" \
# 	  --non-interactive

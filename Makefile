TESTS_FILE="$(HOME)/common-lisp/bianet-neuron/bianet-neuron-tests.lisp"
TEST_WORK_FILE="$(HOME)/common-lisp/bianet-neuron/bianet-neuron-test-work.lisp"
LISP=/usr/bin/sbcl
# Reporter can be list dot tap or fiveam.
REPORTER=list
test:
	$(LISP) --eval "(ql:quickload :prove)" \
	  --eval "(require :prove)" \
	  --eval "(prove:run #P\"$(TESTS_FILE)\" :reporter :$(REPORTER))" \
	  --non-interactive
work:
	$(LISP) --eval "(ql:quickload :prove)" \
	  --eval "(require :prove)" \
	  --eval "(prove:run #P\"$(TEST_WORK_FILE)\" :reporter :$(REPORTER))" \
	  --non-interactive

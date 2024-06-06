(in-package :cl-user)
(require :bianet-neuron)
(require :prove)
(require :dc-dlist)
(require :dc-eclectic)
(require :vgplot)
(require :bianet-mesh)
(defpackage :bianet-neuron-test-work
  (:use :cl :prove :dc-eclectic :bianet-neuron :dc-dlist :sb-thread :bianet-mesh))
(in-package :bianet-neuron-test-work)

(setf prove:*enable-colors* t)

(defparameter *wait-timeout* 0.1)

(plan 1)

(let ((name "november"))
  (subtest (format nil "Simple 3-layer network ~s full trainining" name)
    (with-simple-network
        (neurons input-layer hidden-layers output-layer name 2 4 1)
      (let ((iterations 1000)
            (training-set #(((0 0) (0))
                            ((0 1) (1))
                            ((1 0) (1))
                            ((1 1) (0)))))
        (enable neurons)
        (loop with l = (length training-set)
              for i from 1 to iterations
              for frame = (aref training-set (mod i 4))
              do (train-frame input-layer output-layer frame))
        (ok (loop for (inputs expected-outputs) across training-set
                  for outputs = (feed-forward input-layer output-layer inputs)
                  for error = (output-layer-error output-layer expected-outputs)
                  do (diag
                      (format nil "(~f, ~f) -> (~,3f) [~f]; e=~,5f"
                              (first inputs) (second inputs)
                              (first outputs)
                              (first expected-outputs)
                              error))
                  finally (return (< error 0.05)))
            "XOR training successful")))))


(finalize)

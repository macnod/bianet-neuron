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

(defun round-to-n-decimals (x n)
  (float (/ (truncate (* x (expt 10 n))) (expt 10 n))))

(defun round-3 (x)
  (round-to-n-decimals x 3))

(defun feedforward (layer)
  (loop for neuron in layer do (transfer neuron))
  (loop for cx in (list-outgoing layer)
        do (setf (input (target cx)) 0.0))
  (loop for cx in (list-outgoing layer)
        do (incf (input (target cx))
                 (* (output (source cx)) (weight cx)))))

(defun backpropagate (layer)
  (loop for neuron in layer do (transfer-error neuron))
  (loop for cx in (list-incoming layer)
        do (setf (err-input (target cx)) 0.0))
  (loop for cx in (list-incoming layer)
        do (incf (err-input (source cx))
                 (* (err (target cx)) (weight cx)))
           (adjust-weight cx)))

(defun zero-inputs (neurons)
  (loop for neuron in neurons do (setf (input neuron) 0.0)))

(defun zero-err-inputs (neurons)
  (loop for neuron in neurons do (setf (err-input neuron) 0.0)))

(plan 1)

(let ((name "papa"))
  (subtest (format nil "Simple 8-layer network ~s full trainining" name)
    (with-simple-network
        (neurons input-layer hidden-layers output-layer name
                 2 12 1)
      (let ((iterations 16000)
            (training-set #(((0 0) (0))
                            ((0 1) (1))
                            ((1 0) (1))
                            ((1 1) (0)))))
        (enable neurons)
        (pass
         (loop with l = (length training-set)
               and start-time = (mark-time)
               for i from 1 to iterations
               for frame = (aref training-set (mod i 4))
               for start-time-tf = (mark-time)
               for tf-time = (progn
                               (train-frame input-layer output-layer frame)
                               (elapsed-time start-time-tf))
               for tf-max = tf-time then (if (> tf-time tf-max) tf-time tf-max)
               for tf-min = tf-time then (if (< tf-time tf-min) tf-time tf-min)
               summing tf-time into tf-total
               finally 
                  (return
                    (format 
                     nil
                     "pCnt=~:d pTot=~,3fs; pAvg=~,3fs; pMin=~,3fs; pMax=~,3fs"
                     iterations
                     (elapsed-time start-time)
                     (/ tf-total iterations)
                     tf-min
                     tf-max))))
        (loop with start-time = (mark-time)
              for (inputs expected-outputs) across training-set
              for outputs = (feed-forward input-layer output-layer inputs)
              for error = (output-layer-error output-layer expected-outputs)
              do (diag
                  (format nil "(~f, ~f) -> (~,3f) [~f]; e=~,5f"
                          (first inputs) (second inputs)
                          (first outputs)
                          (first expected-outputs)
                          error))
              finally 
                 (ok (< error 0.05)
                     (format
                      nil "XOR set inference successful after ~,1f seconds"
                      (elapsed-time start-time))))))))

(finalize)

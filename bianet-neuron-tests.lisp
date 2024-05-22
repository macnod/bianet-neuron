(in-package :cl-user)
(require :bianet-neuron)
(require :prove)
(require :dc-dlist)
(require :dc-eclectic)
(defpackage :bianet-neuron-tests
  (:use :cl :prove :dc-eclectic :bianet-neuron :dc-dlist :sb-thread))
(in-package :bianet-neuron-tests)

(plan 68)

;; next id
(is (bianet-neuron::next-neuron-id) 1 "next-neuron-id 1")
(is (bianet-neuron::next-neuron-id) 2 "next-neuron-id 2")
(is (bianet-neuron::next-cx-id) 1 "next-cx-id 1")
(is (bianet-neuron::next-cx-id) 2 "next-cx-id 2")
(is (bianet-neuron::next-neuron-id) 3 "next-neuron-id 3")
(is (bianet-neuron::next-cx-id) 3 "next-cx-id 3")

;; dlog
(ok (not (bianet-neuron::dlog "hello world"))
    "dlog returns nothing if without open log")
(ok (not (bianet-neuron::close-log))
    "close-log returns nothing without an open log")
(ok (bianet-neuron::open-log :filepath "/tmp/neurons-test.log" :append nil)
    "open-log returns something when the log is not already open")
(ok bianet-neuron::*log* "*log* is set after call to open-log")
(like (bianet-neuron::dlog "hello world") "^[-0-9T:]+ hello world$"
      "dlog returns the log entry when log file is open")
(like (bianet-neuron::dlog "hello ~a" "donnie") "^[-0-9T:]+ hello donnie$"
      "dlog returns the correct log entry for format string and parameters")
(ok (bianet-neuron::close-log) "close-log return T when log file is open")
(ok (not bianet-neuron::*log*) "*log* is NIL after call to close-log")

;; Forward-feed tests
;;
;; In these tests, we create a graph of neurons with 3 layers, where all 
;; the neurons of each layer are connected to all the neurons in the
;; following layer. The lares are input, hidden, and output. The input
;; layer has 3 neurons, the hidden layer has 2 neurons, and the output
;; layer has 2 neurons. It follows that there must be 10 connections.
;; Aside from verifying various aspects of the operation of a neuron,
;; these tests aim to eventually excite the 3 input-layer neurons,
;; causing a cascade of neuron firings that impacts every neuron.
(let* ((first-id (bianet-neuron::next-neuron-id-peek))
       (neurons (loop for a from 1 to 7
                      collect (make-instance 't-neuron)))
       (neuron-id-range (range first-id (+ first-id (1- (length neurons)))))
       (input-1 (nth 0 neurons))
       (input-2 (nth 1 neurons))
       (input-3 (nth 2 neurons))
       (hidden-1 (nth 3 neurons))
       (hidden-2 (nth 4 neurons))
       (output-1 (nth 5 neurons))
       (output-2 (nth 6 neurons))
       (inputs (list input-1 input-2 input-3))
       (hidden (list hidden-1 hidden-2))
       (outputs (list output-1 output-2)))
  (loop for source in inputs
        do (loop for target in hidden
                 do (connect source target)))
  (loop for source in hidden
        do (loop for target in outputs
                 do (connect source target)))

  ;; Test recently created neurons and connections
  (is (length (bianet-neuron::list-incoming neurons)) 10
      "10 connections in total")
  (is (len (incoming input-1)) 0
      "neuron input-1 has 0 incoming connections")
  (is (len (outgoing input-1)) 2
      "neuron input-1 has 2 outgoing connection")
  (is (len (incoming input-2)) 0
      "neuron input-2 has 0 incoming connections")
  (is (len (outgoing input-2)) 2
      "neuron input-2 has 2 outgoing connection")
  (is (len (incoming hidden-1)) 3
      "neuron hidden-1 has 3 incoming connections")
  (is (len (outgoing hidden-1)) 2
      "neuron hidden-1 has 2 outgoing connections")
  (is (len (incoming hidden-2)) 3
      "neuron hidden-2 has 3 incoming connections")
  (is (len (outgoing hidden-2)) 2
      "neuron hidden-2 has 2 outgoing connections")
  (is (len (incoming output-1)) 2
      "neuron output-1 has 2 incoming conections")
  (is (len (outgoing output-1)) 0
      "neuron output-1 has 0 outgoing connections")
  (is (len (incoming output-2)) 2
      "neuron output-2 has 2 incoming conections")
  (is (len (outgoing output-2)) 0
      "neuron output-2 has 0 outgoing connections")

  (ok (loop for neuron in neurons never (enabled neuron))
      "All neurons currently disabled")
  (ok (loop for neuron in neurons
            never (bianet-neuron::neuron-thread neuron))
      "No neuron has a neuron thread")
  (ok (loop for neuron in neurons never (biased neuron))
      "No neuron is biased")
  (ok (loop for neuron in neurons always (zerop (input neuron)))
      "All neurons start out with an input of 0")
  (ok (loop for neuron in neurons always (zerop (output neuron)))
      "All neurons start out with an output of 0")
  (ok (loop for neuron in neurons always (zerop (excitation-count neuron)))
      "No neuron has been excited")
  (ok (loop for neuron in neurons always (zerop (ff-count neuron)))
      "All neurons have a transfer count of 0")
  (ok (loop for neuron in neurons always (zerop (bp-count neuron)))
      "All neurons have a back-propagation count of 0")
  (is (loop for neuron in neurons collect (name neuron))
      (mapcar (lambda (a) (format nil "n-~d" a)) neuron-id-range)
      "Neuron names are correct")

  ;; Enable neurons
  (enable neurons)
  (ok (loop for neuron in neurons
            always (and (enabled neuron)
                        (bianet-neuron::neuron-thread neuron)))
      "All neurons enabled and each has a neuron thread")
  (ok (loop for neuron in neurons always (zerop (ff-count neuron)))
      "No neurons have experienced transfer yet")
  (ok (loop for neuron in neurons 
            always (and (zerop (input neuron))
                        (zerop (output neuron))))
      "All neurons still have inputs and outputs set to 0.0")
  (is (sort (mapcar #'thread-name (list-neuron-threads)) #'string<)
      (sort (mapcar (lambda (a) 
                      (format nil "~a~:d" 
                              bianet-neuron::*neuron-thread-name-prefix* a))
                    neuron-id-range)
            #'string<)
      "list-neuron-threads returns a thread for each neuron")
  (ok (loop for neuron in neurons
            always (thread-alive-p (bianet-neuron::neuron-thread neuron)))
      "All neuron threads are alive")

  ;; Excite neuron input-1
  (ok (excite input-1 1.0) "Neuron input-1 excited")
  (sleep 0.1)
  ;; Check input-1
  (ok (not (zerop (output input-1)))
      "Output of neuron input-1 is no longer 0")
  (is (ff-count input-1) 1
      "Neuron input-1 experienced a transfer")
  (ok (zerop (excitation-count input-1))
      "Neuron input-1 was excited but transfer cleared that")
  (ok (zerop (modulation-count input-1))
      "Neuron input-1 modulation-count is 0")
  ;; Check the rest of the input neurons
  (ok (loop for neuron in (cdr inputs)
            always (and (zerop (input neuron))
                        (zerop (output neuron))
                        (zerop (ff-count neuron))
                        (zerop (excitation-count neuron))
                        (zerop (modulation-count neuron))))
      "The rest of the input neurons are unaffected")
  (ok (loop for neuron in hidden
            always (and (zerop (output neuron))
                        (zerop (ff-count neuron))
                        (zerop (modulation-count neuron))))
      "The hidden neurons are unaffected, except input and excitation-count")
  (ok (loop for neuron in outputs
            always (and (zerop (input neuron))
                        (zerop (output neuron))
                        (zerop (ff-count neuron))
                        (zerop (excitation-count neuron))
                        (zerop (modulation-count neuron))))
      "The output neurons are completely unaffected")
  (loop for neuron in hidden
        for index = 1 then (1+ index)
        do (is (excitation-count neuron) 1
               (format nil "Neuron hidden-~d excitation count is 1" index)))

  ;; For reference, before exciting neuron input-2
  (ok (loop for neuron in neurons
            thereis (zerop (output neuron)))
      "Not all neuron outputs have non-zero values")
  (ok (loop for neuron in neurons
            thereis (zerop (ff-count neuron)))
      "Not all neruons have fired")
  ;; Excite neuron input-2
  (ok (excite input-2 1.0) "Neuron input-2 excited")
  (sleep 0.1)
  (ok (not (zerop (output input-2)))
      "Output of neuron input-2 is no longer 0")
  (is (ff-count input-2) 1
      "Neuron input-2 experienced a transfer")
  (ok (zerop (excitation-count input-2))
      "Neuron input-2 was excited but transfer cleared that")
  (ok (zerop (modulation-count input-2))
      "Neuron input-2 modulation-count is 0")
  ;; Check the rest of the inputs (input-3)
  (ok (and (zerop (input input-3)) 
           (zerop (output input-3))
           (zerop (ff-count input-3))
           (zerop (excitation-count input-3))
           (zerop (modulation-count input-3)))
      "Neuron input-3 remains unaffected after exciting input-2")
  (ok (loop for neuron in hidden
            always (and (zerop (output neuron))
                        (zerop (ff-count neuron))
                        (zerop (modulation-count neuron))))
      "The hidden neurons are unaffected, except input and excitation-count")
  (loop for neuron in hidden
        for index = 1 then (1+ index)
        do (is (excitation-count neuron) 2
               (format nil "Neuron hidden-~d excitation count is 2" index)))

  ;; Excite neuron input-3. Because this is the last input neuron, exciting it
  ;; should cause a cascade effect where all the neurons become excited, with
  ;; all of the inputs becoming non-zero, then being reset to zero. All
  ;; outputs should be non-zero. ff-count should also take the value of 1 in
  ;; every neuron. excitation-count will increase, but will be reset back to
  ;; zero as soon as each neuron fires, and all neurons should fire.
  (ok (excite input-3 1.0) "Neuron input-3 excited")
  (sleep 0.1)
  (ok (loop for neuron in neurons
            always (zerop (input neuron)))
      "All neuron inputs have been reset to 0.0")
  (ok (loop for neuron in neurons
            always (not (zerop (output neuron))))
      "All neuron outputs have non-zero values")
  (ok (loop for neuron in neurons
            always (= (ff-count neuron) 1))
      "Every neuron has fired")
  (ok (loop for neuron in neurons
            always (zerop (excitation-count neuron)))
      "excitation-count has been reset to 0 in every neuron")
  (ok (every (lambda (a b) (< (abs (- a b)) 1e-6))
             (list-outputs outputs)
             (list 0.50840926 0.50916994))
      "neuron outputs are within tolerance"))

(finalize)

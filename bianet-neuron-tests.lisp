(in-package :cl-user)
(require :bianet-neuron)
(require :prove)
(require :dc-dlist)
(require :dc-eclectic)
(defpackage :bianet-neuron-tests
  (:use :cl :prove :dc-eclectic :bianet-neuron :dc-dlist :sb-thread))
(in-package :bianet-neuron-tests)

(plan 49)

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

;; t-neuron
(let* ((first-id (bianet-neuron::next-neuron-id-peek))
       (neurons (list (make-instance 't-neuron)
                      (make-instance 't-neuron)
                      (make-instance 't-neuron)))
       (neuron-id-range (range first-id (+ first-id (1- (length neurons))))))
  (connect (nth 0 neurons) (nth 2 neurons))
  (connect (nth 1 neurons) (nth 2 neurons))

  ;; Test recently created neurons and connections
  (is (length (list-cxs neurons)) 2 "2 connections in total")
  (is (len (incoming (nth 0 neurons))) 0
      "first neuron has 0 incoming connections")
  (is (len (outgoing (nth 0 neurons))) 1
      "first neuron has 1 outgoing connection")
  (is (len (incoming (nth 1 neurons))) 0
      "second neuron has 0 incoming connections")
  (is (len (outgoing (nth 1 neurons))) 1
      "second neuron has 1 outgoing connection")
  (is (len (incoming (nth 2 neurons))) 2
      "third neuron has 2 incoming connections")
  (is (len (outgoing (nth 2 neurons))) 0
      "third neuron has 0 outgoing connections")
  (ok (loop for neuron in neurons never (enabled neuron))
      "All neurons currently disabled")
  (ok (loop for neuron in neurons
            never (bianet-neuron::activation-thread neuron))
      "No neuron has an activation thread")
  (ok (loop for neuron in neurons never (biased neuron))
      "No neuron is biased")
  (ok (loop for neuron in neurons always (zerop (input neuron)))
      "All neurons start out with an input of 0")
  (ok (loop for neuron in neurons always (zerop (output neuron)))
      "All neurons start out with an output of 0")
  (ok (loop for neuron in neurons always (zerop (received neuron)))
      "No neuron has been excited")
  (ok (loop for neuron in neurons always (zerop (transfer-count neuron)))
      "All neurons have a transfer count of 0")
  (is (loop for neuron in neurons collect (name neuron))
      (mapcar (lambda (a) (format nil "n-~d" a)) neuron-id-range)
      "Neuron names are correct")

  ;; Enable neurons
  (enable neurons)
  (ok (loop for neuron in neurons
            always (and (enabled neuron)
                        (bianet-neuron::activation-thread neuron)))
      "All neurons enabled and each has activation thread")
  (ok (loop for neuron in neurons always (zerop (transfer-count neuron)))
      "No neurons have experienced transfer yet")
  (ok (loop for neuron in neurons always
                                  (and (zerop (input neuron))
                                       (zerop (output neuron))))
      "All neurons still have inputs and outputs set to 0")
  (is (sort (mapcar #'thread-name (list-neuron-threads)) #'string<)
      (sort (mapcar (lambda (a) (format nil "nt-~d" a)) neuron-id-range) #'string<)
      "list-neuron-threads returns a thread for each neuron")
  (ok (loop for neuron in neurons
            always (thread-alive-p (bianet-neuron::activation-thread neuron)))
      "All neuron threads are alive")

  ;; Excite input neuron nt-4
  (ok (excite (nth 0 neurons) 1.0) "Neuron nt-4 excited")
  (sleep 0.1)
  (ok (not (zerop (output (nth 0 neurons))))
      "Output of neuron nt-4 is no longer 0")
  (is (transfer-count (nth 0 neurons)) 1
      "Neuron nt-4 experienced a transfer")
  (ok (zerop (received (nth 0 neurons)))
      "Neuron nt-4 was excited but transfer cleared that")
  (ok (zerop (output (nth 1 neurons)))
      "Output of neuron nt-5 is still 0")
  (ok (zerop (transfer-count (nth 1 neurons)))
      "Neuron nt-5 has not experienced a transfer")
  (ok (zerop (received (nth 1 neurons)))
      "Neuron nt-5 has been not been excited")
  (ok (zerop (output (nth 2 neurons)))
      "Output of neuron nt-6 is still 0")
  (ok (zerop (transfer-count (nth 2 neurons)))
      "Nueron nt-6 has not experienced a transfer")
  (is (received (nth 2 neurons)) 1
      "Neuron nt-6 has been excited 1 time")

  ;; Excite input neuron nt-5
  (ok (excite (nth 1 neurons) 1.0) "Neuron nt-5 excited")
  (sleep 0.1)
  (ok (not (zerop (output (nth 1 neurons))))
      "Output of neuron nt-5 is no longer 0")
  (is (transfer-count (nth 1 neurons)) 1 "Neuron nt-5 experienced a transfer")
  (is (transfer-count (nth 2 neurons)) 1 "Neuron nt-6 experienced a transfer")
  (ok (not (zerop (output (nth 2 neurons))))
      "Output of neuron nt-6 is no longer 0"))


(finalize)

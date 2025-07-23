(in-package :cl-user)

(require :prove)
(require :dc-dlist)
(require :dc-eclectic)
(require :bianet-mesh)

(pushnew (truename ".") asdf:*central-registry* :test 'equal)
(asdf:load-system :bianet-neuron)

(defpackage :bianet-neuron-tests
  (:use :cl :prove :dc-eclectic :dc-dlist :sb-thread :bianet-neuron :bianet-mesh))
(in-package :bianet-neuron-tests)

(setf prove:*enable-colors* t)

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

(plan 18)

(subtest "Neuron and connection IDs"
  (is (bianet-neuron::next-neuron-id) 1 "next-neuron-id 1")
  (is (bianet-neuron::next-neuron-id) 2 "next-neuron-id 2")
  (is (bianet-neuron::next-cx-id) 1 "next-cx-id 1")
  (is (bianet-neuron::next-cx-id) 2 "next-cx-id 2")
  (is (bianet-neuron::next-neuron-id) 3 "next-neuron-id 3")
  (is (bianet-neuron::next-cx-id) 3 "next-cx-id 3"))

(subtest "Logging"
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
  (ok (not bianet-neuron::*log*) "*log* is NIL after call to close-log"))

(subtest "Connection weight adjustment"
  (let ((name "alfa"))
    (with-neurons (neurons 2 name)
      (let* ((a (first neurons))
             (b (second neurons))
             (name-a (name a))
             (name-b (name b))
             (a-bp (bp-count a))
             (b-ff (ff-count b))
             (cx (connect a b :weight 0.5 :learning-rate 0.1 :momentum 0.1)))
        (is (name (source cx)) name-a "cx source is a")
        (is (name (target cx)) name-b "cx target is b")
        (is (weight cx) 0.5 "cx weight is 0.5")
        (enable neurons)
        (ok (excite a 1.0) "excite a")
        (ok (wait-for-output b (1+ b-ff) 0.01) "wait for b to output")
        (is (input a) 0.0 "input of a has been cleared to 0.0")
        (is (round-3 (input b)) 0.0 "input of b is 0.0")
        (let ((expected (round-3 0.7310586)))
          (is (round-3 (output a)) expected 
              (format nil "output of a is ~f" expected)))
        (is (round-3 (output b)) 0.59 "output of b is 0.59")
        (is (input b) 0.0 "input of b has been cleared to 0.0")
        (let ((expected (round-3 0.5903783)))
          (is (round-3 (output b)) expected
              (format nil "output of b is ~f" expected))
          (is (round-3 (- (output b))) (- expected)
              (format nil "b expected output - actual output is ~f" 
                      (- expected))))
        (ok (modulate b (- 0.0 (output b))) "modulate cx with 0.5")
        (ok (wait-for-backprop a (1+ a-bp) 0.01) "wait for backprop to a")))))

(subtest "Rapidly exciting a single, isolated neuron"
  (let* ((name "bravo")
         (input-count 1e5)
         (processed 0)
         (success (loop with neuron = (make-instance 't-neuron :name name)
                          initially (enable neuron)
                        for input from 0.0 below 1.0 by (/ 1.0 input-count)
                        for ff-count = (ff-count neuron)
                        do (excite neuron input)
                           (incf processed)
                        always (wait-for-output-p neuron (1+ ff-count) 1.0)
                        finally (disable neuron))))
    (if success
        (pass (format nil "~:d excitations run successfully"
                      (truncate input-count)))
        (fail (format nil "~:d excitations fail at input #~d" 
                      (truncate input-count) processed)))
    (ok (zerop (length (list-neuron-threads name)))
        "No neuron threads remain.")))

(subtest "Rapidly modulate a single, isolated neuron"
  (let* ((name "charlie")
         (err-count 1e5)
         (processed 0)
         (success (loop with neuron = (make-instance 't-neuron :name name)
                          initially (enable neuron)
                        for err from 0.0 below 1.0 by (/ 1.0 err-count)
                        for bp-count = (bp-count neuron)
                        do (modulate neuron err)
                           (incf processed)
                        always (wait-for-backprop-p neuron (1+ bp-count) 1.0)
                        finally (disable neuron))))
    (if success
        (pass (format nil "~:d modulations run successfully"
                      (truncate err-count)))
        (fail (format nil "~:d modulations fail at err #~d"
                      (truncate err-count) processed)))
    (ok (zerop (length (list-neuron-threads name)))
        "No neuron threads remain.")))

(subtest "Rapidly excite and modulate a single, isolated neuron"
  (let* ((name "delta")
         (iteration-count 1e5)
         (processed 0)
         (success (loop with neuron = (make-instance 't-neuron :name name)
                          initially (enable neuron)
                        for i from 0 below iteration-count
                        for ff-count = (ff-count neuron)
                        for bp-count = (bp-count neuron)
                        do (excite neuron 0.5)
                        unless (wait-for-output-p neuron (1+ ff-count) 1.0)
                          do (return nil)
                        do (modulate neuron 0.5)
                        unless (wait-for-backprop-p neuron (1+ bp-count) 1.0)
                          do (return nil)
                        finally (disable neuron)
                                (return t))))
    (if success
        (pass (format nil "~:d iterations run successfully"
                      (truncate iteration-count)))
        (fail (format nil "~:d iterations fail at iteration #~d"
                      (truncate iteration-count) processed)))
    (ok (zerop (length (list-neuron-threads name)))
        "No neuron threads remain.")))

(subtest "Enable neurons in body of with-neurons macro"
  (let ((name "foxtrot"))
    (with-neurons (neurons 5 name)
      (is (length neurons) 5 
          (format nil "Created 5 ~s neuroms" name))
      (is (length (list-neuron-threads name)) 0 
          (format nil "No ~s threads yet" name))
      (enable neurons)
      (is (length (list-neuron-threads name)) 5 
          (format nil "5 ~s threads now" name)))
    (is (length (list-neuron-threads name)) 0
        (format nil "All ~s threads cleand up" name))))

(subtest "Connect 2 neurons manually"
  (let ((name "golf"))
    (with-neurons (neurons 2 name)
      (is-type (connect (first neurons) (second neurons) :weight 0.5)
               't-cx
               (format nil "Connected two ~s neurons" name))
      (is (length (list-outgoing neurons)) 1 "One connection total")
      (is (length (list-outgoing (first neurons))) 1
          "One outgoing connection for the first neuron")
      (ok (not (list-outgoing (second neurons)))
          "No outgoing connections for the second neuron")
      (ok (not (list-incoming (first neurons)))
          "No incoming connections for the first neuron")
      (is (length (list-incoming (second neurons))) 1
          "One incoming connection for the second neuron")
      (let ((cx (value (head (outgoing (first neurons))))))
        (is (source cx) (first neurons)
            "Source of connection is the first neuron")
        (is (target cx) (second neurons) 
            "Target of connection is the second neuron")))))

(subtest "2 relu neurons with 1 connection 1 -> 1; 0 -> 0"
  (let ((name "hotel"))
    (with-neurons (neurons 2 name)
      (loop for neuron in neurons do
        (setf (transfer-function neuron) #'bianet-neuron::relu)
        (setf (transfer-derivative neuron) #'bianet-neuron::relu-derivative))
      (let* ((a (first neurons))
             (b (second neurons))
             (cx (connect a b :weight 0.5 :learning-rate 0.1 :momentum 0.1))
             (input 1.0)
             (expected-output 1.0))
        (is (name (source cx)) (format nil "~a-000000001" name)
            (format nil "cx source is ~a-000000001" name))
        (is (name (target cx)) (format nil "~a-000000002" name)
            (format nil "cx target is ~a-000000002" name))
        (enable neurons)
        (let ((ff (ff-count b))
              (bp (bp-count a)))
          (ok (excite a input) 
              (format nil "excite a with ~f" input))
          (ok (wait-for-output b (1+ ff) 0.01) "wait for b to output")
          (is (last-input a) input
              (format nil "a's last input is ~f" input))
          (is (output a) input (format nil "a's output is ~f" input))
          (is (last-input b) 0.5 "b's last input is 0.5")
          (is (output b) 0.5 "output of b is 0.5")
          (let ((err (- expected-output (output b))))
            (is err 0.5 "error of b is 0.5")
            (is (ff-count a) (1+ ff) "a has ff-count of 1 + ff")
            (is (ff-count b) (1+ ff) "b has ff-count of 1 + ff")
            (is (bp-count a) 0 "bp count of a is still 0")
            (is (bp-count b) 0 "bp count of b is still 0")
            (is (weight cx) 0.5 "cx weight is 0.5")
            (ok (zerop (update-count cx)) "no updates to weight")
            (ok (modulate b err) "modulate b with error 0.5")
            (ok (wait-for-backprop a (1+ bp) 0.1) "wait for backpropagation")
            (is (update-count cx) 1 "1 update to weight")
            (is (weight cx) 0.55 "cx weight has changed from 0.5 to 0.55")
            (is (last-err-input b) err "last-err-input b is 0.5")
            (is (err b) 0.5 "err b is 0.5")
            (is (last-err-input a) (* 1.0 0.25) "last-err-input a is 0.25")
            (is (err a) 0.25 "err a is 0.25")))))))

(let ((name "india"))
  (with-neurons (neurons 10 name)
    (let* ((input-layer (subseq neurons 0 2))
           (hidden-layer (subseq neurons 2 9))
           (output-layer (subseq neurons 9 10))
           (biased-neuron (nth 8 neurons))
           (cx-count (+ (* (length input-layer) (1- (length hidden-layer)))
                        (* (length hidden-layer) (length output-layer))))
           (weights (loop with step = (/ pi cx-count)
                          for a from 1 to cx-count
                          for b = 0.0 then (+ b step)
                          collect (sin b)))
           (weights-stack (copy-seq weights)))

      (subtest "Create 2 13 1 network with one bias and sin weights"
        (setf (biased biased-neuron) t)
        (let ((biased-functions (getf bianet-neuron::*transfer-functions*
                                      :biased)))
          (setf (transfer-function biased-neuron)
                (getf biased-functions :function))
          (setf (transfer-derivative biased-neuron)
                (getf biased-functions :derivative)))
        ;; Input-layer and hidden-layer neurons use relu
        (loop for neuron in (append input-layer output-layer)
              do (setf (transfer-function neuron)
                       #'bianet-neuron::relu)
                 (setf (transfer-derivative neuron)
                       #'bianet-neuron::relu-derivative))
        ;; Output-layer neurons use logistic
        (loop for neuron in output-layer
              do (setf (transfer-function neuron)
                       #'bianet-neuron::logistic)
                 (setf (transfer-derivative neuron)
                       #'bianet-neuron::logistic-derivative))
        ;; Connect input-layer neurons to hidden-layer neurons
        (loop for source in input-layer do
          (loop for target in hidden-layer
                when (not (biased target))
                  do (let ((weight (pop weights-stack)))
                       (connect source target 
                                :weight weight
                                :learning-rate 0.1
                                :momentum weight))))
        ;; Connect hidden-layer neurons to output-layer neurons
        (loop for source in hidden-layer do
          (loop for target in output-layer
                do (let ((weight (pop weights-stack)))
                     (connect source target 
                              :weight weight
                              :learning-rate 0.3
                              :momentum weight))))
        (is (length (list-outgoing neurons)) cx-count
            "Correct number of connections"))
      
      (subtest "Backpropagate rest"
        (diag (format nil "Weights: ~a" weights))
        (loop 
          with training-set = '#(((0.0 0.0) (0.0))
                                 ((0.0 1.0) (1.0))
                                 ((1.0 0.0) (1.0))
                                 ((1.0 1.0) (0.0)))
          and iterations = 1000
          and report-count = 2
          for a from 1 to iterations
          do
             (loop for count from 0 below (length training-set)
                   for (inputs expected) across training-set
                   do (zero-inputs neurons)
                      (apply-inputs input-layer inputs)
                      (feedforward input-layer)
                      (feedforward hidden-layer)
                      (feedforward output-layer)
                      (zero-err-inputs neurons)
                      (apply-error output-layer expected)
                      (backpropagate output-layer)
                      (backpropagate hidden-layer))
          when (zerop (mod a (/ iterations report-count)))
            do (diag (format nil "Iteration ~:d" a))
               (loop for (inputs expected) across training-set
                     do 
                        (zero-inputs neurons)
                        (apply-inputs input-layer inputs)
                        (feedforward input-layer)
                        (feedforward hidden-layer)
                        (feedforward output-layer)
                        (diag (format nil "~a -> ~a, expected ~a"
                                      inputs
                                      (mapcar #'output output-layer)
                                      expected))))))))

(subtest "The with-simple-network macro assembles a network correctly"
  (let ((name "juliet"))
    (with-simple-network 
        (neurons input-layer hidden-layers output-layer name 9 3 2)
      (is (length neurons) 15 (format nil "Created 15 ~s neurons" name))
      (is (length input-layer) 9 "9 input neurons")
      (is (length hidden-layers) 4
          "4 neurons in the hidden layer (3 neurons + 1 bias)")
      (is (length output-layer) 2 "2 output neurons")
      (is (length (remove-if-not #'biased neurons)) 1
          "1 biased neuron in the network")
      (is (length (list-outgoing neurons)) 44 "44 connections total")
      (is (length (list-outgoing input-layer)) 36
          "36 outgoing connections from the input layer")
      (is (length (list-outgoing hidden-layers)) 8
          "8 outgoing connections from the hidden layer")
      (ok (not (list-outgoing output-layer))
          "No outgoing connections from the output layer")
      (ok (not (list-incoming input-layer))
          "No incoming connections to the input layer")
      (is (length (list-incoming hidden-layers)) 36
          "36 incoming connections to the hidden layer")
      (is (length (list-incoming output-layer)) 8
          "8 incoming connections to the output layer")
      (is (length (list-neuron-threads "golf")) 0 
          (format nil "No ~s threads yet" name)))))

(let ((name "kilo"))
  (subtest (format nil "Simple 3-layer network ~s does feedforward" name)
    (with-simple-network
        (neurons input-layer hidden-layers output-layer name 2 3 2)
      (enable neurons)
      (ok (every (lambda (n) (zerop (output n))) output-layer)
          "Output neurons are zeroed")
      (loop for a from 1 to 10 
            for ff-count = (ff-count (first input-layer))
            do (loop for neuron in input-layer do (excite neuron 0.5))
            collect (wait-for-output-p output-layer (1+ ff-count) 0.1)
              into result
            finally (ok (every #'identity result)
                        "All feedforward passes succeed")
                    (ok (every (lambda (n) (not (zerop (output n)))) output-layer)
                        "Output neurons are non-zero"))))
  (is (length (list-neuron-threads name)) 0 
      (format nil "All ~s threads cleand up" name)))

(let ((name "lima"))
  (subtest (format 
            nil 
            "Simple 3-layer network ~s does feedforward and backpropagation"
            name)
    (with-simple-network
        (neurons input-layer hidden-layers output-layer name 2 3 2)
      (enable neurons)
      (ok (every (lambda (n) (zerop (output n))) output-layer)
          "Output neurons have zerop output")
      (ok (every (lambda (n) (zerop (err n))) input-layer)
          "Input neurons have zero error")
      (loop for a from 1 to 10
            for ff-count = (ff-count (first input-layer))
            for bp-count = (bp-count (first input-layer))
            do (loop for neuron in input-layer do (excite neuron 0.5))
            collect (wait-for-output-p output-layer (1+ ff-count) 0.1) 
              into ff-result
            do (loop for neuron in output-layer do (modulate neuron 0.5))
            collect (wait-for-backprop-p input-layer (1+ bp-count) 0.1)
              into bp-result
            finally (ok (every #'identity ff-result)
                        "All feedforward passes succeed")
                    (ok (every #'identity bp-result)
                        "All backpropagation passes succeed")
                    (ok (every (lambda (n) (not (zerop (output n)))) 
                               output-layer)
                        "Output neurons have non-zero output")
                    (ok (every (lambda (n) (not (zerop (err n)))) input-layer)
                        "Input neurons have non-zero error")))))

(let ((name "mike"))
  (subtest (format nil "Simple 3-layer network ~s learns" name)
    (with-simple-network
        (neurons input-layer hidden-layers output-layer name 2 3 2)
      (let ((iterations 10)
            (training-set '(((0 0) (0 1))
                            ((0 1) (1 0))
                            ((1 0) (1 0))
                            ((1 1) (0 1)))))
        (enable neurons)
        (loop for i from 1 to iterations
              for ff-count = (ff-count (first input-layer))
              for bp-count = (bp-count (first input-layer))
              for frame-index = (random (length training-set))
              for (inputs expected-outputs) = (nth frame-index training-set)
              for excitements = (loop for neuron in input-layer
                                      for input in inputs
                                      for excitement = (excite neuron input)
                                      when excitement collect excitement)
              for ff-wait = (when excitements
                              (wait-for-output-p output-layer 
                                                 (1+ ff-count) 0.1))
              for outputs = (when ff-wait
                              (mapcar #'output output-layer))
              for errors = (when outputs
                             (mapcar (lambda (e o) (- e o))
                                     expected-outputs outputs))
              for modulations = (when errors
                                  (loop for neuron in output-layer
                                        for error in errors
                                        always (modulate neuron error)))
              for bp-wait = (when modulations
                              (wait-for-backprop-p input-layer
                                                   (1+ bp-count) 
                                                   0.1))
              finally (ok bp-wait "Network learns"))))))

(let ((name "november"))
  (subtest (format nil "Simple 3-layer network ~s learns with train-frame" name)
    (with-simple-network
        (neurons input-layer hidden-layers output-layer name 2 7 1)
      (let ((iterations 10)
            (training-set '(((0 0) (0))
                            ((0 1) (1))
                            ((1 0) (1))
                            ((1 1) (0)))))
        (enable neurons)
        (ok (loop with l = (length training-set)
                  for i from 1 to iterations
                  for frame in training-set
                  always (train-frame input-layer output-layer frame))
            "Network learns")))))

(let ((name "oscar"))
  (subtest (format nil "Simple 3-layer network ~s full trainining" name)
    (with-simple-network
        (neurons input-layer hidden-layers output-layer name 2 4 1)
      (let ((iterations 10000)
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
               for error = (car (train-frame input-layer output-layer frame))
               for error-history = (list error) then (progn
                                                       (push error error-history)
                                                       (if (< (length error-history) l)
                                                         error-history
                                                         (subseq error-history 0 l)))
               for done = (when (= (length error-history) l)
                            (< (apply #'max error-history) 0.03))
               for tf-time = (elapsed-time start-time-tf)
               for tf-max = tf-time then (if (> tf-time tf-max) tf-time tf-max)
               for tf-min = tf-time then (if (< tf-time tf-min) tf-time tf-min)
               while (not done)
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

;; (let ((name "papa"))
;;   (subtest (format nil "Simple 8-layer network ~s full trainining" name)
;;     (with-simple-network
;;         (neurons input-layer hidden-layers output-layer name
;;                  2 8 4 1)
;;       (let ((iterations 20000)
;;             (training-set #(((0 0) (0))
;;                             ((0 1) (1))
;;                             ((1 0) (1))
;;                             ((1 1) (0)))))
;;         (enable neurons)
;;         (pass
;;          (loop with l = (length training-set)
;;                and start-time = (mark-time)
;;                for i from 1 to iterations
;;                for frame = (aref training-set (mod i 4))
;;                for start-time-tf = (mark-time)
;;                for tf-time = (progn
;;                                (train-frame input-layer output-layer frame)
;;                                (elapsed-time start-time-tf))
;;                for tf-max = tf-time then (if (> tf-time tf-max) tf-time tf-max)
;;                for tf-min = tf-time then (if (< tf-time tf-min) tf-time tf-min)
;;                summing tf-time into tf-total
;;                finally 
;;                   (return
;;                     (format 
;;                      nil
;;                      "pCnt=~:d pTot=~,3fs; pAvg=~,3fs; pMin=~,3fs; pMax=~,3fs"
;;                      iterations
;;                      (elapsed-time start-time)
;;                      (/ tf-total iterations)
;;                      tf-min
;;                      tf-max))))
;;         (loop with start-time = (mark-time)
;;               for (inputs expected-outputs) across training-set
;;               for outputs = (feed-forward input-layer output-layer inputs)
;;               for error = (output-layer-error output-layer expected-outputs)
;;               do (diag
;;                   (format nil "(~f, ~f) -> (~,3f) [~f]; e=~,5f"
;;                           (first inputs) (second inputs)
;;                           (first outputs)
;;                           (first expected-outputs)
;;                           error))
;;               finally 
;;                  (ok (< error 0.05)
;;                      (format
;;                       nil "XOR set inference successful after ~,1f seconds"
;;                       (elapsed-time start-time))))))))

(finalize)


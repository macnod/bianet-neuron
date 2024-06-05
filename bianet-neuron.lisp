(in-package :bianet-neuron)

(defparameter *log* nil)

(defparameter *next-neuron-id* 0)
(defparameter *neuron-id-mutex* (make-mutex :name "neuron-id"))

(defparameter *next-cx-id* 0)
(defparameter *cx-id-mutex* (make-mutex :name "cx-id"))

(defparameter *log-mutex* (make-mutex :name "log"))
(defparameter *log-index* 0)

(defparameter *default-learning-rate* 0.02)
(defparameter *default-momentum* 0.1)
(defparameter *default-min-weight* -0.9)
(defparameter *default-max-weight* 0.9)

(defparameter *neuron-name-prefix* "n-")
(defparameter *testing* t)

(defun next-neuron-id ()
  (with-mutex (*neuron-id-mutex*)
    (incf *next-neuron-id*)))

(defun next-neuron-id-peek ()
  (with-mutex (*neuron-id-mutex*)
    (1+ *next-neuron-id*)))

(defun next-cx-id ()
  (with-mutex (*cx-id-mutex*)
    (incf *next-cx-id*)))

(defun next-cx-id-peek ()
  (with-mutex (*cx-id-mutex*)
    (1+ *next-cx-id*)))

(defun open-log (&key (filepath "/tmp/neurons.log") (append t))
  "Opens a log file, allowing the DLOG function to cease to be a
no-op. FILEPATH represents the path to the log file. APPEND indicates
that if a file exists at FILEPATH, call to dlog should append log
entries to the end of the existing file. If APPEND is NIL, the file at
FILEPATH is cleared. Regardless of the value of APPEND, if the file at
FILEPATH doesn't exist, this function creates it.

If *LOG* is set (if this function was called and CLOSE-LOG was never
called), then this function does nothing and returns NIL. If *LOG* is
NIL (if this function has not been called or it was called and then
CLOSE-LOG was called), then this function opens the log file, sets
*LOG* to the file stream, and returns the file stream."
  (unless *log*
    (setf *log* (open filepath
                      :direction :output
                      :if-exists (if append :append :supersede)
                      :if-does-not-exist :create))))

(defun close-log()
  "Closes the file stream that was opened by OPEN-LOG. If a file stream
is not open (if *LOG* is NIL), then this function does nothing and
returns NIL. If a file stream is open (*LOG* contains a stream), then
this fucntion closes the stream and returns T."
  (when *log*
    (close *log*)
    (setf *log* nil)
    t))

(defun dlog (format-string &rest values)
  "If the log file is open (see OPEN-LOG), this function creates a string
by calling FORMAT with FORMAT-STRING and with VALUES, writes the
string to the log stream, and returns the string. If the log file is
not open, this function does nothing."
  (when *log*
    (apply #'log-it (append (list *log* format-string) values))))

(defun make-random-weight-fn (&key (min -0.5) (max 0.5))
  (lambda (&key rstate
             global-index
             global-fraction
             layer-fraction
             neuron-fraction)
    (declare (ignore global-index global-fraction layer-fraction
                     neuron-fraction))
    (+ min (random (- max min) rstate))))

(defun make-progressive-weight-fn (&key (min -0.5) (max 0.5))
  (lambda (&key rstate
             global-index
             global-fraction
             layer-fraction
             neuron-fraction)
    (declare (ignore rstate global-index global-fraction neuron-fraction))
    (+ min (* layer-fraction (- max min)))))

(defun make-sinusoid-weight-fn (&key (min -0.5) (max 0.5))
  (lambda (&key rstate
             global-index
             global-fraction
             layer-fraction
             neuron-fraction)
    (declare (ignore rstate global-fraction layer-fraction neuron-fraction))
    (+ (* (/ (+ (sin global-index) 1) 2) (- max min)) min)))

;; (defun logistic (x)
;;   (declare (single-float x)
;;            (optimize (speed 3) (safety 0)))
;;   (cond ((> x (the single-float 16.64)) (the single-float 1.0))
;;         ((< x (the single-float -88.7)) (the single-float 0.0))
;;         ((< (the single-float (abs x)) (the single-float 1e-8))
;;          (the single-float 0.5))
;;         (t (/ (the single-float 1.0) 
;;               (the single-float (1+ (the single-float (exp (- x)))))))))

(defun logistic (x)
  (/ 1.0 (1+ (exp (- x)))))

;; (defun logistic (x)
;;   (cond ((> x 16.64) 1.0)
;;         ((< x -88.7) 0.0)
;;         ((< (abs x) 1e-8) 0.5)
;;         (t (/ 1 (1+ (exp (- x)))))))

(defun logistic-slow (x)
  (cond ((> x 16.64) 1.0)
        ((< x -88.7) 0.0)
        ((< (abs x) 1e-8) 0.5)
        (t (/ 1 (1+ (exp (- x)))))))

;; (defun logistic-derivative (x)
;;   (declare (single-float x)
;;            (optimize (speed 3) (safety 0)))
;;   (* x (- (the single-float 1.0) x)))

(defun logistic-derivative (x)
  (* x (- 1.0 x)))

(defun relu (x)
  (max 0.0 x))

(defun relu-derivative (x)
  (if (> x 0) 1.0 0.0))

(defun relu-leaky (x)
  (max 0.0 x))

(defun relu-leaky-derivative (x)
  (if (<= x 0.0) 0.001 1.0))

(defparameter *transfer-functions*
  (list :logistic (list :function #'logistic
                        :derivative #'logistic-derivative)
        :relu (list :function #'relu
                    :derivative #'relu-derivative)
        :relu-leaky (list :function #'relu-leaky
                          :derivative #'relu-leaky-derivative)
        :biased (list :function (lambda (x)
                                  (declare (ignore x))
                                  1.0)
                      :derivative (lambda (x)
                                    (declare (ignore x))
                                    0.0))))

(defclass t-cx ()
  ((id :reader id :type integer :initform (next-cx-id))
   (source :reader source :initarg :source :type t-neuron
           :initform (error ":source required"))
   (target :reader target :initarg :target :type t-neuron
           :initform (error ":target required"))
   (weight :accessor weight :initarg :weight :initform 0.1 :type float)
   (last-weight :accessor last-weight :initform 0.0 :type float)
   (learning-rate :accessor learning-rate :initarg :learning-rate
                  :type float :initform 0.02)
   (momentum :accessor momentum :initarg :momentum :type float
             :initform 0.1)
   (delta :accessor delta :initarg :delta :type float :initform 0.0)
   (fire-count :accessor fire-count :type integer :initform 0)
   (update-count :accessor update-count :type integer :initform 0)
   (cx-mutex :accessor cx-mutex)))

(defmethod initialize-instance :after ((cx t-cx) &key)
  (setf (cx-mutex cx) (make-mutex :name (format 
                                       nil
                                       "cx-~a-~a-~6,'0d"
                                       (name (source cx))
                                       (name (target cx))
                                       (id cx)))))

(defclass t-neuron ()
  ((id :reader id :type integer :initform (next-neuron-id))
   (name :accessor name :initarg :name :type string :initform "")
   (input :accessor input :type float :initform 0.0)
   (last-input :accessor last-input :type float :initform 0.0)
   (biased :accessor biased :initarg :biased :type boolean :initform nil)
   (layer :accessor layer :initarg :layer :type integer :initform 0)
   (transfer-key :accessor transfer-key :initarg :transfer-key
                 :initform :logistic)
   (transfer-function :accessor transfer-function :type function)
   (transfer-derivative :accessor transfer-derivative :type function)
   (output :accessor output :type float :initform 0.0)
   (err :accessor err :type float :initform 0.0)
   (err-input :accessor err-input :type float :initform 0.0)
   (last-err-input :accessor last-err-input :type float :initform 0.0)
   (excitation-count :accessor excitation-count :type integer :initform 0)
   (modulation-count :accessor modulation-count :type integer :initform 0)
   (excited :accessor excited :type boolean :initform nil)
   (modulated :accessor modulated :type boolean :initform nil)
   (incoming :accessor incoming :type dlist :initform (make-instance 'dlist))
   (outgoing :accessor outgoing :type dlist :initform (make-instance 'dlist))
   (gate :accessor gate)
   (enabled :accessor enabled :type boolean :initform nil)
   (neuron-thread :accessor neuron-thread :initform nil)
   (ff-count :accessor ff-count :type integer :initform 0)
   (bp-count :accessor bp-count :type integer :initform 0)
   (neuron-mutex :accessor neuron-mutex :type mutex :initform (make-mutex))))

(defmethod initialize-instance :after ((neuron t-neuron) &key)
  (when (zerop (length (name neuron)))
    (setf (name neuron) (make-neuron-name neuron)))
  (let ((transfer (getf *transfer-functions* 
                        (if (biased neuron) :biased (transfer-key neuron)))))
    (setf (transfer-function neuron)
          (getf transfer :function)
          (transfer-derivative neuron)
          (getf transfer :derivative)))
  (setf (gate neuron) 
        ;; A neuron's gate and the neuron share the same name
        (make-gate :name (name neuron) :open nil))
  (setf (neuron-mutex neuron)
        (make-mutex :name (format nil "n-~a" (name neuron)))))

(defmethod ff-count ((neurons list))
  (loop for neuron in neurons collect (ff-count neuron)))

(defmethod make-neuron-name ((neuron t-neuron))
  (format nil "~a~9,'0d" *neuron-name-prefix* (id neuron)))

(defmethod enable ((neuron t-neuron))
  (unless (enabled neuron)
    (when (neuron-thread neuron)
      (when (thread-alive-p (neuron-thread neuron))
        (error "Neuron thread ~a is running while neuron not enabled"
               (thread-name (neuron-thread neuron))))
      (terminate-thread (neuron-thread neuron)))
    (close-gate (gate neuron))
    (setf (enabled neuron) t
          (excited neuron) nil
          (modulated neuron) nil
          (excitation-count neuron) 0
          (modulation-count neuron) 0
          (ff-count neuron) 0
          (bp-count neuron) 0
          (neuron-thread neuron) (make-thread 
                                  (lambda () 
                                    (neuron-loop neuron))
                                  ;; A thread and its neuron share the same name
                                  :name (name neuron)))
    t))

(defmethod enable ((neurons list))
  (loop for neuron in neurons counting (enable neuron)))

(defmethod enable ((neurons dlist))
  (loop for neuron-node = (head neurons) then (next neuron-node)
        while neuron-node
        counting (enable (value neuron-node))))

(defmethod disable ((neuron t-neuron))
  (let ((disabled (when (and (enabled neuron)
                             (neuron-thread neuron))
                    (terminate-thread (neuron-thread neuron))
                    t)))
    (setf (enabled neuron) nil
          (neuron-thread neuron) nil)
    (sleep 0.1)
    disabled))

(defmethod disable ((neurons list))
  (loop for neuron in neurons collect (disable neuron)))

(defmethod transfer ((neuron t-neuron))
  (setf (output neuron) (funcall (transfer-function neuron) (input neuron))
        (last-input neuron) (input neuron)
        (input neuron) 0.0)
  (output neuron))

(defmethod fire-output ((neuron t-neuron))
  (loop for cx-node = (head (outgoing neuron)) then (next cx-node)
        while cx-node
        for cx = (value cx-node)
        for target = (target cx)
        do (excite target (* (output neuron) (weight cx)))
           (incf (fire-count cx))))

(defmethod transfer-error ((neuron t-neuron))
  (setf (err neuron) (* (funcall (transfer-derivative neuron) (output neuron))
                        (err-input neuron))
        (last-err-input neuron) (err-input neuron)
        (err-input neuron) 0.0)
  (err neuron))

(defmethod adjust-weights ((neuron t-neuron))
  (loop for cx-node = (head (outgoing neuron)) then (next cx-node)
        while cx-node
        for cx = (value cx-node)
        do (adjust-weight cx)))

(defmethod adjust-weight ((cx t-cx))
  (multiple-value-bind (new-weight new-delta)
      (compute-new-weight (weight cx)
                          (delta cx)
                          (err (target cx))
                          (output (source cx))
                          (learning-rate cx)
                          (momentum cx))
    (setf (last-weight cx) (weight cx)
          (delta cx) new-delta
          (weight cx) new-weight)
    (incf (update-count cx))))

(defun compute-new-weight (old-weight
                           old-delta
                           target-error
                           source-output
                           learning-rate
                           momentum)
  (let* ((new-delta (+ (* learning-rate target-error source-output)
                       (* momentum old-delta))))
    (values (+ old-weight new-delta) new-delta)))

(defmethod neuron-loop ((neuron t-neuron))
  (loop while (enabled neuron)
        do (with-mutex ((neuron-mutex neuron))
             (when (excited neuron)
               (dlog "neuron ~a excited with input ~f"
                     (name neuron) (input neuron))
               (transfer neuron)
               (fire-output neuron)
               (incf (ff-count neuron))
               (setf (excited neuron) nil
                     (excitation-count neuron) 0)
               (dlog "neuron ~a ff-count=~d" (name neuron) (ff-count neuron)))
             (when (modulated neuron)
               (dlog "neuron ~a modulated with err-input ~f" 
                     (name neuron) (err-input neuron))
               (transfer-error neuron)
               (fire-error neuron)
               (adjust-weights neuron)
               (incf (bp-count neuron))
               (setf (modulated neuron) nil
                     (modulation-count neuron) 0))
             (dlog "neuron ~a closing gate, which was previously ~a"
                   (name neuron)
                   (if (close-gate (gate neuron))
                       "open"
                       "already closed"))
             (dlog "neuron ~a gate is now ~a"
                   (name neuron)
                   (if (gate-open-p (gate neuron)) "open" "closed")))
           (wait-on-gate (gate neuron))))

(defun default-weight ()
  (let ((weight-amplitude 0.3)
        (weight-frequency 0.01))
    (* weight-amplitude
       (sin (* weight-frequency
               (float (next-cx-id-peek)))))))

(defmethod connect ((source t-neuron) (target t-neuron)
                    &key
                      (weight (default-weight))
                      (learning-rate *default-learning-rate*)
                      (momentum *default-momentum*))
  (with-mutex ((neuron-mutex source))
    (with-mutex ((neuron-mutex target))
      (when (and
             (loop
               for cx-node = (head (outgoing source)) then (next cx-node)
               while cx-node
               for cx = (value cx-node)
               for cx-target = (target cx)
               never (= (id target) (id cx-target)))
             (loop
               for cx-node = (head (incoming target)) then (next cx-node)
               while cx-node
               for cx = (value cx-node)
               for cx-source = (source cx)
               never (= (id source) (id cx-source))))
        (let ((cx (make-instance 't-cx
                                 :momentum momentum
                                 :learning-rate learning-rate
                                 :weight weight
                                 :source source
                                 :target target)))
          (push-tail (outgoing source) cx)
          (push-tail (incoming target) cx)
          cx)))))

(defmethod disconnect ((source t-neuron) (target t-neuron))
  (with-mutex ((neuron-mutex source))
    (with-mutex ((neuron-mutex target))
      (loop for cx-node = (head (outgoing source)) then (next cx-node)
            while cx-node
            for cx = (value cx-node)
            for cx-target = (target cx)
            when (= (id target) (id cx-target))
              do (delete-node (outgoing source) cx-node))
      (loop for cx-node = (head (incoming target)) then (next cx-node)
            while cx-node
            for cx = (value cx-node)
            for cx-source = (source cx)
            when (= (id source) (id cx-source))
              do (delete-node (incoming target) cx-node)
                 (return cx)))))

(defmethod isolate ((neuron t-neuron))
  (let ((source (loop
                  for cx-node = (head (incoming neuron)) then (next cx-node)
                  while cx-node
                  for cx = (value cx-node)
                  for source = (source cx)
                  do (disconnect source neuron)
                  counting cx))
        (target (loop
                  for cx-node = (head (outgoing neuron)) then (next cx-node)
                  while cx-node
                  for cx = (value cx-node)
                  for target = (target cx)
                  do (disconnect neuron target)
                  counting cx)))
    (list :incoming source :outgoing target)))

(defmethod isolate ((neurons list))
  (loop for neuron in neurons collect (isolate neuron)))

(defmethod excite ((neuron t-neuron) value)
  (with-mutex ((neuron-mutex neuron))
    (when (enabled neuron)
      (dlog "exciting neuron ~a with value ~f" 
            (name neuron) value)
      (incf (input neuron) value)
      (incf (excitation-count neuron))
      (when (or (zerop (len (incoming neuron)))
                (= (excitation-count neuron) (len (incoming neuron))))
        (setf (excited neuron) t)
        (dlog "neuron ~a opening gate, which was previously ~a"
              (name neuron)
              (if (open-gate (gate neuron)) "closed" "already open"))
        (dlog "neuron ~a gate is now ~a"
              (name neuron)
              (if (gate-open-p (gate neuron)) "open" "closed")))
      t)))

(defmethod excite ((neurons list) (values list))
  (loop for neuron in neurons
        for value in values
        counting (excite neuron value)))

(defmethod excite ((neurons dlist) (values list))
  (loop for neuron-node = (head neurons) then (next neuron-node)
        for value in values
        while neuron-node
        for neuron = (value neuron-node)
        counting (excite neuron value)))

(defmethod fire-error ((neuron t-neuron))
  (loop for cx-node = (head (incoming neuron)) then (next cx-node)
        while cx-node
        for cx = (value cx-node)
        for upstream-neuron = (source cx)
        for weight = (weight cx)
        for err = (err neuron)
        do (modulate upstream-neuron (* err weight))))

(defmethod modulate ((neuron t-neuron) err)
  (with-mutex ((neuron-mutex neuron))
    (when (enabled neuron)
      (incf (err-input neuron) err)
      (incf (modulation-count neuron))
      (when (or (zerop (len (outgoing neuron)))
                (= (modulation-count neuron) (len (outgoing neuron))))
        (setf (modulated neuron) t)
        (open-gate (gate neuron)))
      t)))

(defun list-neuron-threads (name-prefix)
  (remove-if-not
   (lambda (th) (scan (format nil "^~a" name-prefix)
                      (thread-name th)))
   (list-all-threads)))

(defmethod list-incoming ((neuron t-neuron))
  (loop for cx-node = (head (incoming neuron)) then (next cx-node)
        while cx-node
        collect (value cx-node)))

(defmethod list-incoming ((neurons list))
  (loop for neuron in neurons
        append (list-incoming neuron)))

(defmethod list-incoming ((neurons dlist))
  (loop for neuron-node = (head neurons) then (next neuron-node)
        while neuron-node
        append (list-incoming (value neuron-node))))

(defmethod list-incoming-weights ((neuron t-neuron))
  (mapcar (lambda (cx)
            (list :source (name (source cx))
                  :target (name (target cx))
                  :weight (weight cx)
                  :fire-count (fire-count cx)))
          (list-incoming neuron)))

(defmethod list-outgoing ((neuron t-neuron))
  (to-list (outgoing neuron)))

(defmethod list-outgoing ((neurons list))
  (loop for neuron in neurons
        appending (list-outgoing neuron)))

(defmethod list-outgoing ((neurons dlist))
  (loop for neuron-node = (head neurons) then (next neuron-node)
        while neuron-node
        appending (list-outgoing (value neuron-node))))

(defmethod list-incoming-weights ((neurons list))
  (loop for neuron in neurons appending (list-incoming-weights neuron)))

(defmethod list-outputs ((neurons list))
  (loop for neuron in neurons collect (output neuron)))

(defun make-simple-network (name layer-counts)
  (loop with cx-count = (loop with layers = (length layer-counts)
                              for layer-count in layer-counts
                              for next-layer-count in (cdr layer-counts)
                              for index from 0 below layers
                              for cx-count = (* layer-count (1+ next-layer-count))
                                then (+ cx-count (* (1+ layer-count) 
                                                    (1+ next-layer-count)))
                              finally (return (1- cx-count)))
        with weights = (loop with step = (/ pi cx-count)
                             for a from 1 to cx-count
                             for b = 0.0 then (+ b step)
                             collect (sin b) into w
                             finally (return (map 'vector 'identity w)))
        for layer-count in layer-counts
        for layer-index = 0 then (1+ layer-index)
        for is-input-layer = (zerop layer-index)
        for is-output-layer = (= (1+ layer-index) (length layer-counts))
        collect (loop with layer-size = (if (or is-input-layer is-output-layer)
                                            layer-count
                                            (1+ layer-count))
                      for a from 1 to layer-size
                      for transfer-key = (if is-output-layer
                                             :logistic
                                             :relu)
                      for biased = (and (not (or is-input-layer is-output-layer))
                                        (= a (1- layer-size)))
                      for neuron = (make-instance 
                                    't-neuron
                                    :name (format nil "~a-~d-~d" 
                                                  name (1+ layer-index) a)
                                    :transfer-key transfer-key
                                    :biased biased
                                    :layer layer-index)
                      collect neuron)
        into layers
        finally (loop with weight-index = 0
                      for layer in (butlast layers)
                      for next-layer in (cdr layers)
                      do (loop for source in layer
                               do (loop for target in next-layer
                                        for weight = (aref weights weight-index)
                                        do (connect source target
                                                    :weight weight
                                                    :learning-rate weight
                                                    :momentum weight)
                                           (incf weight-index))))
                (return (reduce #'append layers))))

(defmacro with-simple-network ((neurons-var
                                input-layer-var
                                hidden-layers-var
                                output-layer-var
                                name 
                                &rest layer-counts) 
                               &body body)
  `(let* ((counts ',layer-counts)
          (,neurons-var (make-simple-network 
                                   ,name counts))
          (layer-count (length counts))
          (,input-layer-var (remove-if-not 
                                       (lambda (n) (zerop (layer n))) 
                                       ,neurons-var))
          (,hidden-layers-var 
            (remove-if
             (lambda (n) 
               (or (zerop (layer n))
                   (= (layer n) (1- layer-count))))
             ,neurons-var))
          (,output-layer-var 
            (remove-if-not
             (lambda (n) 
               (= (layer n) (1- layer-count)))
             ,neurons-var))
          (sum-of-layer-lengths 
            (+ (length ,input-layer-var)
               (length ,hidden-layers-var)
               (length ,output-layer-var)))
          (biased-neuron-count 
            (max (- (length counts) 2) 0)))
     (assert (= sum-of-layer-lengths 
                (+ (reduce '+ counts) biased-neuron-count)))
     (unwind-protect
          (progn ,@body)
       (disable ,neurons-var))))

(defmacro with-neurons ((neurons-var neuron-count name)
                        &body body)
  `(let ((,neurons-var
           (loop for a from 1 to ,neuron-count
                 collect
                 (make-instance 't-neuron 
                                :name (format nil "~a-~9,'0d" ,name a)))))
     (unwind-protect
          (progn ,@body)
       (disable ,neurons-var))))

(defmethod wait-for-output ((neuron t-neuron) 
                            (target-ff-count integer) 
                            (timeout-seconds float))
  (loop with start-time = (mark-time)
        while (< (ff-count neuron) target-ff-count)
        when (> (elapsed-time start-time) timeout-seconds)
          do (error "Timed out after ~f seconds" timeout-seconds)
        finally (return (= (ff-count neuron) target-ff-count))))

(defmethod wait-for-output ((neurons list)
                            (target-ff-count integer)
                            (timeout-seconds float))
  (loop with start-time = (mark-time)
        while (some (lambda (neuron)
                      (< (ff-count neuron) target-ff-count))
                    neurons)
        when (> (elapsed-time start-time) timeout-seconds)
          do (error "Timed out after ~f seconds" timeout-seconds)
        finally (return (every (lambda (neuron)
                                 (= (ff-count neuron) target-ff-count))
                               neurons))))

(defmethod wait-for-backprop ((neuron t-neuron)
                              (target-bp-count integer)
                              (timeout-seconds float))
  (loop with start-time = (mark-time)
        while (< (bp-count neuron) target-bp-count)
        when (> (elapsed-time start-time) timeout-seconds)
          do (error "Timed out after ~f seconds" timeout-seconds)
        finally (return (= (bp-count neuron) target-bp-count))))

(defmethod wait-for-backprop ((neurons list)
                              (target-bp-count integer)
                              (timeout-seconds float))
  (loop with start-time = (mark-time)
        while (some (lambda (neuron)
                      (< (bp-count neuron) target-bp-count))
                    neurons)
        when (> (elapsed-time start-time) timeout-seconds)
          do (error "Timed out after ~f seconds" timeout-seconds)
        finally (return (every (lambda (neuron)
                                 (= (bp-count neuron) target-bp-count))
                               neurons))))


(defmethod wait-for-output-p ((neuron t-neuron) 
                              (target-ff-count integer) 
                              (timeout-seconds float))
  (loop with start-time = (mark-time)
        while (< (ff-count neuron) target-ff-count)
        when (> (elapsed-time start-time) timeout-seconds)
          do (return nil)
        finally (return t)))

(defmethod wait-for-output-p ((neurons list)
                              (target-ff-count integer)
                              (timeout-seconds float))
  (loop with start-time = (mark-time)
        while (some (lambda (neuron)
                      (< (ff-count neuron) target-ff-count))
                    neurons)
        when (> (elapsed-time start-time) timeout-seconds)
          do (return nil)
        finally (return t)))

(defmethod wait-for-backprop-p ((neuron t-neuron)
                                (target-bp-count integer)
                                (timeout-seconds float))
  (loop with start-time = (mark-time)
        while (< (bp-count neuron) target-bp-count)
        when (> (elapsed-time start-time) timeout-seconds)
          do (return nil)
        finally (return t)))

(defmethod wait-for-backprop-p ((neurons list)
                                (target-bp-count integer)
                                (timeout-seconds float))
  (loop with start-time = (mark-time)
        while (some (lambda (neuron)
                      (< (bp-count neuron) target-bp-count))
                    neurons)
        when (> (elapsed-time start-time) timeout-seconds)
          do (return nil)
        finally (return t)))

(defmethod train-frame ((input-layer list) (output-layer list) (frame list))
  (let* ((inputs (first frame))
         (expected-outputs (second frame))
         (ff-count (ff-count (first input-layer)))
         (bp-count (bp-count (first input-layer)))
         (excited (or (loop for neuron in input-layer
                            for input in inputs
                            always (excite neuron input))
                      (error "excite failed")))
         (ff-wait (or (wait-for-output-p output-layer (1+ ff-count) 0.1)
                      (error "wait-for-output-p failed")))
         (outputs (mapcar #'output output-layer))
         (errors (mapcar (lambda (e o) (- e o)) expected-outputs outputs))
         (when *testing*
           (assert (every (lambda (n) (zerop (err-input n))) output-layer)))
         (modulated (or (loop for neuron in output-layer
                              for error in errors
                              always (modulate neuron error))
                        (error "modulate failed")))
         (bp-wait (or (wait-for-backprop-p input-layer (1+ bp-count) 0.1)
                      (error "wait-for-backprop-p failed"))))
    errors))

(defmethod feed-forward ((input-layer list) (output-layer list) (inputs list))
  (let* ((ff-count (ff-count (first input-layer)))
         (excited (loop for neuron in input-layer
                        for input in inputs
                        always (excite neuron input)))
         (ff-wait (when excited
                    (wait-for-output-p output-layer (1+ ff-count) 0.1))))
    (when ff-wait (mapcar #'output output-layer))))

(defmethod apply-inputs ((input-layer list) (inputs list))
  (loop for neuron in input-layer
        for value in inputs
        do (setf (input neuron) value)))

(defmethod apply-error ((output-layer list) (expected-outputs list))
  (loop for neuron in output-layer
        for expected in expected-outputs
        for actual = (output neuron)
        for error = (- expected actual)
        do (setf (err-input neuron) error)
        collect error))

(defmethod output-layer-error ((output-layer list) (expected-output list))
  (sqrt 
   (reduce 
    '+ 
    (mapcar 
     (lambda (n e)
       (expt (- e (output n)) 2))
     output-layer
     expected-output))))

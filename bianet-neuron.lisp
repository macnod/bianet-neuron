(in-package :bianet-neuron)

(defparameter *log* nil)

(defparameter *next-neuron-id* 0)
(defparameter *neuron-id-mutex* (make-mutex :name "neuron-id"))

(defparameter *next-cx-id* 0)
(defparameter *cx-id-mutex* (make-mutex :name "cx-id"))

(defparameter *default-learning-rate* 0.02)
(defparameter *default-momentum* 0.1)
(defparameter *default-min-weight* -0.9)
(defparameter *default-max-weight* 0.9)


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
  (declare (single-float min max))
  (lambda (&key rstate
             global-index
             global-fraction
             layer-fraction
             neuron-fraction)
    (declare (ignore rstate global-index global-fraction neuron-fraction))
    (+ min (* layer-fraction (- max min)))))

(defun make-sinusoid-weight-fn (&key (min -0.5) (max 0.5))
  (declare (single-float min max))
  (lambda (&key rstate
             global-index
             global-fraction
             layer-fraction
             neuron-fraction)
    (declare (ignore rstate global-fraction layer-fraction neuron-fraction))
    (+ (* (/ (+ (sin (coerce global-index 'single-float)) 1) 2) (- max min)) min)))

(defun logistic (x)
  (declare (single-float x)
           (optimize (speed 3) (safety 0)))
  (cond ((> x (the single-float 16.64)) (the single-float 1.0))
        ((< x (the single-float -88.7)) (the single-float 0.0))
        ((< (the single-float (abs x)) (the single-float 1e-8)) (the single-float 0.5))
        (t (/ (the single-float 1.0) (the single-float (1+ (the single-float (exp (- x)))))))))

(defun logistic-slow (x)
  (cond ((> x 16.64) 1.0)
        ((< x -88.7) 0.0)
        ((< (abs x) 1e-8) 0.5)
        (t (/ 1 (1+ (exp (- x)))))))

(defun logistic-derivative (x)
  (declare (single-float x)
           (optimize (speed 3) (safety 0)))
  (* x (- (the single-float 1.0) x)))

(defun relu (x)
  (declare (single-float x)
           (optimize (speed 3) (safety 0)))
  (the single-float (max (the single-float 0.0) x)))

(defun relu-derivative (x)
  (declare (single-float x)
           (optimize (speed 3) (safety 0)))
  (if (<= x (the single-float 0.0))
      (the single-float 0.0)
      (the single-float 1.0)))

(defun relu-leaky (x)
  (declare (single-float x)
           (optimize (speed 3) (safety 0)))
  (the single-float (max (the single-float 0.0) x)))

(defun relu-leaky-derivative (x)
  (if (<= x (the single-float 0.0))
      (the single-float 0.001)
      (the single-float 1.0)))

(defparameter *transfer-functions*
  (list :logistic (list :function #'logistic
                        :derivative #'logistic-derivative)
        :relu (list :function #'relu
                    :derivative #'relu-derivative)
        :relu-leaky (list :function #'relu-leaky
                          :derivative #'relu-leaky-derivative)))
(defclass t-cx ()
  ((id :reader id :type integer :initform (next-cx-id))
   (source :reader source :initarg :source :type t-neuron
          :initform (error ":source required"))
   (target :reader target :initarg :target :type t-neuron
           :initform (error ":target required"))
   (weight :accessor weight :initarg :weight :initform 0.1 :type single-float)
   (weight-history :accessor weight-history :type dlist
                   :initform (make-instance 'dlist))
   (learning-rate :accessor learning-rate :initarg :learning-rate
                  :type single-float :initform 0.02)
   (momentum :accessor momentum :initarg :momentum :type single-float
             :initform 0.1)
   (delta :accessor delta :initarg :delta :type single-float :initform 0.0)
   (fire-count :accessor fire-count :type integer :initform 0)
   (update-count :accessor update-count :type integer :initform 0)
   (cx-mtx :reader cx-mtx :initform (make-mutex))))

(defclass t-neuron ()
  ((id :reader id :type integer :initform (next-neuron-id))
   (name :accessor name :initarg :name :type string :initform "")
   (input :accessor input :type single-float :initform 0.0)
   (biased :accessor biased :initarg :biased :type boolean :initform nil)
   (transfer-key :accessor transfer-key :initarg :transfer-key
                 :initform :logistic)
   (transfer-function :accessor transfer-function :type function)
   (transfer-derivative :accessor transfer-derivative :type function)
   (output :accessor output :type single-float :initform 0.0)
   (expected-output :accessor expected-output :type single-float :initform 0.0)
   (err :accessor err :type single-float :initform 0.0)
   (err-derivative :accessor err-derivative :type single-float :initform 0.0)
   (x-coor :accessor x-coor :type single-float :initarg :x :initform 0.0)
   (y-coor :accessor y-coor :type single-float :initarg :y :initform 0.0)
   (z-coor :accessor z-coor :type single-float :initarg :z :initform 0.0)
   (received :accessor received :type integer :initform 0)
   (incoming :accessor incoming :type dlist :initform (make-instance 'dlist))
   (outgoing :accessor outgoing :type dlist :initform (make-instance 'dlist))
   (transfer-gate :accessor transfer-gate)
   (enabled :accessor enabled :type boolean :initform nil)
   (activation-thread :accessor activation-thread :initform nil)
   (activation-count :accessor activation-count :type integer :initform 0)
   (transfer-count :accessor transfer-count :type integer :initform 0)
   (input-mtx :reader input-mtx :initform (make-mutex))
   (output-mtx :reader output-mtx :initform (make-mutex))
   (err-mtx :reader err-mtx :initform (make-mutex))
   (err-der-mtx :reader err-der-mtx :initform (make-mutex))))

(defmethod initialize-instance :after ((neuron t-neuron) &key)
  (when (zerop (length (name neuron)))
    (setf (name neuron) (format nil "n-~:d" (id neuron))))
  (when (biased neuron)
    (setf (input neuron) 1.0))
  (let ((transfer (getf *transfer-functions* (transfer-key neuron))))
    (setf (transfer-function neuron)
          (getf transfer :function)
          (transfer-derivative neuron)
          (getf transfer :derivative)))
  (let ((gate-name (format nil "ng-~:d" (id neuron))))
    (setf (transfer-gate neuron) (make-gate :name gate-name :open nil))))

(defmethod enable ((neuron t-neuron))
  (when (and (not (enabled neuron))
             (or (not (activation-thread neuron))
                 (not (thread-alive-p (activation-thread neuron)))))
    (let ((name (format nil "nt-~:d" (id neuron))))
      (setf (enabled neuron) t)
      (close-gate (transfer-gate neuron))
      (setf (received neuron) 0)
      (setf (activation-thread neuron)
            (make-thread (lambda () (activate neuron)) :name name))
      name)))

(defmethod enable ((neurons list))
  (loop for neuron in neurons collect (enable neuron)))


(defmethod enable ((neurons dlist))
  (loop for neuron-node = (head neurons) then (next neuron-node)
        while neuron-node
        collect (enable (value neuron-node))))

(defmethod disable ((neuron t-neuron))
  (when (and (enabled neuron)
             (and (activation-thread neuron)
                  (thread-alive-p (activation-thread neuron))))
    (let ((name (thread-name (activation-thread neuron))))
      (setf (enabled neuron) nil)
      (open-gate (transfer-gate neuron))
      (join-thread (activation-thread neuron))
      (setf (activation-thread neuron) nil)
      name)))

(defmethod disable ((neurons list))
  (loop for neuron in neurons collect (disable neuron)))

(defmethod activate ((neuron t-neuron))
  (loop
    while (enabled neuron)
    do 
       (wait-on-gate (transfer-gate neuron))
       (when (enabled neuron)
         (transfer neuron)
         (loop 
           for cx-node = (head (outgoing neuron)) then (next cx-node)
           while cx-node
           for cx = (value cx-node)
           for value = (* (output neuron) (weight cx))
           for log = (dlog "~a: sending ~,4f to ~a" 
                           (id neuron) value (id (target cx)))
           do (excite (target cx) value)
              (incf (fire-count cx))))
       (dlog "~a: closing transfer-gate" (id neuron))
       (close-gate (transfer-gate neuron))
       (dlog "~a: transfer-gate closed" (id neuron))))

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
  (with-mutex ((input-mtx source))
    (with-mutex ((input-mtx target))
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
  (with-mutex ((input-mtx source))
    (with-mutex ((input-mtx target))
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
  (with-mutex ((input-mtx neuron))
    (when (enabled neuron)
      (incf (input neuron) value)
      (incf (received neuron))
      (when (or (zerop (len (incoming neuron)))
                (= (received neuron) (len (incoming neuron))))
        (dlog "~a: opening transfer-gate" (id neuron))
        (open-gate (transfer-gate neuron))
        (dlog "~a: transfer-gate is open" (id neuron))
        t))))

(defmethod excite ((neurons list) (values list))
  (loop for neuron in neurons 
        for value in values
        counting (excite neuron value)))

(defmethod transfer ((neuron t-neuron))
  (setf (output neuron) (funcall (transfer-function neuron) (input neuron)))
  (incf (transfer-count neuron))
  (unless (biased neuron)
    (with-mutex ((input-mtx neuron))
        (setf (input neuron) 0.0
              (received neuron) 0)))
    (with-mutex ((err-mtx neuron))
      (setf (err neuron) 0.0)))

(defun list-neuron-threads ()
  (remove-if-not
   (lambda (th) (scan "^nt-" (thread-name th)))
   (list-all-threads)))

(defmethod list-cxs ((neuron t-neuron))
  (loop for cx-node = (head (outgoing neuron)) then (next cx-node)
        while cx-node
        collect (value cx-node)))

(defmethod list-cxs ((neurons list))
  (loop for neuron in neurons
        append (list-cxs neuron)))

(defmethod list-cxs ((neurons dlist))
  (loop for neuron-node = (head neurons) then (next neuron-node)
        while neuron-node
        append (list-cxs (value neuron-node))))

(defmethod list-weights ((neuron t-neuron))
  (mapcar (lambda (cx)
            (list :source (name (source cx))
                  :target (name (target cx))
                  :weight (weight cx)
                  :fire-count (fire-count cx)))
          (list-cxs neuron)))

(defmethod list-weights ((neurons list))
  (loop for neuron in neurons appending (list-weights neuron)))

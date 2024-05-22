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

(defparameter *neuron-name-prefix* "n-")
(defparameter *neuron-thread-name-prefix* "nt-")
(defparameter *neuron-gate-name-prefix* "ng-")

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
        ((< (the single-float (abs x)) (the single-float 1e-8))
         (the single-float 0.5))
        (t (/ (the single-float 1.0) 
              (the single-float (1+ (the single-float (exp (- x)))))))))

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
                          :derivative #'relu-leaky-derivative)
        :biased (list :function (lambda (x) (declare (ignore x)) (relu 1.0))
                      :derivative #'relu-derivative)))

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
   (err :accessor err :type single-float :initform 0.0)
   (err-input :accessor err-input :type single-float :initform 0.0)
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
   (input-mtx :reader input-mtx :initform (make-mutex))
   (err-input-mtx :reader err-input-mtx :initform (make-mutex))))

(defmethod make-neuron-name ((neuron t-neuron))
  (format nil "~a~:d" *neuron-name-prefix* (id neuron)))

(defmethod make-neuron-thread-name ((neuron t-neuron))
  (format nil "~a~:d" *neuron-thread-name-prefix* (id neuron)))

(defmethod make-neuron-gate-name ((neuron t-neuron))
  (format nil "~a~:d" *neuron-gate-name-prefix* (id neuron)))

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
        (make-gate :name (make-neuron-gate-name neuron) :open nil)))

(defmethod enable ((neuron t-neuron))
  (when (and (not (enabled neuron)))
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
                                    :name (make-neuron-thread-name neuron)))
    t))

(defmethod enable ((neurons list))
  (loop for neuron in neurons counting (enable neuron)))

(defmethod enable ((neurons dlist))
  (loop for neuron-node = (head neurons) then (next neuron-node)
        while neuron-node
        counting (enable (value neuron-node))))

(defmethod disable ((neuron t-neuron))
  (when (enabled neuron)
    (when (or (not (neuron-thread neuron))
              (and (neuron-thread neuron)
                   (not (thread-alive-p (neuron-thread neuron)))))
      (error "Neuron thread ~ais not running while neuron enabled"
             (if (neuron-thread neuron)
                 ""
                 (format nil "~a " 
                         (thread-name (neuron-thread neuron))))))
    (let ((name (thread-name (neuron-thread neuron))))
      (join-thread (neuron-thread neuron))
      (setf (neuron-thread neuron) nil
            (enabled neuron) nil)
      t)))

(defmethod disable ((neurons list))
  (loop for neuron in neurons collect (disable neuron)))

(defmethod transfer ((neuron t-neuron))
  (with-mutex ((input-mtx neuron))
    (setf (output neuron) (funcall (transfer-function neuron) (input neuron))
          (input neuron) 0.0
          (excited neuron) nil
          (excitation-count neuron) 0)))

(defmethod fire-output ((neuron t-neuron))
  (loop for cx-node = (head (outgoing neuron)) then (next cx-node)
        while cx-node
        for cx = (value cx-node)
        for target = (target cx)
        do (excite target (* (output neuron) (weight cx)))))

(defmethod transfer-error ((neuron t-neuron))
  (with-mutex ((err-input-mtx neuron))
    (setf (err neuron) (funcall (transfer-derivative neuron) (err-input neuron))
          (err-input neuron) 0.0
          (modulated neuron) nil
          (modulation-count neuron) 0)))

(defmethod neuron-loop ((neuron t-neuron))
  (loop while t
        do (when (excited neuron)
             (transfer neuron)
             (fire-output neuron)
             (incf (ff-count neuron)))
           (when (modulated neuron)
             (transfer-error neuron)
             (fire-error neuron)
             (incf (bp-count neuron)))
           (close-gate (gate neuron))
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
      (incf (excitation-count neuron))
      (when (or (zerop (len (incoming neuron)))
                (= (excitation-count neuron) (len (incoming neuron))))
        (setf (excited neuron) t)
        (open-gate (gate neuron))))))

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
        do (modulate upstream-neuron (* (err neuron) (weight cx)))))

(defmethod modulate ((neuron t-neuron) err)
  (with-mutex ((err-input-mutex neuron))
    (when (enabled neuron)
      (incf (err-input neuron) err)
      (incf (modulation-count neuron))
      (when (or (zerop (len (outgoing neuron)))
                (= (modulation-count neuron) (len (outgoing neuron))))
        (setf (modulated neuron) t)
        (open-gate (gate neuron))))))

(defun list-neuron-threads ()
  (remove-if-not
   (lambda (th) (scan (format nil "^~a" *neuron-thread-name-prefix*)
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

(defmethod list-incoming-weights ((neurons list))
  (loop for neuron in neurons appending (list-incoming-weights neuron)))

(defmethod list-outputs ((neurons list))
  (loop for neuron in neurons collect (output neuron)))

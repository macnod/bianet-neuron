(in-package :bianet-neuron)

(defparameter *log* nil)

(defparameter *next-neuron-id* 0)
(defparameter *neuron-id-mutex* (make-mutex :name "neuron-id"))

(defparameter *next-cx-id* 0)
(defparameter *cx-id-mutex* (make-mutex :name "cx-id"))

(defparameter *log-mutex* (make-mutex :name "log"))
(defparameter *log-index* 0)

(defparameter *default-learning-rate* 0.1)
(defparameter *default-momentum* 0.3)

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

(defun logistic (x)
  (/ 1.0 (1+ (exp (- x)))))

(defun logistic-derivative (x)
  (* x (- 1.0 x)))

(defun relu (x)
  (max 0.0 x))

(defun relu-derivative (x)
  (if (> x 0) 1.0 0.0))

(defparameter *transfer-functions*
  (list :logistic (list :function #'logistic
                        :derivative #'logistic-derivative)
        :relu (list :function #'relu
                    :derivative #'relu-derivative)
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
   (last-output :accessor last-output :type float :initform 0.0)
   (err :accessor err :type float :initform 0.0)
   (err-input :accessor err-input :type float :initform 0.0)
   (last-err-input :accessor last-err-input :type float :initform 0.0)
   (excitation-count :accessor excitation-count :type integer :initform 0)
   (modulation-count :accessor modulation-count :type integer :initform 0)
   (excited :accessor excited :type boolean :initform nil)
   (modulated :accessor modulated :type boolean :initform nil)
   (incoming :accessor incoming :type dlist :initform (make-instance 'dlist))
   (outgoing :accessor outgoing :type dlist :initform (make-instance 'dlist))
   (enabled :accessor enabled :type boolean :initform nil)
   (neuron-thread :accessor neuron-thread :initform nil)
   (ff-count :accessor ff-count :type integer :initform 0)
   (bp-count :accessor bp-count :type integer :initform 0)
   (i-mailbox :accessor i-mailbox :type mailbox :initform (make-mailbox))
   (e-mailbox :accessor e-mailbox :type mailbox :initform (make-mailbox))))

(defmethod initialize-instance :after ((neuron t-neuron) &key)
  (when (zerop (length (name neuron)))
    (setf (name neuron) (make-neuron-name neuron)))
  (let ((transfer (getf *transfer-functions* 
                        (if (biased neuron) :biased (transfer-key neuron)))))
    (setf (transfer-function neuron)
          (getf transfer :function)
          (transfer-derivative neuron)
          (getf transfer :derivative))))

(defmethod make-neuron-name ((neuron t-neuron))
  (format nil "~a~9,'0d" *neuron-name-prefix* (id neuron)))

(defmethod enable ((neuron t-neuron))
  (unless (enabled neuron)
    (when (neuron-thread neuron)
      (when (thread-alive-p (neuron-thread neuron))
        (error "Neuron thread ~a is running while neuron not enabled"
               (thread-name (neuron-thread neuron))))
      (terminate-thread (neuron-thread neuron)))
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

(defmethod disable ((neuron t-neuron))
  (let ((disabled (when (and (enabled neuron)
                             (neuron-thread neuron))
                    (terminate-thread (neuron-thread neuron))
                    t)))
    (setf (enabled neuron) nil
          (neuron-thread neuron) nil)
    disabled))

(defmethod transfer ((neuron t-neuron))
  (let ((new-output (funcall (transfer-function neuron) (input neuron))))
    (setf (last-output neuron) (output neuron)
          (output neuron) new-output
          (last-input neuron) (input neuron)
          (input neuron) 0.0)
    (dlog "Thread ~a in neuron ~a transferring input (~,3f) to output (~,3f)"
          (when (neuron-thread neuron)
            (thread-name (neuron-thread neuron)))
          (name neuron)
          (last-input neuron)
          (output neuron))
    (output neuron)))

(defmethod fire-output ((neuron t-neuron))
  (dlog "Thread ~a firing output of neuron ~a (~,3f)"
        (when (neuron-thread neuron)
          (thread-name (neuron-thread neuron)))
        (name neuron)
        (output neuron))
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
        do (evaluate-input-messages neuron)
        when (excited neuron) do
          (transfer neuron)
          (fire-output neuron)
          (incf (ff-count neuron))
          (setf (excited neuron) nil
                (excitation-count neuron) 0)
        do (evaluate-error-messages neuron)
        when (modulated neuron) do
          (transfer-error neuron)
          (fire-error neuron)
          (adjust-weights neuron)
          (incf (bp-count neuron))
          (setf (modulated neuron) nil
                (modulation-count neuron) 0)))

(defmethod connect ((source t-neuron) 
                    (target t-neuron) 
                    &key
                      (weight (error "weight is required"))
                      (learning-rate *default-learning-rate*)
                      (momentum *default-momentum*))
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
      cx)))

(defmethod disconnect ((source t-neuron) (target t-neuron))
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
             (return cx)))

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

(defmethod evaluate-input-messages ((neuron t-neuron))
  (loop until (mailbox-empty-p (i-mailbox neuron))
        for value = (receive-message (i-mailbox neuron))
        for log = (dlog "Thread ~a in neuron ~a received value ~,3f"
                        (when (neuron-thread neuron)
                          (thread-name (neuron-thread neuron)))
                        (name neuron)
                        value)
        do (excite-internal neuron value)))

(defmethod evaluate-error-messages ((neuron t-neuron))
  (loop until (mailbox-empty-p (e-mailbox neuron))
        for err = (receive-message (e-mailbox neuron))
        do (modulate-internal neuron err)))

(defmethod excite-internal ((neuron t-neuron) value)
  (dlog "Thread ~a in neuron ~a exciting with ~,3f"
        (when (neuron-thread neuron)
          (thread-name (neuron-thread neuron)))
        (name neuron)
        value)
  (incf (input neuron) value)
  (incf (excitation-count neuron))
  (when (or (zerop (len (incoming neuron)))
            (= (excitation-count neuron) (len (incoming neuron))))
    (dlog "Setting neuron ~a to excited" (name neuron))
    (setf (excited neuron) t)))

(defmethod excite ((neuron t-neuron) value)
  (send-message (i-mailbox neuron) value)
  value)

(defmethod fire-error ((neuron t-neuron))
  (loop for cx-node = (head (incoming neuron)) then (next cx-node)
        while cx-node
        for cx = (value cx-node)
        for upstream-neuron = (source cx)
        for weight = (weight cx)
        for err = (err neuron)
        do (modulate upstream-neuron (* err weight))))

(defmethod modulate-internal ((neuron t-neuron) err)
  (incf (err-input neuron) err)
  (incf (modulation-count neuron))
  (when (or (zerop (len (outgoing neuron)))
            (= (modulation-count neuron) (len (outgoing neuron))))
    (setf (modulated neuron) t)))

(defmethod modulate ((neuron t-neuron) err)
  (send-message (e-mailbox neuron) err)
  err)

(defun list-neuron-threads (name-prefix)
  (remove-if-not
   (lambda (th) (scan (format nil "^~a" name-prefix)
                      (thread-name th)))
   (list-all-threads)))

(defmethod list-incoming ((neuron t-neuron))
  (loop for cx-node = (head (incoming neuron)) then (next cx-node)
        while cx-node
        collect (value cx-node)))

(defmethod list-incoming-weights ((neuron t-neuron))
  (mapcar (lambda (cx)
            (list :source (name (source cx))
                  :target (name (target cx))
                  :weight (weight cx)
                  :fire-count (fire-count cx)))
          (list-incoming neuron)))

(defmethod list-outgoing ((neuron t-neuron))
  (to-list (outgoing neuron)))

(defmethod list-incoming-weights ((neurons list))
  (loop for neuron in neurons appending (list-incoming-weights neuron)))

(defmethod wait-for-output ((neuron t-neuron) 
                            (target-ff-count integer)
                            (timeout-seconds float))
  (loop with start-time = (mark-time)
        while (< (ff-count neuron) target-ff-count)
        when (> (elapsed-time start-time) timeout-seconds)
          do (error "Timed out after ~f seconds" timeout-seconds)
        finally (return (= (ff-count neuron) target-ff-count))))

(defmethod wait-for-backprop ((neuron t-neuron)
                              (target-bp-count integer)
                              (timeout-seconds float))
  (loop with start-time = (mark-time)
        while (< (bp-count neuron) target-bp-count)
        when (> (elapsed-time start-time) timeout-seconds)
          do (error "Timed out after ~f seconds" timeout-seconds)
        finally (return (= (bp-count neuron) target-bp-count))))

(defmethod wait-for-output-p ((neuron t-neuron) 
                              (target-ff-count integer) 
                              (timeout-seconds float))
  (loop with start-time = (mark-time)
        while (< (ff-count neuron) target-ff-count)
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

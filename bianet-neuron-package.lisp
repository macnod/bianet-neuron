(defpackage :bianet-neuron
  (:use :cl :dc-dlist :dc-eclectic :sb-concurrency :sb-thread :cl-ppcre)
  (:export
   activation-count
   biased
   connect
   delta
   disable
   disconnect
   enable
   enabled
   err
   err-derivative
   excite
   expected-output
   fire-count
   id
   incoming
   input
   isolate
   learning-rate
   list-cxs
   list-neuron-threads
   list-weights
   momentum
   name
   outgoing
   output
   received
   source
   t-neuron
   target
   transfer-count
   update-count
   weight
   weight-history
   x-coor
   y-coor
   z-coor
   ))

   

(asdf:defsystem :bianet-neuron
  :description "A bianet neuron, used in bianet neural networks."
  :author "Donnie Cameron <macnod@gmail.com>"
  :license "MIT License"
  :depends-on (:sb-concurrency
               :dc-dlist
               :dc-eclectic)
  :serial t
  :components ((:file "bianet-neuron-package")
               (:file "bianet-neuron")))

# BlazefaceTF

Unofficial Tensorflow implementation of [BlazeFace](https://sites.google.com/view/perception-cv4arvr/blazeface)
 
 ```python
import numpy
from blazeface import BlazeFace

model = BlazeFace()

x = np.random.uniform(0, 255, (1,128,128,3))
predictions = model(x)
```

##  Dependiencies

* Tensorflow (2.0.0)
* Keras
* Numpy

## Paper
### BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs
[[Project Page]](https://sites.google.com/view/perception-cv4arvr/blazeface)
[[Original Implementation]](https://github.com/google/mediapipe/tree/master/mediapipe/models#blazeface-face-detection-model)

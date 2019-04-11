# Fast Slic

Fast-slic is a SLIC-variant algorithm implementation that aims for significantly low runtime with cpu. It runs 6-8 times faster than existing SLIC implementations, at the cost of accuracy to some extent. 

## Installation
```python
pip install fast_slic
```

## Basic Usage
```python
import numpy as np

from fast_slic import Slic
from PIL import Image

with Image.open("fish.jpg") as f:
   image = np.array(f)
# import cv2; image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)   # You can convert the image to CIELAB space if you need.
slic = Slic(num_components=200, compactness_shift=6)
assignment = slic.iterate(image) # Cluster Map
print(assignment)
print(slic.slic_model.clusters) # The cluster information of superpixels.
```

## Performance

With max iteration set to 10, run times of slic implementations for 640x480 image are as follows:

| Implementation                                  | Run time(ms)   |
| -----------------------------------------       | --------------:|
| skimage.segment.slic                            | 216ms          |
| cv2.ximgproc.createSuperpixelSLIC.iterate       | 142ms          |
| fast_slic(single core build)                    | 58ms           |
| fast_slic(w/ OpenMP supports, default in GCC)   | **36ms**       |


 
(RGB-to-CIELAB conversion time is not included. Tested with Ryzen 2600x 4.0Hz O.C.)

# TODO

 - [ ] Reduce coarseness
 - [ ] Include simple CRF utilities

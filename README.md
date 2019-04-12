# Fast Slic

Fast-slic is a SLIC-variant algorithm implementation that aims for significantly low runtime with cpu. It runs 7-11 times faster than existing SLIC implementations.

It started as a part of my hobby project that demanded true "real time" capability in video stream processing. Among pipelines of it was a postprocessing pipeline smoothing the result of image with SLIC superpixels and CRF. Unfortunately, there were no satisfying library for real-time(>30fps) goal. [gSLICr](https://github.com/carlren/gSLICr) was the most promising candidate, but I couldn't make use of it due to limited hardware and inflexible license of CUDA. Therefore, I made the lightweight variant of SLIC, sacrificing a little of accuracy, to gain super-fast implementation.

## Demo
<table>
   <tr>
      <td><img alt="demo_clownfish" src="https://user-images.githubusercontent.com/2352985/55978839-c5504780-5ccb-11e9-9820-d8ddf950f230.png"></td>
      <td><img alt="demo_tiger" src="https://user-images.githubusercontent.com/2352985/55949421-86030600-5c8d-11e9-9693-b05f00f1c792.jpg"></td>
   </tr>
</table>

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
| fast_slic(single core build)                    | 43ms           |
| **fast_slic(w/ OpenMP supports, default in GCC)**   | **16ms**       |

 
(RGB-to-CIELAB conversion time is not included. Tested with Ryzen 2600x 6C12T 4.0Hz O.C.)

## Known Issues
 * If you give too large value of `compactness_shift`, score variables overflow and you get an artistic painting of diamond shaped boxes rather than superpixels you want.

## TODO
 - [ ] Remove or merge small blobs
 - [ ] Include simple CRF utilities
 - [ ] More scalable parallel loop in cluster assignment. I suspect there is false sharing problem in the loop.
 - [ ] would be great if I can optimize loop more. SIMD?

# Fast Slic

Fast-slic is a SLIC-variant algorithm implementation that aims for significantly low runtime with cpu. It runs 7-20 times faster than existing SLIC implementations. Fast-slic can process 1280x720 image stream at 60fps.

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
slic = Slic(num_components=200, compactness=6)
assignment = slic.iterate(image) # Cluster Map
print(assignment)
print(slic.slic_model.clusters) # The cluster information of superpixels.
```

If your machine has AVX2 instruction set, you can make it three times faster using `fast_slic.avx2.SlicAvx2` class instead of `fast_slic.Slic`. Haswell and newer Intel cpus, Excavator, and Ryzen support this.

```python
import numpy as np

# Much faster than the standard class
from fast_slic.avx2 import SlicAvx2
from PIL import Image

with Image.open("fish.jpg") as f:
   image = np.array(f)
# import cv2; image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)   # You can convert the image to CIELAB space if you need.
slic = SlicAvx2(num_components=200, compactness=6)
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
| fast_slic.Slic(single core build)               | 43ms           |
| fast_slic.avx2.SlicAvx2(single core build /w avx2 support)      | 17ms           |
| **fast_slic.Slic(w/ OpenMP support)**           | **16ms**       |
| **fast_slic.avx2.SlicAvx2(w/ OpenMP, avx2 support)**   | **8ms**       |

 
(RGB-to-CIELAB conversion time is not included. Tested with Ryzen 2600x 6C12T 4.0Hz O.C.)

## Known Issues
 * `SlicAvx2` is kind of clumsy at side and corner processing. This is intended behavior to remove overhead of conditional branches required for boundary check. If this is problem to you, put paddings of size of `ceil(sqrt(image_height * image_width / num_components)` around an image.
 * `compactness` is allocated as an 8-byte integer internally, so its value cannot exceed 255. You will never need more than 255 compactness in practice.
 * Windows build is quite slower compared to those of linux and mac. Maybe it is due to openmp overhead?

 
## Experimental
 * To push the limit, compile it with `FAST_SLIC_AVX2_FASTER` flag and get more performance gain. (though performance margin was small in my pc)
## TODO
 - [ ] Remove or merge small blobs
 - [ ] Include simple CRF utilities
 - [ ] Add tests
 - [x] Windows build
 - [x] More scalable parallel loop in cluster assignment. I suspect there is false sharing problem in the loop.
 - [x] would be great if I can optimize loop more. SIMD?

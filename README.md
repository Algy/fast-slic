# Fast Slic

Fast-slic is a SLIC-variant algorithm implementation that aims for significantly low runtime with cpu. It runs 7-20 times faster than existing SLIC implementations. Fast-slic can process 1280x720 image stream at 60fps.

It started as a part of my hobby project that demanded true "real time" capability in video stream processing. Among pipelines of it was a postprocessing pipeline smoothing the result of image with SLIC superpixels and CRF. Unfortunately, there were no satisfying library for real-time(>30fps) goal. [gSLICr](https://github.com/carlren/gSLICr) was the most promising candidate, but I couldn't make use of it due to limited hardware and inflexible license of CUDA. Therefore, I made the blazingly fast variant of SLIC using only CPU.

[Paper preprint](https://github.com/Algy/fast-slic/files/4009304/fastslic.pdf)
## Demo
<table>
   <tr>
      <td><img alt="demo_clownfish" src="https://user-images.githubusercontent.com/2352985/56845088-8a1e5d00-68f6-11e9-9950-cab56cf32e80.jpg"></td>
      <td><img alt="demo_tiger" src="https://user-images.githubusercontent.com/2352985/56845090-8e4a7a80-68f6-11e9-9a51-b1da31d5ef77.jpg"></td>
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
slic = Slic(num_components=1600, compactness=10)
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
slic = SlicAvx2(num_components=1600, compactness=10)
assignment = slic.iterate(image) # Cluster Map
print(assignment)
print(slic.slic_model.clusters) # The cluster information of superpixels.
```

If your machine is ARM with NEON instruction set, which is commonly supported by recent mobile devices and even Raspberry Pi, you can make it two-fold faster by using `fast_slic.neon.SlicNeon` class instead of the original one.


## Performance

With max iteration set to 10, run times of slic implementations for 640x480 image are as follows:

| Implementation                                  | Run time(ms)   |
| -----------------------------------------       | --------------:|
| skimage.segment.slic                            | 216ms          |
| cv2.ximgproc.createSuperpixelSLIC.iterate       | 142ms          |
| fast_slic.Slic(single core build)               | 20ms           |
| fast_slic.avx2.SlicAvx2(single core build /w avx2 support)      | 12ms           |
| **fast_slic.Slic(w/ OpenMP support)**           | **8.8ms**       |
| **fast_slic.avx2.SlicAvx2(w/ OpenMP, avx2 support)**   | **5.6ms**       |

 
(RGB-to-CIELAB conversion time is not included. Tested with Ryzen 2600x 6C12T 4.0Hz O.C.)

## Known Issues
 * Windows build is quite slower compared to those of linux and mac. Maybe it is due to openmp overhead?

 
## Tips
 * It automatically removes small isolated area of pixels at cost of significant (but not huge) overhead. You can skip denoising process by setting `min_size_factor` to 0. (e.g. `Slic(num_components=1600, compactness=10, min_size_factor=0)`). The setting makes it 20-40% faster. 
 * To push to the limit, compile it with `FAST_SLIC_AVX2_FASTER` flag and get more performance gain. (though performance margin was small in my pc)
 
## TODO
 - [ ] Publish as a research paper
 - [x] Remove or merge small blobs
 - [x] Include simple CRF utilities
 - [x] Add tests
 - [x] Windows build
 - [x] More scalable parallel loop in cluster assignment. I suspect there is false sharing problem in the loop.
 - [x] would be great if I can optimize loop more. SIMD?

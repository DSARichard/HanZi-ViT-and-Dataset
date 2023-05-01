import cv2
import numpy as np

def pair(t):
  return t if isinstance(t, tuple) else (t, t)

def overlap_patches(img, patch_size, overlap_size):
  wi, hi = img.shape[:2]
  wp, hp = pair(patch_size)
  wo, ho = pair(overlap_size)
  assert (wi - wo)%(wp - wo) == 0 and (hi - ho)%(hp - ho) == 0, "Patch and overlap sizes must match image dimensions"
  nr, nc = (wi - wo)//(wp - wo), (hi - ho)//(hp - ho)
  img1 = np.empty((nr*wp, nc*hp) if(img.ndim == 2) else (nr*wp, nc*hp, img.shape[2]), img.dtype)
  for i in range(nr):
    for j in range(nc):
      r, c, r1, c1 = i*(wp - wo), j*(hp - ho), i*wp, j*hp
      img1[r1:r1 + wp, c1:c1 + hp] = img[r:r + wp, c:c + hp]
  return img1

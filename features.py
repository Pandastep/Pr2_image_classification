import numpy as np
from scipy.cluster.vq import vq

def build_histogram(descriptors, codebook, k):
    histogram = np.zeros(k)
    if descriptors is not None:
        words, _ = vq(descriptors, codebook)
        for idx in words:
            histogram[idx] += 1
    return histogram

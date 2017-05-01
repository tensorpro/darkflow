from __future__ import absolute_import

from darkflow.net.build import TFNet

class YOLO:
    def __init__(self, weights, cfg, mem_frac=1):
        options = {"model": cfg,
                   "load": weights,
                   "threshold": 0.1,
                   "gpu":mem_frac}
        self.net = TFNet(options)

    def __call__(self, img):
        return net.return_predict(img)

from pathlib import Path
from fastai.vision import *
from fastai import *
from fastai.vision.data import ImageDataLoaders
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *

path = 'sorted_data'


tfms = aug_transforms(do_flip = False, flip_vert = False, max_lighting = 0.1, max_zoom = 1.05, max_warp = 0., max_rotate = 5)
np.random.seed(42)
data = ImageDataLoaders.from_folder(path, train = ".", valid_pct = 0.2,
        ds_tfms = tfms, bs = 144, num_workers = 4)


data.show_batch(nrows = 3, figsize = (7, 8))

learn = vision_learner(data, models.resnet34, metrics = accuracy)
learn.fit_one_cycle(5)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr = slice(1e-5, 1e-4))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr = slice(1e-5, 1e-4))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, max_lr = slice(1e-5, 1e-4))

np.random.seed(42)
data = ImageDataLoaders.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=tfms, size=288, num_workers=4).normalize(imagenet_stats)
data.batch_size=25

learn.data = data
data.train_ds[0][0].shape

data.classes

data.show_batch(rows = 3, figsize = (7, 8))

learn.freeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-4))
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-5,1e-4))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-4))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-5))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, slice(1e-5,1e-4))

interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

len(data.valid_ds) == len(losses) == len(idxs)

interp.plot_confusion_matrix(figsize = (12, 12), dpi = 60)


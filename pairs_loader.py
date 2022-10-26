import os
import math
import skimage.io
from imageio import imread
from skimage.transform import resize
from torchvision.transforms.functional import resize as resize_tensor
import cv2
import random
import json
import numpy as np
import h5py
import torch
import utils


def get_image_filenames(dir, focuses=None):
    """Returns all files in the input directory dir that are images"""
    image_types = ('jpg', 'jpeg', 'tiff', 'tif', 'png',
                   'bmp', 'gif', 'exr', 'dpt', 'hdf5')
    if isinstance(dir, str):
        files = os.listdir(dir)
        exts = (os.path.splitext(f)[1] for f in files)
        if focuses is not None:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types and int(os.path.splitext(f)[0].split('_')[-1]) in focuses]
        else:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types]
        return images
    elif isinstance(dir, list):
        # Suppport multiple directories (randomly shuffle all)
        images = []
        for folder in dir:
            files = os.listdir(folder)
            exts = (os.path.splitext(f)[1] for f in files)
            images_in_folder = [os.path.join(folder, f)
                                for e, f in zip(exts, files)
                                if e[1:] in image_types]
            images = [*images, *images_in_folder]

        return images


class PairsLoader(torch.utils.data.IterableDataset):
    """Loads (phase, captured) tuples for forward model training

    Class initialization parameters
    -------------------------------

    :param data_path:
    :param plane_idxs:
    :param batch_size:
    :param image_res:
    :param shuffle:
    :param avg_energy_ratio:
    :param slm_type:


    """

    def __init__(self, data_path, plane_idxs=None, batch_size=1,
                 image_res=(800, 1280), shuffle=True,
                 avg_energy_ratio=None, slm_type='leto'):
        """

        """
        print(data_path)
        if isinstance(data_path, str):
            if not os.path.isdir(data_path):
                raise NotADirectoryError(f'Data folder: {data_path}')
            self.phase_path = os.path.join(data_path, 'phase')
            self.captured_path = os.path.join(data_path, 'captured')
        elif isinstance(data_path, list):
            self.phase_path = [os.path.join(path, 'phase')
                               for path in data_path]
            self.captured_path = [os.path.join(
                path, 'captured') for path in data_path]

        self.all_plane_idxs = plane_idxs
        self.avg_energy_ratio = avg_energy_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_res = image_res
        self.slm_type = slm_type.lower()
        self.im_names = get_image_filenames(self.phase_path)
        self.im_names.sort()

        # create list of image IDs with augmentation state
        self.order = ((i,) for i in range(len(self.im_names)))
        self.order = list(self.order)

    def __iter__(self):
        self.ind = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self

    def __len__(self):
        return len(self.im_names)

    def __next__(self):
        if self.ind < len(self.order):
            phase_idx = self.order[self.ind]

            self.ind += 1
            return self.load_pair(phase_idx[0])
        else:
            raise StopIteration

    def load_pair(self, filenum):
        phase_path = self.im_names[filenum]
        captured_path = os.path.splitext(os.path.dirname(phase_path))[0]
        captured_path = os.path.splitext(os.path.dirname(captured_path))[0]
        captured_path = os.path.join(captured_path, 'captured')

        # load phase
        phase_im_enc = imread(phase_path)
        im = (1 - phase_im_enc / np.iinfo(np.uint8).max) * 2 * np.pi - np.pi
        phase_im = torch.tensor(im, dtype=torch.float32).unsqueeze(0)

        _, captured_filename = os.path.split(
            os.path.splitext(self.im_names[filenum])[0])
        idx = captured_filename.split('/')[-1]

        # load focal stack
        captured_amps = []
        for plane_idx in self.all_plane_idxs:
            captured_filename = os.path.join(
                captured_path, f'{idx}_{plane_idx}.png')
            captured_intensity = utils.im2float(
                skimage.io.imread(captured_filename))
            captured_intensity = torch.tensor(
                captured_intensity, dtype=torch.float32)
            if self.avg_energy_ratio is not None:
                # energy compensation;
                captured_intensity /= self.avg_energy_ratio[plane_idx]
            captured_amp = torch.sqrt(captured_intensity)
            captured_amps.append(captured_amp)
        captured_amps = torch.stack(captured_amps, 0)

        return phase_im, captured_amps

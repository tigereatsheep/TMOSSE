# coding: utf-8
import cv2
import math
import glob
import pyhog
import numpy as np
import skimage.transform as transform
import skimage.color as color
import skimage.io as io


def repeat_to_third_dim(mat, repetitions=28):
    return np.tile(np.expand_dims(mat, 2), [repetitions])


def get_scale_sample(im, pos, base_target_size, scale_factors, cos_window, scale_model_size):

    n = len(scale_factors)
    for i in range(n):
        patch_size = np.floor(base_target_size * scale_factors[i])
        xs = np.floor(pos[1]) + np.arange(patch_size[1]) - np.floor(patch_size[1] / 2)
        ys = np.floor(pos[0]) + np.arange(patch_size[0]) - np.floor(patch_size[0] / 2)
        x_start = np.maximum(0, np.floor(pos[1]) - np.floor(patch_size[1] / 2))
        y_start = np.maximum(0, np.floor(pos[0]) - np.floor(patch_size[0] / 2))
        x_stop = np.minimum(im.shape[1] - 1, np.floor(pos[1]) + patch_size[1] - np.floor(patch_size[1] / 2))
        y_stop = np.minimum(im.shape[0] - 1, np.floor(pos[0]) + patch_size[0] - np.floor(patch_size[0] / 2))

        im_patch = im[y_start.astype(np.int32):y_stop.astype(np.int32),
                      x_start.astype(np.int32):x_stop.astype(np.int32)]

        im_patch_resized = transform.resize(im_patch, scale_model_size)

        temp_hog = pyhog.features_pedro(repeat_to_third_dim(im_patch_resized, 3), 4)
        temp = temp_hog[:, :, :32]

        if i == 0:
            features = np.zeros((temp.size, n))
        features[:, i] = temp.ravel() * cos_window[i]

    return features


def get_features(im):

    f = np.zeros((im.shape[0], im.shape[1], 28))
    if len(im.shape) == 2:
        f[:, :, 0] = im - 0.5
        f[1:-1, 1:-1, 1:] = pyhog.features_pedro(color.gray2rgb(im), 1)[:, :, :27]
    else:
        f[:, :, 0] = color.rgb2gray(im_patch) - 0.5
        f[1:-1, 1:-1, 1:] = pyhog.features_pedro(color.gray2rgb(im), 1)[:, :, :27]
    return f


def get_sample(im, pos, base_target_size, scale_factor, cos_window):

    h, w = im.shape[:2]

    if np.isscalar(base_target_size):
        base_target_size = np.array([base_target_size, base_target_size])

    patch_size = np.floor(base_target_size * scale_factor)
    if patch_size[0] < 1:
        patch_size[0] = 2 
    if patch_size[1] < 1:
        patch_size[1] = 2 

    x_start = np.maximum(0, np.floor(pos[1]) - np.floor(patch_size[1] / 2))
    y_start = np.maximum(0, np.floor(pos[0]) - np.floor(patch_size[0] / 2))
    x_stop = np.minimum(w - 1, np.floor(pos[1]) + patch_size[1] - np.floor(patch_size[1] / 2))
    y_stop = np.minimum(h - 1, np.floor(pos[0]) + patch_size[0] - np.floor(patch_size[0] / 2))
    im_patch = im[y_start.astype(np.int32):y_stop.astype(np.int32),
                  x_start.astype(np.int32):x_stop.astype(np.int32)]
    im_patch = transform.resize(im_patch, base_target_size)
    features = get_features(im_patch)
    features = repeat_to_third_dim(cos_window) * features
    return features


class Tracker(object):

    def __init__(self, **params):

        self.pos = None
        self.padding = params.get('padding', 2.0)
        self.output_sigma_factor = params.get('output_sigma_factor', 0.0625)
        self.scale_sigma_factor = params.get('scale_sigma_factor', 0.25)
        self.lbd = params.get('lambda', 0.01)
        self.lr = params.get('lr', 0.025)
        self.num_scales = params.get('num_scales', 17)
        self.scale_step = params.get('scale_step', 1.05)
        self.translation_model_max_area = \
                        params.get('translation_model_max_area', 1024 * 1024)
        self.scale_model_max_area = params.get('scale_model_max_area', 512)

        self.current_scale_factor = 1.0
        self.scale_model_factor = 1.0

        self.base_target_size = None
        self.search_size = None
        self.yf, self.spatial_cos_window = None, None
        self.ysf, self.scale_cos_window = None, None
        self.hf_den, self.hf_num = None, None
        self.sf_den, self.sf_num = None, None

        self.scale_model_size = None
        self.scale_factors = None

        self.min_scale_factor, self.max_scale_factor = None, None

    def initial(self, img, pos, target_size):

        self.pos = np.floor(pos)
        height, width = img.shape[:2]

        # Initialize the current scale factor
        if np.prod(target_size) > self.translation_model_max_area:
            self.current_scale_factor = \
                    sqrt(prod(target_size) / self.translation_model_max_area)
        self.base_target_size = np.array(target_size) / self.current_scale_factor

        # Initialize the first search window size
        self.search_size = np.floor(self.base_target_size * (1 + self.padding))

        # Initialize the spatial filter template in frequency domain
        self.yf = self.get_spatial_filter()

        # Initialize the scale model factor
        if np.prod(target_size) > self.scale_model_max_area:
            self.scale_model_factor = np.sqrt(self.scale_model_max_area / np.prod(target_size))
        self.scale_model_size = np.floor(target_size * self.scale_model_factor)

        # Initialize the scale filter template in frequency domain
        self.ysf = self.get_scale_filter()

        # Initialize Hanning windows
        self.spatial_cos_window = self.get_spatial_cos_window()
        self.scale_cos_window = self.get_scale_cos_window()

        # Compute the scale factors
        self.scale_factors = self.compute_scale_factors()

        # Compute min and max scale factor
        self.min_scale_factor, self.max_scale_factor = self.compute_min_max_scale_factor(height, width)
        self.update(img, init_tracker=True)

    def update(self, im, init_tracker=False):

        sample = get_sample(im, self.pos, self.search_size, self.current_scale_factor, self.spatial_cos_window)
        sample_f = np.fft.fft2(sample, axes=(0, 1))
        new_hf_num = repeat_to_third_dim(self.yf) * np.conj(sample_f)
        new_hf_den = np.sum(sample_f * np.conj(sample_f), axis=2)

        scale_sample = get_scale_sample(im, self.pos, self.base_target_size,
            self.current_scale_factor * self.scale_factors, self.scale_cos_window, self.scale_model_size)
        scale_sample_f = np.fft.fft(scale_sample, axis=1)
        new_sf_num = self.ysf * np.conj(scale_sample_f)
        new_sf_den = np.sum(scale_sample_f * np.conj(scale_sample_f), axis=0)

        if self.hf_den is None or init_tracker:
            self.hf_den = new_hf_den
            self.hf_num = new_hf_num
            self.sf_den = new_sf_den
            self.sf_num = new_sf_num
        else:
            self.hf_den = (1 - self.lr) * self.hf_den + self.lr * new_hf_den
            self.hf_num = (1 - self.lr) * self.hf_num + self.lr * new_hf_num
            self.sf_den = (1 - self.lr) * self.sf_den + self.lr * new_sf_den
            self.sf_num = (1 - self.lr) * self.sf_num + self.lr * new_sf_num

    def predict(self, im):
        sample = get_sample(im, self.pos, self.search_size, self.current_scale_factor, self.spatial_cos_window)
        sample_f = np.fft.fft2(sample, axes=(0, 1))
        response = np.real(np.fft.ifft2(np.sum(self.hf_num * sample_f, 2) / (self.hf_den + self.lbd), axes=(0, 1)))
        [row, col] = np.argwhere(response == np.max(response.ravel()))[0]
        self.pos += np.round((-self.search_size / 2 + np.array([row, col]) + 1) * self.current_scale_factor)

        scale_sample = get_scale_sample(im, self.pos, self.base_target_size,
            self.current_scale_factor * self.scale_factors, self.scale_cos_window, self.scale_model_size)
        scale_sample_f = np.fft.fft(scale_sample, axis=1)
        scale_response = np.real(np.fft.ifft(np.sum(self.sf_num * scale_sample_f, axis=0) /
                                             (self.sf_den + self.lbd)))

        recovered_scale = np.argwhere(scale_response == np.max(scale_response.ravel()))[0] - 1

        current_scale_factor = self.current_scale_factor * self.scale_factors[recovered_scale]

        if current_scale_factor < self.min_scale_factor:
            self.current_scale_factor = self.min_scale_factor
        elif current_scale_factor > self. max_scale_factor:
            self.current_scale_factor = self.max_scale_factor
        else:
            self.current_scale_factor = current_scale_factor

        return response, scale_response

    def get_spatial_filter(self):

        output_sigma = np.sqrt(np.prod(self.base_target_size)) * self.output_sigma_factor
        [rs, cs] = np.meshgrid(np.arange(1, self.search_size[0] + 1) - np.floor(self.search_size[0] / 2),
                           np.arange(1, self.search_size[1] + 1) - np.floor(self.search_size[1] / 2),
                           indexing='ij')
        y = np.exp(-0.5 * ((rs ** 2 + cs ** 2) / output_sigma ** 2))
        yf = np.fft.fft2(y, axes=(0, 1))
        return yf

    def get_scale_filter(self):

        scale_sigma = np.sqrt(self.num_scales) * self.scale_sigma_factor
        ss = np.arange(self.num_scales) - np.ceil(self.num_scales / 2)
        ys = np.exp(-0.5 * (ss ** 2) / scale_sigma ** 2)
        ysf = np.fft.fft(ys)
        return ysf

    def get_spatial_cos_window(self):

        cos_window = np.hanning(self.search_size[0]).reshape(-1, 1) * \
                     np.hanning(self.search_size[1])
        return cos_window

    def get_scale_cos_window(self):

        if self.num_scales % 2 == 0:
            cos_window = np.hanning(self.num_scales + 1)
            cos_window = cos_window[1:]
        else:
            cos_window = np.hanning(self.num_scales)
        return cos_window

    def compute_scale_factors(self):

        ss = np.arange(self.num_scales)
        scale_factors = self.scale_step ** (np.ceil(self.num_scales / 2) - (ss + 1))
        return scale_factors

    def compute_min_max_scale_factor(self, height, width):

        min_scale_factor = self.scale_step ** \
                           np.ceil(np.log(np.max(5 / self.search_size)) /
                                                 np.log(self.scale_step))
        max_scale_factor = self.scale_step ** \
                           np.floor(np.log(np.min(np.array([height, width]) /
                                    self.base_target_size)) /
                                    np.log(self.scale_step))
        return min_scale_factor, max_scale_factor


if __name__ == '__main__':
    
    tracker = Tracker()
    img_seqs = []
    for i in range(1, 1351):
        img_seqs.append('./dog1/imgs/img%05d.jpg' % i)
    init_pos = np.array([112, 139]) + np.array([36, 51]) / 2
    init_target_size = np.array([36, 51])
    tracker.initial(io.imread(img_seqs[0]), init_pos, init_target_size)
    image = io.imread(img_seqs[0])
    cv2.rectangle(image, (139, 112), (190, 148), (0, 255, 0))
    cv2.imwrite('./00001.jpg', image)
    for i in range(2, 200):
        image = io.imread(img_seqs[i])
        tracker.predict(image)
        tracker.update(image)
        pos = tracker.pos
        target_size = tracker.base_target_size
        cv2.rectangle(image, (int(pos[1] - target_size[1] / 2), int(pos[0] - target_size[0] / 2)),
                             (int(pos[1] + target_size[1] / 2), int(pos[0] + target_size[0] / 2)),
                             (0, 255, 0))
        cv2.imwrite('%05d.jpg' % i, image)

    


#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import cv2

class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class RunningStatsNormalizer(BaseNormalizer):
    def __init__(self, read_only=False):
        BaseNormalizer.__init__(self, read_only)
        self.needs_reset = True
        self.read_only = read_only

    def reset(self, x_size):
        self.m = np.zeros(x_size)
        self.v = np.zeros(x_size)
        self.n = 0.0
        self.needs_reset = False

    def state_dict(self):
        return {'m': self.m, 'v': self.v, 'n': self.n}

    def load_state_dict(self, stored):
        self.m = stored['m']
        self.v = stored['v']
        self.n = stored['n']
        self.needs_reset = False

    def __call__(self, x):
        if np.isscalar(x) or len(x.shape) == 1:
            # if dim of x is 1, it can be interpreted as 1 vector entry or batches of scalar entry,
            # fortunately resetting the size to 1 applies to both cases
            if self.needs_reset: self.reset(1)
            return self.nomalize_single(x)
        elif len(x.shape) == 2:
            if self.needs_reset: self.reset(x.shape[1])
            new_x = np.zeros(x.shape)
            for i in range(x.shape[0]):
                new_x[i] = self.nomalize_single(x[i])
            return new_x
        else:
            assert 'Unsupported Shape'

    def nomalize_single(self, x):
        is_scalar = np.isscalar(x)
        if is_scalar:
            x = np.asarray([x])

        if not self.read_only:
            new_m = self.m * (self.n / (self.n + 1)) + x / (self.n + 1)
            self.v = self.v * (self.n / (self.n + 1)) + (x - self.m) * (x - new_m) / (self.n + 1)
            self.m = new_m
            self.n += 1

        std = (self.v + 1e-6) ** .5
        x = (x - self.m) / std
        if is_scalar:
            x = np.asscalar(x)
        return x

class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        if not np.isscalar(x):
            x = np.asarray(x)
        return self.coef * x

class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        RescaleNormalizer.__init__(self, 1.0 / 255)

class GrayscaleImageNormalizer(BaseNormalizer):
    def __init__(self):
        BaseNormalizer.__init__(self)
        self.target_size=(144,144)

    def __call__(self, x):
        # Validate the input
        if not isinstance(x, np.ndarray):
            raise ValueError("Input observation must be a NumPy array.")
        
        if x.size == 0:
            raise ValueError("Input observation is empty")
        
        if x.dtype != np.uint8:
            x = x.astype(np.uint8)

        # Extract dimensions
        batch_size, channels, height, width = x.shape
        if channels != 3:
            raise ValueError(f"Expected input with 3 channels (RGB), but got {channels} channels.")

        # Initialize an array to hold the grayscale results
        grayscale_batch = np.zeros((batch_size, 1, height, width), dtype=np.float32)  # Grayscale format

        for i in range(batch_size):
            # Transpose to (H, W, C) for OpenCV, process, then transpose back
            rgb_image = np.transpose(x[i], (1, 2, 0))  # (H, W, C)
            grayscale = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

            # Resize the grayscale image
            resized = cv2.resize(grayscale, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize to [0, 1]
            normalized = resized / 255.0

            # Store in the output batch (H, W -> 1, H, W for grayscale)
            grayscale_batch[i, 0, :, :] = normalized

        return grayscale_batch


class SignNormalizer(BaseNormalizer):
    def __call__(self, x):
        return np.sign(x)

class RewardRunningStatsNormalizer(RunningStatsNormalizer):
    def __init__(self, read_only=False):
        RunningStatsNormalizer.__init__(self, read_only)
        self.needs_reset = True
        self.read_only = read_only

    def reset(self, x_size):
        self.m = np.zeros(x_size)
        self.v = np.zeros(x_size)
        self.n = 0.0
        self.needs_reset = False

    def state_dict(self):
        return {'m': self.m, 'v': self.v, 'n': self.n}

    def load_state_dict(self, stored):
        self.m = stored['m']
        self.v = stored['v']
        self.n = stored['n']
        self.needs_reset = False

    def __call__(self, x):
        if np.isscalar(x) or len(x.shape) == 1:
            # if dim of x is 1, it can be interpreted as 1 vector entry or batches of scalar entry,
            # fortunately resetting the size to 1 applies to both cases
            if self.needs_reset: self.reset(1)
            return self.nomalize_single(x)
        elif len(x.shape) == 2:
            if self.needs_reset: self.reset(x.shape[1])
            new_x = np.zeros(x.shape)
            for i in range(x.shape[0]):
                new_x[i] = self.nomalize_single(x[i])
            return new_x
        else:
            assert 'Unsupported Shape'

    def nomalize_single(self, x):
        is_scalar = np.isscalar(x)
        if is_scalar:
            x = np.asarray([x])

        if not self.read_only:
            new_m = self.m * (self.n / (self.n + 1)) + x / (self.n + 1)
            self.v = self.v * (self.n / (self.n + 1)) + (x - self.m) * (x - new_m) / (self.n + 1)
            self.m = new_m
            self.n += 1

        std = (self.v + 1e-6) ** .5
        x = x / std
        if is_scalar:
            x = np.asscalar(x)
        return x


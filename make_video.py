import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import OrderedDict
from torchvision import transforms

import wsol
from config import get_configs
from util import t2n, normalize

_INPUT_ERROR = "File not exists."


class ObjectVideoMaker(object):
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
        "OpenImages": 100,
    }
    _RESIZE_SIZE = (224, 224)
    _IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
    _IMAGE_STD_VALUE = [0.229, 0.224, 0.225]

    def __init__(self):
        self.args = get_configs()
        self.norm_method = self.args.norm_method
        self.percentile = self.args.percentile
        self.cam_threshold = self.args.cam_threshold
        self.concatenate_video = self.args.concatenate

        self.model = self._set_model()
        self.fps, self.frame_size, self.vid = self.get_video()
        self.result_vid = self.set_video()

    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("Loading model {}".format(self.args.architecture))
        model = wsol.__dict__[self.args.architecture](
            dataset_name=self.args.dataset_name,
            architecture_type=self.args.architecture_type,
            pretrained=self.args.pretrained,
            num_classes=num_classes,
            large_feature_map=self.args.large_feature_map,
            pretrained_path=self.args.pretrained_path,
            adl_drop_rate=self.args.adl_drop_rate,
            adl_drop_threshold=self.args.adl_threshold,
            acol_drop_threshold=self.args.acol_threshold,
            original_feature_map=self.args.original_feature_map,
        )
        # To Use Every GPU
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model = model.cuda()
        return model

    def get_video(self):
        input_path = os.path.join(os.getcwd(), self.args.input_name)
        print("File Path : ", input_path)
        if os.path.isfile(input_path):
            vid = cv2.VideoCapture(input_path)
        else:
            raise NotImplementedError(_INPUT_ERROR)
        vid_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vid.get(cv2.CAP_PROP_FPS)
        frame_size = (vid_width, vid_height)
        print(f"Frame Size : {frame_size}")

        return fps, frame_size, vid

    def set_video(self):
        fourcc = self.choose_codec()
        if self.concatenate_video:
            return cv2.VideoWriter(self.args.output_name, fourcc, self.fps,
                                   (self.frame_size[0], self.frame_size[1] * 2))
        else:
            return cv2.VideoWriter(self.args.output_name, fourcc, self.fps, self.frame_size)

    def choose_codec(self):
        cod_type = self.args.video_codec
        if cod_type == 'divx':
            return cv2.VideoWriter_fourcc(*"DIVX")
        elif cod_type == 'fmp4':
            return cv2.VideoWriter_fourcc(*"FMP4")
        elif cod_type == 'x264':
            return cv2.VideoWriter_fourcc(*"X264")
        elif cod_type == 'mjpg':
            return cv2.VideoWriter_fourcc(*"MJPG")
        else:
            return cv2.VideoWriter_fourcc(*"XVID")

    def release_video(self):
        self.vid.release()
        self.result_vid.release()

    def evaluate(self):
        self.model.eval()
        frames = []
        # Read Video and Write Video
        while True:
            retval, frame = self.vid.read()
            if not retval:
                break
            frames.append(frame)
        frames = np.array(frames).transpose(0, 3, 1, 2)
        frames_torch = torch.from_numpy(np.array(frames)).type(torch.FloatTensor)
        frames_torch = F.interpolate(frames_torch, self._RESIZE_SIZE)
        frames_torch = normalize(frames_torch, self._IMAGE_MEAN_VALUE, self._IMAGE_STD_VALUE)
        frames_torch = frames_torch.cuda()
        for frame_torch, frame_numpy in zip(frames_torch, frames):
            frame_torch = frame_torch.unsqueeze(dim=0)
            cam = t2n(self.model(x=frame_torch, labels=None, return_cam=True).squeeze())
            cam_resized = cv2.resize(cam, self.frame_size, interpolation=cv2.INTER_CUBIC)
            cam_normalized = self.normalize_scoremap(cam_resized)
            _, cam_threshold = cv2.threshold(
                src=cam_normalized,
                thresh=self.cam_threshold,
                maxval=1,
                type=cv2.THRESH_TRUNC)
            cam_normalized = np.expand_dims(cam_normalized, axis=0)
            # cam_normalize
            fuse_frame = frame_numpy - (255 * (1 - cam_normalized))
            fuse_frame = fuse_frame.transpose(1, 2, 0)
            fuse_frame[np.where(fuse_frame < 0)] = 0
            if self.concatenate_video:
                frame_numpy = frame_numpy.transpose(1, 2, 0)
                concatenate_frame = np.concatenate((frame_numpy, fuse_frame), axis=0)
                self.result_vid.write(concatenate_frame.astype(np.uint8))
            else:
                self.result_vid.write(fuse_frame.astype(np.uint8))

    def normalize_scoremap(self, cam):
        if np.isnan(cam).any():
            return np.zeros_like(cam)
        if cam.min() == cam.max():
            return np.zeros_like(cam)
        if self.norm_method == 'minmax':
            cam -= cam.min()
            cam /= cam.max()
        elif self.norm_method == 'max':
            cam = np.maximum(0, cam)
            cam /= cam.max()
        elif self.norm_method == 'pas':
            cam -= cam.min()
            cam_copy = cam.flatten()
            cam_copy.sort()
            maxx = cam_copy[int(cam_copy.size * 0.9)]
            cam /= maxx
            cam = np.minimum(1, cam)
        elif self.norm_method == 'ivr':
            cam_copy = cam.flatten()
            cam_copy.sort()
            minn = cam_copy[int(cam_copy.size * self.percentile)]
            cam -= minn
            cam = np.maximum(0, cam)
            cam /= cam.max()
        else:
            print('Norm not defined')
        return cam

    def load_checkpoint_used_path(self):
        checkpoint_path = self.args.checkpoint_path
        if os.path.isfile(checkpoint_path):
            new_state_dict = OrderedDict()
            checkpoint = torch.load(checkpoint_path)
            # For DataParallel
            if torch.cuda.device_count() == 1:
                for k, v in checkpoint['state_dict'].items():
                    if "module." in k:
                        name = k[7:]
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
                self.model.load_state_dict(new_state_dict)
            else:
                for k, v in checkpoint['state_dict'].items():
                    if "module." not in k:
                        name = "module." + k
                        new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)

            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))


def main():
    ovm = ObjectVideoMaker()
    ovm.evaluate()
    ovm.release_video()


if __name__ == '__main__':
    main()
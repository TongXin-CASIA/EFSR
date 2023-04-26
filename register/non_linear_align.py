import os
import cv2
import torch
import numpy as np
from abc import abstractmethod
from easydict import EasyDict
from register.PWCLite import PWCLite
from register.flow import warp_image_relative, warp_image_two_steps


def img2tensor(img, data_type=np.uint8):
    if data_type == np.uint16 or data_type == "uint16":
        img = torch.from_numpy((img / 65535.).astype(np.float32))
    else:
        img = torch.from_numpy(img) / 255.
    if len(img.shape) == 3:
        img = img.permute([2, 0, 1])
    else:
        img = img.unsqueeze(0)
    return img.unsqueeze(0)


def tensor2img(tensor, data_type=np.uint8):
    if data_type == np.uint16 or data_type == "uint16":
        tensor = (tensor * 65535).cpu().numpy().astype(np.uint16)
    else:
        tensor = (tensor * 255).cpu().numpy().astype(np.uint8)
    if tensor.shape[1] != 1:
        tensor = tensor.transpose([0, 2, 3, 1])
    return tensor.squeeze()


class NonLinearAlign:
    def __init__(self):
        pass

    @abstractmethod
    def generate_field(self, img_r, img_m):
        pass

    @abstractmethod
    def warp_with_field(self, img_m, field, mode):
        pass


class FlowAligner(NonLinearAlign):
    def __init__(self, model_path, is_context=True, device="cuda"):
        super(FlowAligner, self).__init__()
        cfg = EasyDict({
            'upsample': True,
            'n_frames': 2,
            'reduce_dense': True,
            'is_context': is_context,
        })
        self.device = device
        self.model = PWCLite(cfg)
        self.model = self.restore_model(self.model, model_path)
        self.model.to(self.device)
        self.model.eval()

    def estimate(self, img_r, img_m):
        """
        params: img_r
        params: img_m
        """
        if not isinstance(img_r, torch.Tensor):
            img_r = img2tensor(img_r)
        if not isinstance(img_m, torch.Tensor):
            img_m = img2tensor(img_m)
        if img_r.shape[1] == 1:
            img_r = img_r.repeat(1, 3, 1, 1)
        if img_m.shape[1] == 1:
            img_m = img_m.repeat(1, 3, 1, 1)
        # imgs = [self.transform(img_r), self.transform(img_m)]
        imgs = [img_r, img_m]
        img_pair = torch.cat(imgs, 1).to(self.device)
        return self.model(img_pair)['flows_fw'][0].detach().cpu()

    def generate_field(self, img_r, img_m):
        """
        params: img_r
        params: img_m
        """
        img_r = img2tensor(img_r)
        img_m = img2tensor(img_m)
        flow = self.estimate(img_r, img_m)
        return flow

    def warp_with_field(self, img_m, field, mode="bilinear"):
        """
        params: img_m
        params: field
        """
        data_type = img_m.dtype
        img_m = img2tensor(img_m, data_type)
        warped = warp_image_relative(img_m, field, mode)
        warped = tensor2img(warped, data_type)
        return warped

    def warp_with_field_two_steps(self, img_m, field1, field2, mode="bilinear"):
        """
        params: img_m
        params: field
        """
        data_type = img_m.dtype
        img_m = img2tensor(img_m, data_type)
        warped = warp_image_two_steps(img_m, field1, field2, mode)
        warped = tensor2img(warped, data_type)
        return warped

    def load_model(self, model_path):
        self.model.load_model(model_path)

    def load_checkpoint(self, model_path):
        weights = torch.load(model_path)
        epoch = None
        if 'epoch' in weights:
            epoch = weights.pop('epoch')
        if 'state_dict' in weights:
            state_dict = (weights['state_dict'])
        else:
            state_dict = weights
        return epoch, state_dict

    def restore_model(self, model, pretrained_file):
        epoch, weights = self.load_checkpoint(pretrained_file)

        model_keys = set(model.state_dict().keys())
        weight_keys = set(weights.keys())

        # load weights by name
        weights_not_in_model = sorted(list(weight_keys - model_keys))
        model_not_in_weights = sorted(list(model_keys - weight_keys))
        if len(model_not_in_weights):
            print('Warning: There are weights in model but not in pre-trained.')
            for key in (model_not_in_weights):
                print(key)
                weights[key] = model.state_dict()[key]
        if len(weights_not_in_model):
            print('Warning: There are pre-trained weights not in model.')
            for key in (weights_not_in_model):
                print(key)
            from collections import OrderedDict
            new_weights = OrderedDict()
            for key in model_keys:
                new_weights[key] = weights[key]
            weights = new_weights

        model.load_state_dict(weights)
        return model

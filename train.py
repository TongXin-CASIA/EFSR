import os
import cv2
import torch
import json
import time
import numpy as np
import torch.nn as nn
from copy import deepcopy
from easydict import EasyDict
from utils.torch_utils import init_seed, restore_model, save_checkpoint
from utils.warp_utils import flow_warp
from Architecture.pwclite import PWCLite
from Dataset.get_dataset import get_dataloader
from losses.get_loss import get_loss
from utils.misc_utils import AverageMeter
from transforms.ar_transforms.sp_transfroms import RandomAffineFlow
from transforms.ar_transforms.oc_transforms import run_slic_pt, random_crop
from skimage.metrics import structural_similarity as compute_ssim

from Experiment.serial_reg import ext_call_serial
from Experiment.pairwise_reg import ext_call_pairwise


def u8img(img):
    return (img * 255).astype(np.uint8)


def compute_ncc(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1 = (img1 - img1.mean()) / img1.std()
    img2 = (img2 - img2.mean()) / img2.std()
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    # return np.mean(np.multiply((img1 - np.mean(img1)), (img2 - np.mean(img2)))) / (np.std(img1) * np.std(img2))


def train_one_epoch(cfg, model, loss_func, optimizer, train_loader, device, epoch):
    # run atst: true
    # run ot: false
    # run st: true
    # mask st: true
    # method
    sp_transform = RandomAffineFlow(cfg.st_cfg, addnoise=cfg.st_cfg.add_noise).to(device)
    # define meters
    am_batch_time = AverageMeter()
    am_data_time = AverageMeter()
    model.train()
    end = time.time()

    key_meter_names = ['Loss', 'l_ph', 'l_sm', 'flow_mean', 'l_atst', 'l_ot']
    key_meters = AverageMeter(i=len(key_meter_names), precision=4)

    for i_step, data in enumerate(train_loader):
        # read data to device
        img1, img2 = data['img1'].to(device), data['img2'].to(device)
        img_pair = torch.cat([img1, img2], 1)

        # measure data loading time
        am_data_time.update(time.time() - end)

        # run 1st pass
        res_dict = model(img_pair, with_bk=True)
        flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
        flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flows_12, flows_21)]
        loss, l_ph, l_sm, flow_mean = loss_func(flows, img_pair)

        flow_ori = res_dict['flows_fw'][0].detach()

        if cfg.run_atst:
            # true
            img1, img2 = data['img1_ph'].to(device), data['img2_ph'].to(device)

            # construct augment sample
            noc_ori = loss_func.pyramid_occu_mask1[0]  # non-occluded region
            s = {'imgs': [img1, img2], 'flows_f': [flow_ori], 'masks_f': [noc_ori]}
            st_res = sp_transform(deepcopy(s)) if cfg.run_st else deepcopy(s)  # true
            flow_t, noc_t = st_res['flows_f'][0], st_res['masks_f'][0]

            # run 2nd pass
            img_pair = torch.cat(st_res['imgs'], 1)
            flow_t_pred = model(img_pair, with_bk=False)['flows_fw'][0]

            if not cfg.mask_st:
                # False
                noc_t = torch.ones_like(noc_t)
            l_atst = ((flow_t_pred - flow_t).abs() + cfg.ar_eps) ** cfg.ar_q
            l_atst = (l_atst * noc_t).mean() / (noc_t.mean() + 1e-7)

            loss += cfg.w_ar * l_atst
        else:
            l_atst = torch.zeros_like(loss)

        if cfg.run_ot:
            # false
            img1, img2 = data['img1_ph'].to(device), data['img2_ph'].to(device)
            # run 3rd pass
            img_pair = torch.cat([img1, img2], 1)

            # random crop images
            img_pair, flow_t, occ_t = random_crop(img_pair, flow_ori, 1 - noc_ori, cfg.ot_size)

            # slic 200, random select 8~16
            if cfg.ot_slic:
                img2 = img_pair[:, 3:]
                seg_mask = run_slic_pt(img2, n_seg=200,
                                       compact=cfg.ot_compact, rd_select=[8, 16],
                                       fast=cfg.ot_fast).type_as(img2)  # Nx1xHxW
                noise = torch.rand(img2.size()).type_as(img2)
                img2 = img2 * (1 - seg_mask) + noise * seg_mask
                img_pair[:, 3:] = img2

            flow_t_pred = model(img_pair, with_bk=False)['flows_fw'][0]
            noc_t = 1 - occ_t
            l_ot = ((flow_t_pred - flow_t).abs() + cfg.ar_eps) ** cfg.ar_q
            l_ot = (l_ot * noc_t).mean() / (noc_t.mean() + 1e-7)

            loss += cfg.w_ar * l_ot
        else:
            l_ot = torch.zeros_like(loss)

        # update meters
        key_meters.update(
            [loss.item(), l_ph.item(), l_sm.item(), flow_mean.item(),
             l_atst.item(), l_ot.item()],
            img_pair.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        # loss.backward()

        scaled_loss = 1024. * loss
        scaled_loss.backward()

        for param in [p for p in model.parameters() if p.requires_grad]:
            if param.grad is not None:
                param.grad.data.mul_(1. / 1024)

        # clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        # measure elapsed time
        am_batch_time.update(time.time() - end)
        end = time.time()
        if i_step % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {am_batch_time.val[0]:.3f} ({am_batch_time.avg[0]:.3f})\t'
                  'Data {am_data_time.val[0]:.3f} ({am_data_time.avg[0]:.3f})\t'
                  'Loss {key_meters.val[0]:.4f} ({key_meters.avg[0]:.4f})\t'
                  'l_ph {key_meters.val[1]:.4f} ({key_meters.avg[1]:.4f})\t'
                  'l_sm {key_meters.val[2]:.4f} ({key_meters.avg[2]:.4f})\t'
                  'flow_mean {key_meters.val[3]:.4f} ({key_meters.avg[3]:.4f})\t'
                  'l_atst {key_meters.val[4]:.4f} ({key_meters.avg[4]:.4f})\t'
                  'l_ot {key_meters.val[5]:.4f} ({key_meters.avg[5]:.4f})'.format(
                epoch, i_step, len(train_loader), am_batch_time=am_batch_time,
                am_data_time=am_data_time, key_meters=key_meters))
    print(
        'Epoch: [{0}][{1}/{2}]\t, Time {batch_time.val[0]:.3f} ({batch_time.avg[0]:.3f})\t, Data {data_time.val[0]:.3f} '
        '({data_time.avg[0]:.3f})\t, {key_meter}'.format(epoch, len(train_loader), len(train_loader),
                                                         batch_time=am_batch_time, data_time=am_data_time,
                                                         key_meter=key_meters))
    return key_meters

def valid(cfg, model, valid_loader, epoch):
    with torch.no_grad():
        model.eval()
        ssim_meter = AverageMeter()
        ncc_meter = AverageMeter()
        for i_step, data in enumerate(valid_loader):
            # TODO Notice batch size missing
            img1, img2 = data['img1'].to(cfg.device), data['img2'].to(cfg.device)
            img_pair = torch.cat([img1, img2], 1)
            flow_t_pred = model(img_pair, with_bk=False)['flows_fw'][0]
            # warp img2 to img1
            img2_warp = flow_warp(img2, flow_t_pred)
            # convert to numpy
            img1, img2, img2_warp = img1[0, 0].cpu().numpy(), img2[0, 0].cpu().numpy(), img2_warp[0, 0].cpu().numpy()
            # compute metrics
            ssim_meter.update(compute_ssim(img1, img2_warp).item())
            ncc_meter.update(compute_ncc(img1, img2_warp).item())
            # save result img2_warp, overlap image
            # cv2.imwrite("rst/{}/{}_{}_{}.png".format(cfg.name, epoch, i_step, "w"), u8img(img2_warp))
            # cv2.imwrite("rst/{}/{}_{}_{}.png".format(cfg.name, epoch, i_step, "b"), u8img((img1 + img2) / 2.0))
            # cv2.imwrite("rst/{}/{}_{}_{}.png".format(cfg.name, epoch, i_step, "n"), u8img((img1 + img2_warp) / 2.0))
    print('Valid Epoch: [{0}][{1}/{2}]\t, SSIM {ssim:.4f}\t, NCC {ncc:.4f}'.format(0, len(valid_loader),
                                                                                   len(valid_loader),
                                                                                   ssim=ssim_meter.avg[0],
                                                                                   ncc=ncc_meter.avg[0])

def train(cfg, model, loss_func, optimizer, train_loader, valid_loader, test_loader):
    model = model.to(cfg.device)
    # create directory of rst/
    if not os.path.exists("rst/" + cfg.name):
        os.makedirs("rst/" + cfg.name)
    for epoch in range(cfg.train.epoch_num):
        key_meters = train_one_epoch(cfg.train, model, loss_func, optimizer, train_loader, cfg.device,
                                     epoch)
        # valid(cfg, model, test_loader, epoch + 1)
        torch.save(model.state_dict(), 'models/{}_{}.pth'.format(cfg.name, epoch))
        serial_rst = ext_call_serial(cfg.name, 'models/{}_{}.pth'.format(cfg.name, epoch), cfg.device)
        pair_rst = ext_call_pairwise(cfg.name, 'models/{}_{}.pth'.format(cfg.name, epoch), cfg.device)

        with open(rst_file, 'a') as f:
            f.write("{},{},{},{},{}\n".format(cfg.name, 'models/{}_{}.pth'.format(cfg.name, epoch)
                                              , serial_rst, pair_rst[0], pair_rst[1]))


configs_lst = ["FE_ft_c"]
if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    # create a rst.csv
    for cfg_name in configs_lst:
        cfg_path = 'configs/{}.json'.format(cfg_name)
        print('Loading {}'.format(cfg_path))
        init_seed(0)
        with open(cfg_path) as f:
            cfg = EasyDict(json.load(f))

        train_loader, valid_loader, test_loader = get_dataloader(cfg)

        model = PWCLite(cfg.model)
        model = restore_model(model, cfg.train.pretrained_model)
        loss = get_loss(cfg.loss)
        optimizer = torch.optim.Adam(model.parameters(), cfg.train.lr, betas=(cfg.train.momentum, cfg.train.beta),
                                     eps=1e-7)
                                     
        train(cfg, model, loss, optimizer, train_loader, valid_loader, test_loader)
        torch.save(model.state_dict(), 'models/{}.pth'.format(cfg.name))

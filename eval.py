import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
import time

def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()  # Add an adder for SSIM

        # Hardware warm-up
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, _ = data
            input_img = input_img.to(device)
            tm = time.time()
            _ = model(input_img)
            _ = time.time() - tm

            if iter_idx == 20:
                break

        # Main Evaluation
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            tm = time.time()

            pred = model(input_img)[2]

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

            # Calculate PSNR
            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)

            # Calculate SSIM
            min_dim = min(pred_numpy.shape[-2], pred_numpy.shape[-3])
            win_size = min(7, min_dim)
            ssim_value = ssim(label_numpy, pred_numpy, data_range=1, win_size=win_size, channel_axis=-1)
            ssim_adder(ssim_value)

            print('%d iter PSNR: %.2f SSIM: %.4f time: %f' % (iter_idx + 1, psnr, ssim_value, elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.4f' % (ssim_adder.average()))  # Print average SSIM
        print("Average time: %f" % adder.average())

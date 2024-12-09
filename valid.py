import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gopro = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()

    with torch.no_grad():

        for idx, data in enumerate(gopro):
            input_img, label_img = data
            input_img = input_img.to(device)
            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(input_img)

            pred_clip = torch.clamp(pred[2], 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            # Calculate PSNR
            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)

            # Calculate SSIM
            min_dim = min(p_numpy.shape[-2], p_numpy.shape[-3])
            win_size = min(7, min_dim)
            ssim_value = ssim(label_numpy, p_numpy, data_range=1, win_size=win_size, channel_axis=-1)
            ssim_adder(ssim_value)

    model.train()
    return psnr_adder.average(), ssim_adder.average()  # Return both PSNR and SSIM

import math


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, cal_type='y'):
    if hr.nelement() == 1:
        return 0

    diff = (sr - hr) / rgb_range

    if cal_type == 'y':
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

    if scale == 1:
        valid = diff
    else:
        valid = diff[..., scale:-scale, scale:-scale]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

import lpips
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores


def calculate_lpips_score(source_image, target_image):
    d = loss_fn_alex(source_image, target_image)
    return d


def calculate_aed_score(source_image, target_image):
    return np.linalg.norm(source_image - target_image)

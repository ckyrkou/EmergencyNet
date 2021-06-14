import albumentations
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import cv2

def add_random_shadow(image):
    w,h = image.shape[:2]
    top_y = w*np.random.uniform()
    top_x = 0
    bot_x = h
    bot_y = w*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    cond1 = shadow_mask==1
    cond0 = shadow_mask==0
    h, l, s = cv2.split(image_hls)

    l = l.astype(np.float32)

    if np.random.randint(2)==1:
        l[cond1] = l[cond1] - (20+np.random.randint(50))
    else:
        l[cond0] = l[cond0] - (20+np.random.randint(50))

    l[np.where(l < 0.0)] = 0
    l[np.where(l > 255.0)] = 255.0

    l = np.array(l, dtype=np.uint8)

    image_hls = cv2.merge((h, l, s))

    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2BGR)
    return image


class AddShadow(ImageOnlyTransform):
    def __init__(
        self, always_apply=False, p=1.0
    ):
        super(AddShadow, self).__init__(always_apply, p)


    def apply(self, img,**params):
        return add_random_shadow(img)



def create_augmentations(img_height=224,img_width=224,p=0.1):
    AUGMENTATIONS = albumentations.Compose([
        albumentations.Resize(img_height, img_width, p=1.),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.IAAPerspective(p=p, scale=(0.01, 0.05)),
        albumentations.GridDistortion(p=p, distort_limit=0.2), albumentations.GridDistortion(p=p,distort_limit=0.2),
        albumentations.CoarseDropout(p=p, max_holes=10, max_height=25, max_width=25),
        albumentations.GaussNoise(p=p,var_limit=(40.0, 70.0)),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, interpolation=1,border_mode=4, always_apply=False, p=2*p),
        albumentations.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=2*p),
        albumentations.Blur(p=p,blur_limit=10),
        albumentations.ToGray(p=p),
        albumentations.ChannelShuffle(p=0.05),
        albumentations.RandomGamma(p=p,gamma_limit=(20,200)),
        AddShadow(p=p),
    ])
    return AUGMENTATIONS


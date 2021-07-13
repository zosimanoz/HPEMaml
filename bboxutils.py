from torchvision import transforms
import torchvision
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.autograd import Variable

from facenet_pytorch import MTCNN, InceptionResnetV1


def crop_image(cv2_frame, x1, y1, x2, y2, w, h, ad, img_w, img_h):
    xw1 = max(int(x1 - ad * w), 0)
    yw1 = max(int(y1 - ad * h), 0)
    xw2 = min(int(x2 + ad * w), img_w - 1)
    yw2 = min(int(y2 + ad * h), img_h - 1)

    # Crop image
    img = cv2_frame[yw1:yw2 + 1, xw1:xw2 + 1, :]
    return img, xw1, yw1, xw2, yw2


def img_transform(img, transformations):
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img).cuda()
    return img


def get_bounding_box(frame):

    img_h, img_w, _ = np.shape(frame)
    ad = 0.2
    bounding_box_arr = []
    face_keypoints_arr = []
    w_arr = []

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=300, margin=0)

    box_cord, probs, landmarks = mtcnn.detect(frame, landmarks=True)
    return box_cord, probs, landmarks


def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format

        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

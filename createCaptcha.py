import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import transform as tf
from matplotlib import pyplot as plt


def create_captcha(text, shear=0, size=(200,40), scale=1):
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(r'Coval-Black.ttf', 22)
    draw.text((2,2), text, fill=1, font=font)

    image = np.array(im)

    affine_tf = tf.AffineTransform(shear=shear)
    image_tf = tf.warp(image, affine_tf)

    return image_tf/image_tf.max()  # value fall between 0 and 1


# image = create_captcha("CONY", shear=0.2)
# plt.imshow(image, cmap='Greys')
# plt.savefig('CONY.png')

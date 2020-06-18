from skimage.measure import label, regionprops
from datasetPrepare.createCaptcha import create_captcha
from matplotlib import pyplot as plt


def segment_image(image):
    labeled_image = label(image > 0)
    subimages = []
    for region in regionprops(labeled_image):
        start_x, start_y, end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x, start_y:end_y])
    if len(subimages) == 0:
        return [image, ]
    return subimages


image_cony = create_captcha("LUNANA", shear=0.2)
subimages_cony = segment_image(image_cony)

f, axes = plt.subplots(1, len(subimages_cony), figsize=(10,3))
for i in range(len(subimages_cony)):
    axes[i].imshow(subimages_cony[i], cmap='Greys')
plt.savefig('slice.png')
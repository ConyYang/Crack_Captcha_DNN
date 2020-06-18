from sklearn.utils import check_random_state
from datasetPrepare.createCaptcha import create_captcha
import numpy as np
from skimage.transform import resize
from datasetPrepare.sliceImg import segment_image
from sklearn.preprocessing import OneHotEncoder

onehot = OneHotEncoder()

random_state = check_random_state(2)
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
shear_values = np.arange(0, 0.36, 0.03)


def generate_Dataset(random_state=None):
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)
    return create_captcha(letter, shear, size=(30, 30)), letters.index(letter)


# image, letter_index = generate_Dataset(random_state)
# plt.imshow(image, cmap='ocean')
# plt.savefig('random.png')
# print('The True Letter is: {}'.format(letters[letter_index]))

image_dataset, image_indexes = zip(*(generate_Dataset(random_state) for i in range(3000)))

# onehot encode indexes shape(3000, 26)
image_indexes = np.array(image_indexes)
label = onehot.fit_transform(image_indexes.reshape(image_indexes.shape[0], 1))
label = label.todense()
label = np.array(label)

# lower pixels to 20*20 shape(3000, 400)
image_dataset = np.array(image_dataset, dtype='float')
dataset = np.array([resize(segment_image(sample)[0], (20, 20)) for sample in image_dataset])
# 3d to 2d
images = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))


np.savetxt('images.csv', images, delimiter=',', fmt='%f')
np.savetxt('label.csv', label, delimiter=',', fmt='%d')
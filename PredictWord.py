from sliceImg import segment_image
import numpy as np
from skimage import transform as tf
from createDataset import letters
import pickle
from createCaptcha import create_captcha

def predict_captcha(captcha_image, clf):
    subimages = segment_image(captcha_image)

    dataset = np.array([tf.resize(subimage, (20, 20)) for subimage in subimages])
    X_test = dataset.reshape((dataset.shape[0], dataset.shape[1] * dataset.shape[2]))
    y_pred = clf.predict_proba(X_test)
    predictions = np.argmax(y_pred, axis=1)

    assert len(y_pred) == len(X_test)

    predict_words = str.join("", [letters[prediction] for prediction in predictions])
    return predict_words

MLP_clf = pickle.load(open('MultiPerceptronModel.sav', 'rb'))
word = "CONYAA"
captcha_Img = create_captcha(word, shear=0.2)
print(predict_captcha(captcha_Img, MLP_clf))
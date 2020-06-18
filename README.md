# Crack Captcha use DNN

### Intro
This app io
### Set up
pip install Pillow==2.2.2
pip install scikit-image
!wget -nc http://labfile.oss.aliyuncs.com/courses/1133/Coval-Black.ttf
pip install pandas
pip install scikit-learn
pip install streamlit 
pip install flask

### Mechanism 
We use Multilayer Perceptron to train the model.
Parameters:
- hidden_layer_sizes: the #no. of hidden neurons
- random_state: random seed. can be set for repetitive training
- activation function: default: ReLu()

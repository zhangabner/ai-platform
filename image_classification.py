
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate)
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras
from keras.models import Model


import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps
import cv2
from sklearn.utils import class_weight, shuffle
from keras.losses import binary_crossentropy, categorical_crossentropy
#from keras.applications.resnet50 import preprocess_input
from keras.applications.densenet import DenseNet121,DenseNet169
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score
from keras.utils import Sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import imgaug as ia
import random
from os.path import isfile

WORKERS = 2
CHANNEL = 3

import warnings
warnings.filterwarnings("ignore")
SIZE = 300
IMG_SIZE =SIZE
NUM_CLASSES = 5


# In[4]:

df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')


# ### Visualising some sample pictures of different classes.

# In[5]:

def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'id_code']
        image_id = df.loc[i,'diagnosis']
        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)
    
    plt.tight_layout()

#display_samples(df_train)


# In[6]:



train_2015 = "../input/retinopathy-train-2015/rescaled_train_896/rescaled_train_896/"
train      = '../input/aptos2019-blindness-detection/train_images/'
test       = '../input/aptos2019-blindness-detection/test_images/'
test_2015 = "/kaggle/input/2015test34/2015test34/2015test34/"

# old_train1 = pd.read_csv('../input/retinopathy-train-2015/rescaled_train_896/trainLabels.csv')
train_csv  =pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

duplicatedlist1=np.array(['86b3a7929bec','a4012932e18d','a19507501b40','d51c2153d151'
,'79ce83c07588','878a3a097436','c9f0dc2c8b43','bd5013540a13'
,'92b0d27fc0ec','857002ed4e49','3e86335bc2fd','e135d7ba9a0e'
,'84b79243e430','bd34a0639575','ec6f1797a25a','94372043d55b'
,'d81b6ed83bc2','badb5ff8d3c7','a8582e346df0','c027e5482e8c'
,'ca6842bfcbc9','a15652b22ab8','2c2aa057afc5','77a9538b8362'
,'c05b7b4c22fe','65e51e18242b','cc12453ea915','e2c3b037413b'
,'9f4132bd6ed6','1b862fb6f65d','bf7b4eae7ad0','c6a8f8f998a2'
,'81914ceb4e74','d6b109c82067','65c958379680','dc3c0d8ee20b'
,'c0e509786f7f','f9e1c439d4c8','3c53198519f7','98104c8c67eb'
,'668a319c2d23','5a36cea278ae','3044022c6969','ff0740cb484a'
,'ea05c22d92e9','1f07dae3cadb','f920ccd926db','f6f3ea0d2693'
,'c2d2b4f536da','9a94e0316ee3','e4e343eaae2a','a3b2e93d058b'
,'c1c8550508e0','ff52392372d3','cac40227d3b2','36ec36c301c1'
,'e1fb532f55df','2b21d293fdf2','81b0a2651c45','d567a1a22d33'
,'fea14b3d44b0','6b00cb764237','d801c0a66738','595446774178'
,'ea9e0fb6fb0b','91cbe1c775ef','d1cad012a254','8a759f94613a'
,'7bf981d9c7fe','b10fca20c885','98e8adcf085c','cd4e7f9fa1a9'
,'8acffaf1f4b9','e7a7187066ad','b376def52ccc','3ca637fddd56'
,'91b6ebaa3678','a56230242a95','c8d2d32f7f29','6cb98da77e3e'
,'e037643244b7','be161517d3ac','da0a1043abf7','33105f9b3a04'
,'ee2c2a5f7d0e','eadc57064154','8d7bb0649a02','d144144a2f3f'
,'c68dfa021d62','d994203deb64','63a03880939c','e76a9cbb2a8c'
,'ba2624883599','59bd19c1c5bb','f4d3777f2710','9c52b87d01f1'
,'ed3a0fc5b546','19e350c7c83c','a47432cd41e7','b8ebedd382de'
,'f9ecf1795804','576e189d23d4','1e143fa3de57','ca25745942b0'
,'435d900fa7b2','94b1d8ad35ec','df4913ca3712','86baef833ae0'
,'38fe9f854046','a7b0d0c51731'])

for id_code in duplicatedlist1:
    train_csv=train_csv[train_csv['id_code']!=id_code]
    

def expand_path(p):
    p = str(p)
    if isfile(train + p + ".png"):
        return train + (p + ".png")
    if isfile(train_2015 + p + '.png'):
        return train_2015 + (p + ".png")
    if isfile(test_2015 + p + '.jpeg'):
        return test_2015 + (p + ".jpeg")
    if isfile(test + p + ".png"):
        return test + (p + ".png")
    return p
#The Code from: https://www.kaggle.com/ratthachat/aptos-updated-albumentation-meets-grad-cam

def crop_image_from_gray2(img,tol=7,threth_h=80,threth_w=90,cut_h=6,cut_w=1):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        nonblackrate = mask.sum()/(img.shape[0]*img.shape[1])    
        nonblackrate = int(nonblackrate*100)
        if nonblackrate < threth_h:
            if cut_h==0:
                img = img
            else:
                cutrate = random.choice([cut_h,cut_h+1,cut_h+2,cut_h+3,cut_h+4])
                new_heigth0 = int (img.shape[0]*0.01*cutrate)
                cutrate = random.choice([cut_h,cut_h+1,cut_h+2,cut_h+3,cut_h+4])
                new_heigth01 = int (img.shape[0]*0.01*cutrate)
                new_heigth1 = img.shape[0] - new_heigth01
                img = img[new_heigth0:new_heigth1,:]
            
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        nonblackrate = mask.sum()/(img.shape[0]*img.shape[1])
        nonblackrate = int(nonblackrate*100)
        
        if nonblackrate < threth_w:
            if cut_w==0:
                img = img
            else:
                cutrate = random.choice([cut_w,cut_w+1,cut_w+2,cut_w+3,cut_w+4])
                new_heigth0 = int (img.shape[1]*0.01*cutrate)
                cutrate = random.choice([cut_w,cut_w+1,cut_w+2,cut_w+3,cut_w+4])
                new_heigth01 = int (img.shape[1]*0.01*cutrate)
                new_heigth1 = img.shape[1] - new_heigth01
                img = img[:,new_heigth0:new_heigth1]
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        nonblackrate = mask.sum()/(img.shape[0]*img.shape[1])
        nonblackrate = int(nonblackrate*100)
        
        return img
    
# In[7]:

x = train_csv['id_code']
y = train_csv['diagnosis']


# In[8]:

y = to_categorical(y, num_classes=NUM_CLASSES)
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15,
                                                      stratify=y, random_state=8)
print(train_x.shape)
print(train_y.shape)
print(valid_x.shape)
print(valid_y.shape)


# In[9]:

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10), # rotate by -45 to +45 degrees
                shear=(-5, 5), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
                    ]),
                    iaa.Invert(0.01, per_channel=True), # invert color channels
                    iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.9, 1.1), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-1, 0),
                            first=iaa.Multiply((0.9, 1.1), per_channel=True),
                            second=iaa.ContrastNormalization((0.9, 1.1))
                        )
                    ]),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True)


# In[10]:

class My_Generator(Sequence):

    def __init__(self, image_filenames, labels,
                 batch_size, is_train=True,
                 mix=False, augment=False):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.is_train = is_train
        self.is_augment = augment
        if(self.is_train):
            self.on_epoch_end()
        self.is_mix = mix

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        if(self.is_train):
            return self.train_generate(batch_x, batch_y)
        return self.valid_generate(batch_x, batch_y)

    def on_epoch_end(self):
        if(self.is_train):
            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
        else:
            pass
    
    def mix_up(self, x, y):
        lam = np.random.beta(0.2, 0.4)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)        
        
        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]
        
        return mixed_x, mixed_y

    def train_generate(self, batch_x, batch_y):
        batch_images = []
        for (sample, label) in zip(batch_x, batch_y):
            
#             img = cv2.imread('../input/aptos2019-blindness-detection/train_images/'+sample+'.png')
#             img = cv2.resize(img, (SIZE, SIZE))
#             img = cv2.addWeighted( img,4, cv2.GaussianBlur( img , (0,0) ,  10) ,-4 ,128)
            
            p = sample
            p_path = expand_path(p)
            image = cv2.imread(p_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = crop_image_from_gray2(image)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            img = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)


            if(self.is_augment):
                img = seq.augment_image(img)
            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255
        batch_y = np.array(batch_y, np.float32)
        if(self.is_mix):
            batch_images, batch_y = self.mix_up(batch_images, batch_y)
        return batch_images, batch_y

    def valid_generate(self, batch_x, batch_y):
        batch_images = []
        for (sample, label) in zip(batch_x, batch_y):
#             img = cv2.imread('../input/aptos2019-blindness-detection/train_images/'+sample+'.png')
#             img = cv2.resize(img, (SIZE, SIZE))
#             img = cv2.addWeighted( img,4, cv2.GaussianBlur( img , (0,0) ,  10) ,-4 ,128)
            p = sample
            p_path = expand_path(p)
            image = cv2.imread(p_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = crop_image_from_gray2(image)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            img = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)

            batch_images.append(img)
        batch_images = np.array(batch_images, np.float32) / 255
        batch_y = np.array(batch_y, np.float32)
        return batch_images, batch_y


# In[11]:

get_ipython().system('ls /kaggle/input')


# In[12]:

def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = DenseNet121(include_top=False,
                   weights=None,
                   input_tensor=input_tensor)
    base_model.load_weights("../input/densenet-keras/DenseNet-BC-121-32-no-top.h5")
#     x1 = GlobalAveragePooling2D()(base_model.output)
#     x2 = GlobalMaxPooling2D()(base_model.output)
# # to account for missing values from the attention model
#     x = concatenate([x1,x2], axis=-1)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)

    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    final_output = Dense(n_out, activation='softmax', name='final_output')(x)
    model = Model(input_tensor, final_output) 
    return model


# In[13]:

# create callbacks list
from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                             EarlyStopping, ReduceLROnPlateau,CSVLogger)

epochs = 30; batch_size = 32
checkpoint = ModelCheckpoint('../working/densenet_.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 
                                   verbose=1, mode='auto', epsilon=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=9)

csv_logger = CSVLogger(filename='../working/training_log.csv',
                       separator=',',
                       append=True)

train_generator = My_Generator(train_x, train_y, 128, is_train=True)
train_mixup = My_Generator(train_x, train_y, batch_size, is_train=True, mix=False, augment=True)
valid_generator = My_Generator(valid_x, valid_y, batch_size, is_train=False)

model = create_model(
    input_shape=(SIZE,SIZE,3), 
    n_out=NUM_CLASSES)


# In[14]:

# reference link: https://www.kaggle.com/christofhenkel/weighted-kappa-loss-for-keras-tensorflow
def kappa_loss(y_true, y_pred, y_pow=2, eps=1e-12, N=5, bsize=32, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""

    with tf.name_scope(name):
        y_true = tf.to_float(y_true)
        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)
    
        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))
    
        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)
    
        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)
    
        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                              tf.to_float(bsize))
    
        return nom*0.5 / (denom + eps) + categorical_crossentropy(y_true, y_pred)*0.5


# In[15]:

from keras.callbacks import Callback
class QWKEvaluation(Callback):
    def __init__(self, validation_data=(), batch_size=64, interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.batch_size = batch_size
        self.valid_generator, self.y_val = validation_data
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_generator(generator=self.valid_generator,
                                                  steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),
                                                  workers=1, use_multiprocessing=False,
                                                  verbose=1)
            def flatten(y):
                return np.argmax(y, axis=1).reshape(-1)
            
            score = cohen_kappa_score(flatten(self.y_val),
                                      flatten(y_pred),
                                      labels=[0,1,2,3,4],
                                      weights='quadratic')
            print("\n epoch: %d - QWK_score: %.6f \n" % (epoch+1, score))
            self.history.append(score)
            if score >= max(self.history):
                print('saving checkpoint: ', score)
                self.model.save('../working/densenet_bestqwk.h5')

qwk = QWKEvaluation(validation_data=(valid_generator, valid_y),
                    batch_size=batch_size, interval=1)


# In[16]:

# warm up model
for layer in model.layers:
    layer.trainable = False

for i in range(-3,0):
    model.layers[i].trainable = True

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-3))

model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(float(len(train_y)) / float(128)),
    epochs=2,
    workers=WORKERS, use_multiprocessing=True,
    verbose=1,
    callbacks=[qwk])


# In[17]:

# train all layers
for layer in model.layers:
    layer.trainable = True
callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early, qwk]
model.compile(loss='categorical_crossentropy',
            # loss=kappa_loss,
            optimizer=Adam(lr=1e-4))
model.fit_generator(
    train_mixup,
    steps_per_epoch=np.ceil(float(len(train_x)) / float(batch_size)),
    validation_data=valid_generator,
    validation_steps=np.ceil(float(len(valid_x)) / float(batch_size)),
    epochs=epochs,
    verbose=1,
    workers=1, use_multiprocessing=False,
    callbacks=callbacks_list)


# In[18]:

submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
model.load_weights('../working/densenet_bestqwk.h5')
predicted = []


# In[19]:

# reference:https://www.kaggle.com/CVxTz/cnn-starter-nasnet-mobile-0-9709-lb 
for i, name in tqdm(enumerate(submit['id_code'])):
    path = os.path.join('../input/aptos2019-blindness-detection/test_images/', name+'.png')
    image = cv2.imread(path)
    image = cv2.resize(image, (SIZE, SIZE))
    image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)
    X = np.array((image[np.newaxis])/255)
    score_predict=((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()
    label_predict = np.argmax(score_predict)
    predicted.append(str(label_predict))


# In[20]:

submit['diagnosis'] = predicted
submit.to_csv('submission.csv', index=False)
submit.head()


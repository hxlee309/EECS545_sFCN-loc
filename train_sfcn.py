#######################################################
#Todo: Add downstream of the network and parameter sharing
#######################################################

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from PIL import Image
import glob
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle
import os
import matplotlib.pyplot as plt
import cv2

# Import necessary items from Keras
import keras
from keras import optimizers
from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, concatenate, Input, Add
from keras.layers.convolutional import Cropping2D, ZeroPadding2D
from keras.backend import spatial_2d_padding
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import backend as K
from keras.backend import eval as evaluation
from math import sqrt, pow

# Callback for checking learning updates
class AdamLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        beta_1=0.9
        beta_2=0.999
        optimizer = self.model.optimizer
        lr = evaluation(optimizer.lr) * (1. / (1. + evaluation(optimizer.decay) * evaluation(optimizer.iterations)))
        t = evaluation(optimizer.iterations) + 1
        lr_t = lr * (sqrt(1. - pow(beta_2, t)) /(1. - pow(beta_1, t)))
        print("Base lr: ", evaluation(optimizer.lr))
        print("t: ", t)
        print('\nLR: {:.6f}\n'.format(lr_t))

# Specify Which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '3'	

#Training data partitions for pickle files: 6000, 12000, 18000, 24000, 30000, 36000, 42000, 48000, 54000, 60000, 66000, 70000
partition1 = 0
partition2 = 14000
partition = partition2 - partition1 #Number of images for training

train_images = np.zeros((partition,500,500,3))
cnt_images = np.zeros((partition,500,500,3))
labels = np.zeros((partition,500,500,1))

#Load Training Images
count = 1

for img_name in sorted(glob.glob("train_image_500_jpg/train_image_*.jpg"), key = lambda name: int(name[32:-4])):
	if count >= partition1:
		print("training_image_count: ", count)
		img = Image.open(img_name)
		img = np.array(img)
		train_images[count-1-partition1,:,:,:] = img
	if count == partition2:
		break
	count += 1

#Load Contour Images
count = 1

for img_name in sorted(glob.glob("train_contour_500/contour_train_image_*.jpg"), key = lambda name: int(name[38:-4])):
	if count >= partition1:
		print("contour_image_count: ", count)
		img = Image.open(img_name)
		img = np.array(img)
		cnt_images[count-1-partition1,:,:,0] = img
		cnt_images[count-1-partition1,:,:,1] = img
		cnt_images[count-1-partition1,:,:,2] = img
	if count == partition2:
		break
	count += 1

#Load Training Labels
count = 1	

for img_name in sorted(glob.glob("train_label_500_jpg/train_label_*.jpg"), key = lambda name: int(name[32:-4])):
	if count >= partition1:
		print("training_label_count: ", count)
		img = Image.open(img_name)
		img = np.array(img)
		img = img/255.0 #Normalization
		labels[count-1-partition1,:,:,0] = img
	if count == partition2:
		break
	count += 1

# Test size may be 10% or 20%
border = int(0.9*partition)
img_train = train_images[:border]
img_val = train_images[border:]
cnt_train = cnt_images[:border]
cnt_val = cnt_images[border:]

y_train = labels[:border]
y_val = labels[border:]

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 10
epochs = 20
pool_size = (2, 2)
input_shape = img_train.shape[1:]
print(input_shape)
print("Data preprocessing is done!")

# Normalizes incoming inputs. First layer needs the input shape to work
inputs_img = Input(shape=input_shape) # 500 x 500 x 3 images
inputs_cnt = Input(shape=input_shape) # 500 x 500 x 3 contour
inputs_img_ = BatchNormalization(input_shape = input_shape)(inputs_img)
inputs_cnt_ = BatchNormalization(input_shape = input_shape)(inputs_cnt)

# Below layers were re-named for easier reading of model summary; this not necessary
# Conv & Pool 1
pad1_img = ZeroPadding2D(padding=(6, 6))(inputs_img_) # 512 x 512 x 3
pad1_cnt = ZeroPadding2D(padding=(6, 6))(inputs_cnt_) # 512 x 512 x 3

shared_conv1_1 = Conv2D(16, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv1_1')
shared_conv1_2 = Conv2D(16, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv1_2')

conv1_1_img = shared_conv1_1(pad1_img) # 512 x 512 x 16
conv1_1_cnt = shared_conv1_1(pad1_cnt) # 512 x 512 x 16
conv1_2_img = shared_conv1_2(conv1_1_img) # 512 x 512 x 16
conv1_2_cnt = shared_conv1_2(conv1_1_cnt) # 512 x 512 x 16
pool1_img = MaxPooling2D(pool_size=pool_size)(conv1_2_img) # 256 x 256 x 16
pool1_cnt = MaxPooling2D(pool_size=pool_size)(conv1_2_cnt) # 256 x 256 x 16

# Conv & Pool 2
shared_conv2_1 = Conv2D(32, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv2_1')
shared_conv2_2 = Conv2D(32, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv2_2')

conv2_1_img = shared_conv2_1(pool1_img) # 256 x 256 x 32
conv2_1_cnt = shared_conv2_1(pool1_cnt) # 256 x 256 x 32
conv2_2_img = shared_conv2_2(conv2_1_img) # 256 x 256 x 32
conv2_2_cnt = shared_conv2_2(conv2_1_cnt) # 256 x 256 x 32
pool2_img =  MaxPooling2D(pool_size=pool_size)(conv2_2_img) # 128 x 128 x 32
pool2_cnt =  MaxPooling2D(pool_size=pool_size)(conv2_2_cnt) # 128 x 128 x 32

# Conv & Pool 3
shared_conv3_1 = Conv2D(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv3_1')
shared_conv3_2 = Conv2D(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv3_2')
shared_conv3_3 = Conv2D(64, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv3_3')

conv3_1_img = shared_conv3_1(pool2_img) # 128 x 128 x 64
conv3_1_cnt = shared_conv3_1(pool2_cnt) # 128 x 128 x 64
conv3_2_img = shared_conv3_2(conv3_1_img) # 128 x 128 x 64
conv3_2_cnt = shared_conv3_2(conv3_1_cnt) # 128 x 128 x 64
conv3_3_img = shared_conv3_3(conv3_2_img) # 128 x 128 x 64
conv3_3_cnt = shared_conv3_3(conv3_2_cnt) # 128 x 128 x 64
pool3_img =  MaxPooling2D(pool_size=pool_size)(conv3_3_img) # 64 x 64 x 64
pool3_cnt =  MaxPooling2D(pool_size=pool_size)(conv3_3_cnt) # 64 x 64 x 64

# Conv & Pool 4
shared_conv4_1 = Conv2D(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv4_1')
shared_conv4_2 = Conv2D(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv4_2')
shared_conv4_3 = Conv2D(128, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv4_3')

conv4_1_img = shared_conv4_1(pool3_img) # 64 x 64 x 128
conv4_1_cnt = shared_conv4_1(pool3_cnt) # 64 x 64 x 128
conv4_2_img = shared_conv4_2(conv4_1_img) # 64 x 64 x 128
conv4_2_cnt = shared_conv4_2(conv4_1_cnt) # 64 x 64 x 128
conv4_3_img = shared_conv4_3(conv4_2_img) # 64 x 64 x 128
conv4_3_cnt = shared_conv4_3(conv4_2_cnt) # 64 x 64 x 128
pool4_img =  MaxPooling2D(pool_size=pool_size)(conv4_3_img) # 32 x 32 x 128
pool4_cnt =  MaxPooling2D(pool_size=pool_size)(conv4_3_cnt) # 32 x 32 x 128

# Conv & Pool 5
shared_conv5_1 = Conv2D(256, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv5_1')
shared_conv5_2 = Conv2D(256, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv5_2')
shared_conv5_3 = Conv2D(256, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv5_3')

conv5_1_img = shared_conv5_1(pool4_img) # 32 x 32 x 256
conv5_1_cnt = shared_conv5_1(pool4_cnt) # 32 x 32 x 256
conv5_2_img = shared_conv5_2(conv5_1_img) # 32 x 32 x 256
conv5_2_cnt = shared_conv5_2(conv5_1_cnt) # 32 x 32 x 256
conv5_3_img = shared_conv5_3(conv5_2_img) # 32 x 32 x 256
conv5_3_cnt = shared_conv5_3(conv5_2_cnt) # 32 x 32 x 256
pool5_img =  MaxPooling2D(pool_size=pool_size)(conv5_3_img) # 16 x 16 x 256
pool5_cnt =  MaxPooling2D(pool_size=pool_size)(conv5_3_cnt) # 16 x 16 x 256

#Fully Conv
shared_conv6 = Conv2D(512, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv6')
shared_conv7 = Conv2D(512, (3, 3), padding='same', strides=(1,1), activation = 'relu', name = 'Conv7')
shared_dropout_6 = Dropout(0.2)
shared_dropout_7 = Dropout(0.2)

conv6_img = shared_conv6(pool5_img) # 16 x 16 x 512
conv6_cnt = shared_conv6(pool5_cnt) # 16 x 16 x 512
conv6_img = shared_dropout_6(conv6_img)
conv6_cnt = shared_dropout_6(conv6_cnt)# 16 x 16 x 512
conv7_img = shared_conv7(conv6_img) # 16 x 16 x 512
conv7_cnt = shared_conv7(conv6_cnt) # 16 x 16 x 512
conv7_img = shared_dropout_7(conv7_img)
conv7_cnt = shared_dropout_7(conv7_cnt)# 16 x 16 x 512
# paramter sharing end !

# upper branch
concat_conv7 = concatenate([conv7_img, conv7_cnt], axis=-1) # 16 x 16 x 1024
upper_conv = Conv2D(512, (1, 1), padding='valid', strides=(1,1), activation = 'relu', name = 'upper_conv')(concat_conv7)  # 16 x 16 x 512
upper_conv_ = UpSampling2D(size=pool_size)(upper_conv)
upper_deconv = Conv2DTranspose(256, (3, 3), padding='same', strides=(1,1), activation = None, name = 'upper_deconv')(upper_conv_) # 32 x 32 x 256

# lower branch
concat_pool4 = concatenate([pool4_img, pool4_cnt], axis=-1) # 32 x 32 x 256
concat_loc = concat_pool4 # No location prior map
lower_conv = Conv2D(256, (1, 1), padding='valid', strides=(1,1), activation = 'relu', name = 'lower_conv')(concat_loc) # 32 x 32 x 256

# merge
fusion = Add()([upper_deconv, lower_conv]) # 32 x 32 x 512

#Deconv 1
deconv1_1 = Conv2DTranspose(256, (3, 3), padding='same', strides=(1,1), activation = None, name = 'Deconv1')(fusion) # 32 x 32 x 256

#Upsampling 2
up_2 = UpSampling2D(size=pool_size)(deconv1_1) # 64 x 64 x 256

#Deconv 2
deconv2_1 = Conv2DTranspose(128, (3, 3), padding='same', strides=(1,1), activation = None, name = 'Deconv2')(up_2) # 64 x 64 x 128

#Upsampling 3
up_3 = UpSampling2D(size=pool_size)(deconv2_1) # 128 x 128 x 128

#Deconv 3
deconv3_1 = Conv2DTranspose(64, (3, 3), padding='same', strides=(1,1), activation = None, name = 'Deconv3')(up_3) # 128 x 128 x 64

#Upsampling 4
up_4 = UpSampling2D(size=pool_size)(deconv3_1) # 256 x 256 x 64

#Deconv 4
deconv4_1 = Conv2DTranspose(16, (3, 3), padding='same', strides=(1,1), activation = None, name = 'Deconv4')(up_4) # 256 x 256 x 16

#Upsampling 5
up_5 = UpSampling2D(size=pool_size)(deconv4_1) # 512 x 512 x 16

#Deconv 5
deconv5_1 = Conv2DTranspose(1, (1, 1), padding='valid', strides=(1,1), activation = None, name = 'Deconv5')(up_5) # 512 x 512 x 1

#Outputs
outputs = Cropping2D(cropping=((6, 6), (6, 6)), input_shape=(512, 512, 1))(deconv5_1) # 500 x 500 x 1

### End of network ###
model = Model(inputs = [inputs_img, inputs_cnt], outputs = outputs)

# # Compiling and training the model
# model.load_weights('')

# Compiling and training the model
model.compile(optimizer='Adam', loss='mean_squared_error')

# TODO:edit inputs, img_train, cnt_train, loc_train
# check syntax
weight_dir = "sfcn_results_v5/save_weights_foo/"
model_dir = "sfcn_results_v5/save_models_foo/"
weight_path = weight_dir + 'sfcn_weight_{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5'
model_path = model_dir + 'sfcn_weight_{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5'
callbacks_weight = keras.callbacks.ModelCheckpoint(weight_path, verbose=1, save_weights_only=True, period=1)
callbacks_model = keras.callbacks.ModelCheckpoint(model_path, verbose=1, period=1)
history = model.fit(x=[img_train, cnt_train], y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([img_val, cnt_val], y_val), callbacks=[callbacks_weight, callbacks_model, AdamLearningRateTracker()])

# Freeze layers since training is done
model.trainable = False
learning_rate = 1e-10*batch_size
model.compile(optimizer='Adam', loss='mean_squared_error')

with open('sfcn_results/loss_history/sfcn_history_v5.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Show summary of model
model.summary()

print("Model and training history are saved!!!")
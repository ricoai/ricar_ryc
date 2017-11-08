"""
pilots.py

Methods to create, use, save and load pilots. Pilots 
contain the highlevel logic used to determine the angle
and throttle of a vehicle. Pilots can include one or more 
models to help direct the vehicles motion.
"""

import os
import shutil
import numpy as np
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from shutil import rmtree

import utils as utils


class KerasPilot():
 
    def load(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.model.summary()

    def train(self,
              train_gen,
              val_gen,
              saved_model_path,
              epochs=500,
              steps=10,
              train_split=0.8,
              is_early_stop=True,
              early_stop_count=5,
              is_tensorboard=False,
              is_plot_results=False,
              is_lr_decay=False):
        """
        Train the model with the given data and validation data.
        Use checkpoint to record last best model.
        Select if you want to use early stop.
        Select if you want to use tensorboard to monitor progress
        Select if you want to plot the results.  These plots will be the same as tensorboard.

        :param train_gen: generator that yields an array of images (Random 90% of given data in Tub)
        :param val_gen: Generator that yields an array of image  (Random 10% of given data in Tub)
        :param saved_model_path: Path to save the model
        :param epochs: Number of times to train on the data.
        :param steps: How many steps per epoch.
        :param is_early_stop: Stop early if the training does not improve.
        :param is_tensorboard: Generate a tensorboard to monitor progress of the neural network.
        :param is_plot_results: Display matplotlib plots.  Same plots as tensorboard
        """

        if is_plot_results or is_tensorboard:
            folder = 'plots'
            # Create folder if it does not exist
            if not os.path.isdir(folder):
                os.makedirs(folder)

            # Create datetime folder
            dt = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = os.path.join(folder, str(dt))

            # Create folder if it does not exist
            if not os.path.isdir(folder):
                os.makedirs(folder)

            # Create Tensorboard folder
            tb_path = os.path.join(folder, 'tb')
            if not os.path.isdir(tb_path):
                os.makedirs(tb_path)
            else:
                rmtree(tb_path)
                os.makedirs(tb_path)

            # Save config
            if os.path.exists('config.py'):
                shutil.copyfile('config.py', os.path.join(folder, 'config.py'))

        # checkpoint to save model after each epoch
        save_best = keras.callbacks.ModelCheckpoint(saved_model_path, 
                                                    monitor='val_loss', 
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    mode='min')

        # At the very least, we need to train and save the model with checkpoints
        callbacks_list = [save_best]

        # Stop training if the validation error stops improving.
        if is_early_stop:
            print("Using Early Stop")
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       min_delta=.0005,
                                                       patience=early_stop_count,
                                                       verbose=1,
                                                       mode='auto')

            callbacks_list.append(early_stop)

        # Add Tensorboard callback
        # This will create a Graph directory
        # Run tensorboard --logdir path_to/Graph
        if is_tensorboard:
            print("Using Tensorboard at " + tb_path + '\nRun tensorboard --logdir=' + tb_path + ' to view the tensorboard.')
            tb_callback = keras.callbacks.TensorBoard(log_dir=tb_path,
                                                      histogram_freq=0,      # Histogram frequency - Does NOT work, val_gen is a generator and not data
                                                      write_grads=True,      # Write Histogram, histogram_freq must be greater than 0
                                                      write_graph=True,      # Write graph to describe network
                                                      write_images=True,
                                                      embeddings_freq=1,
                                                      embeddings_layer_names=['dense_1', 'dense_2', 'dense_3'],
                                                      batch_size=5
                                                      )
            callbacks_list.append(tb_callback)

        if is_lr_decay:
            def scheduler(epoch):
                if epoch % 2 == 0 and epoch != 0:
                    lr = keras.get_value(self.model.optimizer.lr)
                    keras.set_value(self.model.optimizer.lr, lr * .9)
                    print("lr changed to {}".format(lr * .9))
                return keras.get_value(self.model.optimizer.lr)

            lr_decay = keras.callbacks.LearningRateScheduler(scheduler)
            callbacks_list.append(lr_decay)

        # Start to train the model
        hist = self.model.fit_generator(
                        train_gen, 
                        steps_per_epoch=steps, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_data=val_gen,
                        callbacks=callbacks_list, 
                        validation_steps=steps*(1.0 - train_split))

        if is_plot_results:
            # Save config
            if os.path.exists('config.py'):
                shutil.copyfile('config.py', os.path.join(folder, 'config.py'))

            # Copy the model
            if os.path.exists(saved_model_path):
                shutil.copyfile(saved_model_path, os.path.join(folder, os.path.basename(saved_model_path)+'.h5py'))

            # Copy the tensorboard
            #if os.path.exists('./Graph'):
            #    shutil.copytree('./Graph', os.path.join(folder, 'tensorboard'))

            # list all data in history
            print(hist.history.keys())
            # summarize history for loss
            plt.figure('Loss')
            plt.plot(hist.history['loss'])
            plt.plot(hist.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(os.path.join(folder, 'loss.png'))
            #plt.show()
            # summarize history for Angle out loss
            plt.figure('Angle Out Loss')
            plt.plot(hist.history['angle_out_loss'])
            plt.plot(hist.history['val_angle_out_loss'])
            plt.title('Angle Out Loss')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(os.path.join(folder, 'angle_out_loss.png'))
            #plt.show()
            # summarize history for Throttle out loss
            plt.figure('Throttle Out Loss')
            plt.plot(hist.history['throttle_out_loss'])
            plt.plot(hist.history['val_throttle_out_loss'])
            plt.title('Throttle Out Loss')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(os.path.join(folder, 'throttle_out_loss.png'))
            #plt.show()
            # summarize history for Angle out Accuracy
            plt.figure('Angle Out Accuracy')
            plt.plot(hist.history['angle_out_acc'])
            plt.plot(hist.history['val_angle_out_acc'])
            plt.title('Angle Out Accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(os.path.join(folder, 'angle_out_acc.png'))
            #plt.show()
            # summarize history for Throttle out Accuracy
            plt.figure('Throttle Out Accuracy')
            plt.plot(hist.history['throttle_out_acc'])
            plt.plot(hist.history['val_throttle_out_acc'])
            plt.title('Throttle Out Accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(os.path.join(folder, 'throttle_out_acc.png'))
            plt.show()

            print('Result Plots at {}'.format(folder))

        return hist


class KerasRicar(KerasPilot):
    """
    Use ReLU activation.
    """
    def __init__(self,
                 model=None,
                 dropout_1=0.1,
                 dropout_2=0.1,
                 optimizer='rmsprop',
                 learning_rate=1e-5,
                 loss_weight_angle=0.9,
                 loss_weight_throttle=0.001,
                 is_categorical=True,
                 *args, **kwargs):
        super(KerasRicar, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = ricar_cnn(dropout_1=dropout_1,
                                   dropout_2=dropout_2,
                                   optimizer=optimizer,
                                   learning_rate=learning_rate,
                                   loss_weight_angle=loss_weight_angle,
                                   loss_weight_throttle=loss_weight_throttle,
                                   is_categorical=is_categorical)
        
    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, throttle = self.model.predict(img_arr)
        #angle_certainty = max(angle_binned[0])
        angle_unbinned = utils.linear_unbin(angle_binned)
        return angle_unbinned, throttle[0][0]


def ricar_cnn(dropout_1=0.1,
                        dropout_2=0.1,
                        optimizer='rmsprop',
                        learning_rate=1.0e-5,
                        loss_weight_angle=0.9,
                        loss_weight_throttle=0.001,
                        is_categorical=True):
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense
    from keras.optimizers import Adam

    def image_normm(val):
        return val / 127.5 - 1.0

    img_in = Input(shape=(120, 160, 3), name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    #x = Lambda(lambda image_norm, input_shape=(120, 160, 3))
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)                                    # Classify the data into 100 features, make all negatives 0
    x = Dropout(dropout_1)(x)                                               # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)                                     # Classify the data into 50 features, make all negatives 0
    x = Dropout(dropout_2)(x)                                               # Randomly drop out 10% of the neurons (Prevent overfitting)

    if is_categorical:
        # categorical output of the angle
        angle_out = Dense(15, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0
    else:
        angle_out = Dense(1, activation='linear', name='angle_out')(x)

    # continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)      # Reduce to 1 number, Positive number only
    
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    if optimizer == 'adam':
        model.compile(optimizer=Adam(lr=learning_rate),
                      loss={'angle_out': 'mean_squared_error',
                            'throttle_out': 'mean_squared_error'},
                      loss_weights={'angle_out': loss_weight_angle, 'throttle_out': loss_weight_throttle},
                      metrics=['accuracy'])
    else:
        model.compile(optimizer='rmsprop',
                      loss={'angle_out': 'categorical_crossentropy',
                            'throttle_out': 'mean_absolute_error'},
                      loss_weights={'angle_out': loss_weight_angle, 'throttle_out': loss_weight_throttle},
                      metrics=['accuracy'])

    return model


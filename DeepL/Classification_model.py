from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.callbacks import CSVLogger
import os

from Dataset_generator import *

class Classification_model:
    def __init__(self,designator):

        self.num_classes = 10
        self.picturesPFeature_train = 150
        self.picturesPFeature_test = 30
        self.shape = (int(64), int(64), 1)
        # train options
        self.batch_size = 200
        self.epochs = 400
        self.rot_range = 20
        self.width_range = 0.2
        self.height_range = 0.2
        self.zoom = 0.3
        self.activation_conv = "relu"
        self.activation_dense = "softmax"

        self.kernel_size_first = 3
        self.kernel_size_second = 3

        self.designator = designator
        self.create_model1()

    #TRASH
    def create_model1(self):
        
        self.x = Input(self.shape)

        self.y = Conv2D(filters=5,
                   kernel_size=(23, 23),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.x)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=25,
                   kernel_size=(5, 5),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.y)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Flatten()(self.y)
        self.y = Dense(self.num_classes, activation=self.activation_dense, )(self.y)
        self.model = Model(self.x, self.y)

        print("[MESSAGE] Model is defined.")

        # print model summary
        self.model.summary()

        # compile the model aganist the categorical cross entropy loss and use SGD optimizer
        self.model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["mse", "accuracy"])
        print ("[MESSAGE] Model is compiled.")


    #Worse than model4
    def create_model2(self):
        
        self.x = Input(self.shape)

        self.y = Conv2D(filters=5,
                   kernel_size=(21, 21),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.x)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=25,
                   kernel_size=(7, 7),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.y)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=47,
                   kernel_size=(3, 3),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.y)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Flatten()(self.y)
        self.y = Dense(self.num_classes, activation=self.activation_dense, )(self.y)
        self.model = Model(self.x, self.y)

        print("[MESSAGE] Model is defined.")

        # print model summary
        self.model.summary()

        # compile the model aganist the categorical cross entropy loss and use SGD optimizer
        self.model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["mse", "accuracy"])
        print ("[MESSAGE] Model is compiled.")

#THAT ONE IS TRASH

    '''
    def create_model3(self):
        
        self.x = Input(self.shape)

        self.y = Conv2D(filters=10,
                   kernel_size=(31, 31),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.x)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=25,
                   kernel_size=(19, 19),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.y)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=47,
                   kernel_size=(5, 5),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.y)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Flatten()(self.y)
        self.y = Dense(self.num_classes, activation=self.activation_dense, )(self.y)
        self.model = Model(self.x, self.y)

        print("[MESSAGE] Model is defined.")

        # print model summary
        self.model.summary()

        # compile the model aganist the categorical cross entropy loss and use SGD optimizer
        self.model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["mse", "accuracy"])
        print ("[MESSAGE] Model is compiled.")
    '''

    #Similar to model 4. Wrong kernel size. Need to be ascending
    def create_model3(self):
        self.x = Input(self.shape)

        self.y = Conv2D(filters=11,
                   kernel_size=(15, 15),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.x)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=19,
                   kernel_size=(5, 5),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.y)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=47,
                   kernel_size=(3, 3),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.y)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Flatten()(self.y)
        self.y = Dense(self.num_classes, activation=self.activation_dense, )(self.y)
        self.model = Model(self.x, self.y)

        print("[MESSAGE] Model is defined.")

        # print model summary
        self.model.summary()

        # compile the model aganist the categorical cross entropy loss and use SGD optimizer
        self.model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["mse", "accuracy"])
        print ("[MESSAGE] Model is compiled.")


    # Similar to model 3. Wrong kernel size. Need to be ascending
    def create_model4(self):
        
        self.x = Input(self.shape)

        self.y = Conv2D(filters=5,
                   kernel_size=(15, 15),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.x)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=25,
                   kernel_size=(9, 9),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.y)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=47,
                   kernel_size=(3, 3),
                   padding="same",
                   activation=self.activation_conv,
                   )(self.y)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Flatten()(self.y)
        self.y = Dense(self.num_classes, activation=self.activation_dense, )(self.y)
        self.model = Model(self.x, self.y)

        print("[MESSAGE] Model is defined.")

        # print model summary
        self.model.summary()

        # compile the model aganist the categorical cross entropy loss and use SGD optimizer
        self.model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["mse", "accuracy"])
        print ("[MESSAGE] Model is compiled.")

    #same as mode 4, but with ascending kernel size
    def create_model5(self):

        self.x = Input(self.shape)

        self.y = Conv2D(filters=5,
                        kernel_size=(3, 3),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.x)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=25,
                        kernel_size=(9, 9),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=47,
                        kernel_size=(15, 15),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Flatten()(self.y)
        self.y = Dense(self.num_classes, activation=self.activation_dense, )(self.y)
        self.model = Model(self.x, self.y)

        print("[MESSAGE] Model is defined.")

        # print model summary
        self.model.summary()

        # compile the model aganist the categorical cross entropy loss and use SGD optimizer
        self.model.compile(loss="categorical_crossentropy",
                           optimizer="adam",
                           metrics=["mse", "accuracy"])
        print ("[MESSAGE] Model is compiled.")

    #Same as model5 but with avgpooling
    def create_model6(self):

        self.x = Input(self.shape)

        self.y = Conv2D(filters=5,
                        kernel_size=(3, 3),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.x)
        self.y = AveragePooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=25,
                        kernel_size=(9, 9),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = AveragePooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=47,
                        kernel_size=(15, 15),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = AveragePooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Flatten()(self.y)
        self.y = Dense(self.num_classes, activation=self.activation_dense, )(self.y)
        self.model = Model(self.x, self.y)

        print("[MESSAGE] Model is defined.")

        # print model summary
        self.model.summary()

        # compile the model aganist the categorical cross entropy loss and use SGD optimizer
        self.model.compile(loss="categorical_crossentropy",
                           optimizer="adam",
                           metrics=["mse", "accuracy"])
        print ("[MESSAGE] Model is compiled.")

    #With strided convolution rather than Maxpooling, but descending kernel size
    def create_model7(self):

        self.x = Input(self.shape)

        self.y = Conv2D(filters=5,
                        kernel_size=(15, 15),
                        strides=(2, 2),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.x)
        self.y = Conv2D(filters=25,
                        kernel_size=(9, 9),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Conv2D(filters=25,
                        kernel_size=(5, 5),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Conv2D(filters=47,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Conv2D(filters=13,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Flatten()(self.y)
        self.y = Dense(self.num_classes, activation=self.activation_dense, )(self.y)
        self.model = Model(self.x, self.y)

        # With strided convolution, larger network than model7 and ascending kernel size
    def create_model8(self):
        self.x = Input(self.shape)

        self.y = Conv2D(filters=13,
                        kernel_size=(3, 3),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.x)
        self.y = Conv2D(filters=17,
                        kernel_size=(5, 5),
                        strides=(2, 2),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Conv2D(filters=25,
                        kernel_size=(7, 7),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Conv2D(filters=47,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Conv2D(filters=13,
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Flatten()(self.y)
        self.y = Dense(self.num_classes, activation=self.activation_dense, )(self.y)
        self.model = Model(self.x, self.y)

        print("[MESSAGE] Model is defined.")

        # print model summary
        self.model.summary()

        # compile the model aganist the categorical cross entropy loss and use SGD optimizer
        self.model.compile(loss="categorical_crossentropy",
                           optimizer="adam",
                           metrics=["mse", "accuracy"])
        print ("[MESSAGE] Model is compiled.")


    def create_model9(self):
        self.x = Input(self.shape)

        self.y = Conv2D(filters=13,
                        kernel_size=(3, 3),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.x)
        self.y = Conv2D(filters=17,
                        kernel_size=(5, 5),
                        strides=(2, 2),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Conv2D(filters=25,
                        kernel_size=(7, 7),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Conv2D(filters=35,
                        kernel_size=(9, 9),
                        strides=(2, 2),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Conv2D(filters=27,
                        kernel_size=(11, 11),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Conv2D(filters=13,
                        kernel_size=(13, 13),
                        strides=(2, 2),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Conv2D(filters=15,
                        kernel_size=(15, 15),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Flatten()(self.y)
        self.y = Dense(self.num_classes, activation=self.activation_dense, )(self.y)
        self.model = Model(self.x, self.y)

        print("[MESSAGE] Model is defined.")

        # print model summary
        self.model.summary()

        # compile the model aganist the categorical cross entropy loss and use SGD optimizer
        self.model.compile(loss="categorical_crossentropy",
                           optimizer="adam",
                           metrics=["mse", "accuracy"])
        print ("[MESSAGE] Model is compiled.")

    def create_model10(self):

        self.x = Input(self.shape)

        self.y = Conv2D(filters=13,
                        kernel_size=(9, 9),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.x)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=25,
                        kernel_size=(13, 13),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = MaxPooling2D((2, 2), strides=(2, 2))(self.y)
        self.y = Conv2D(filters=35,
                        kernel_size=(15, 15),
                        padding="same",
                        activation=self.activation_conv,
                        )(self.y)
        self.y = Flatten()(self.y)
        self.y = Dense(self.num_classes, activation=self.activation_dense, )(self.y)
        self.model = Model(self.x, self.y)

        print("[MESSAGE] Model is defined.")

        # print model summary
        self.model.summary()

        # compile the model aganist the categorical cross entropy loss and use SGD optimizer
        self.model.compile(loss="categorical_crossentropy",
                           optimizer="adam",
                           metrics=["mse", "accuracy"])
        print ("[MESSAGE] Model is compiled.")

    def train_model(self):
        # load training dataset
        # shape = (width,height,channels) i.e shape = (224,256,3) for a 224x256 (widthxheight) with 3 RGB channels
        (train_x, train_y) = load_train_set(self.num_classes, self.picturesPFeature_train, self.shape)
        (valid_x, valid_y) = load_valid_set(self.num_classes, self.picturesPFeature_test, self.shape)
        train_x = train_x[..., np.newaxis]
        valid_x = valid_x[..., np.newaxis]

        print("[MESSAGE] Loaded testing dataset.")

        # converting the input class labels to categorical labels for training
        train_Y = to_categorical(train_y, num_classes=self.num_classes)
        valid_Y = to_categorical(valid_y, num_classes=self.num_classes)
        print("[MESSAGE] Converted labels to categorical labels.")

        datagen = image.ImageDataGenerator(
	        samplewise_center = True,
	        samplewise_std_normalization = True,
            rotation_range = self.rot_range,
            width_shift_range = self.width_range,
            height_shift_range = self.height_range,
            zoom_range = [1-self.zoom, 1+self.zoom],
            horizontal_flip = True,
            vertical_flip = True)

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(train_x)

        csvlogger = CSVLogger(self.designator + ".csv")
        # fits the model on batches with real-time data augmentation:
        self.model.fit_generator(datagen.flow(train_x, train_Y, batch_size=self.batch_size),
                            steps_per_epoch=len(train_x) / self.batch_size, epochs=self.epochs,
                            callbacks =[csvlogger],
			    validation_data=datagen.flow(valid_x, valid_Y, batch_size=self.batch_size))

        print("[MESSAGE] Model is trained.")

        # save the trained model
        self.model.save(self.designator + ".hdf5")
        print("[MESSAGE] Model is saved.")

    def load_model(self):
        if os.path.isfile(self.designator + ".hdf5"):
            # exists
            self.model.load_weights(self.designator + ".hdf5")
            print("[MESSAGE] Loaded model weights")
        else:
            # doesn't exist
            print("[MESSAGE] No safed model weights found")

    def get_model(self):
        return self.model

    def get_number_of_classes(self):
        output = self.num_classes
        return output

    def get_input_shape(self):
        return self.shape
    
    def set_parameter(self,s,b,e,r,w,h,z, model_choice):
        self.shape = (int(s),int(s),1)
        self.batch_size = b
        self.epochs = e
        self.rot_range = r
        self.width_range = w
        self.height_range = h
        self.zoom = z
        if model_choice == 1:
            self.create_model1()
        elif model_choice == 2:
            self.create_model2()
        elif model_choice == 3:
            self.create_model3()
        elif model_choice == 4:
            self.create_model4()
        elif model_choice == 5:
            self.create_model5()
        elif model_choice == 6:
            self.create_model6()
        elif model_choice == 7:
            self.create_model7()
        elif model_choice == 8:
            self.create_model8()
        elif model_choice == 9:
            self.create_model9()
        elif model_choice == 10:
            self.create_model10()












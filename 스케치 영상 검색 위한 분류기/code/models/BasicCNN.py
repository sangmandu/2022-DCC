from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D


class Model(Model):
    def __init__(self, image_shape):
        super().__init__()
        self.image_shape = image_shape

        self.conv1 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=image_shape)
        self.conv2 = Conv2D(64, (2, 2), activation='relu', padding='same')

        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        self.dp1 = Dropout(0.25)
        self.dp2 = Dropout(0.5)

        self.d1 = Dense(1000, activation='relu')
        self.d2 = Dense(20, activation='softmax')
        self.flatten = Flatten()

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.dp1(x)
        x = self.flatten(x)
        x = self.d1(x)

        x = self.dp2(x)
        x = self.d2(x)

        return x

    def summary(self):
        inputs = Input(self.image_shape)
        Model(inputs, self.call(inputs)).summary()

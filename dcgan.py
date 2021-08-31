from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Reshape, Flatten, Activation, Conv2D
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, UpSampling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np


class DCGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.z_dim = 100

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()
        self.combined = self.build_combined()
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=optimizer)

    # 生成者モデルの構築
    def build_generator(self):
        noise_shape = (self.z_dim,)
        model = Sequential()
        model.add(Dense(1024, input_shape=noise_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(1, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        model.summary()
        return model

    # 識別者モデルの構築
    def build_discriminator(self):
        img_shape = (self.img_rows, self.img_cols, self.channels)
        model = Sequential()
        model.add(Conv2D(64, (5, 5), strides=(1, 1),
                         padding='same', input_shape=img_shape))
        model.add(LeakyReLU(0.2))
        model.add(Conv2D(128, (5, 5), strides=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model

    def build_combined(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model

    def train(self, epochs, batch_size=128, save_interval=50):

        (X_train, _), (_, _) = mnist.load_data()

        # normalize
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        num_batches = int(X_train.shape[0] / half_batch)
        print('Number of batches:', num_batches)

        for epoch in range(epochs):
            for iteration in range(num_batches):
                noise = np.random.normal(0, 1, (half_batch, self.z_dim))
                gen_imgs = self.generator.predict(noise)

                idx = np.random.randint(0, X_train.shape[0], half_batch)
                imgs = X_train[idx]

                d_loss_real = self.discriminator.train_on_batch(
                    imgs, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(
                    gen_imgs, np.zeros((half_batch, 1)))

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                noise = np.random.normal(0, 1, (batch_size, self.z_dim))

                valid_y = np.array([1] * batch_size)

                g_loss = self.combined.train_on_batch(noise, valid_y)

                acc = 100*d_loss[1]
                print(
                    f'epoch:{epoch}, iter:{iteration},  [D loss: {d_loss[0]}, acc.: {acc}] [G loss: {g_loss}]')

            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f'images/mnist_{epoch}.png')
        plt.close()


if __name__ == '__main__':
    gan = DCGAN()
    gan.train(epochs=30000, batch_size=32, save_interval=1)

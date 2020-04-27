import tensorflow as tf
import numpy as np
from Ac_Block_utils import AC_Block

config = {'BN_eps':1e-5, 
          'BN_momentum':0.1, 
          'optimizer_momentum': 0.9, 
          'epoch': 100
         }

def create_cifar_quick():
    img_input = tf.keras.Input(shape=(32, 32, 3), name='original_img')

    AC_out = AC_Block(img_input, name = 'conv_1', filters = 32, kernel_size = 5, padding = 'same', momentum=config['BN_momentum'], epsilon=config['BN_eps'])
    AC_out = tf.keras.layers.ReLU()(AC_out)
    AC_out = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(AC_out)
    AC_out = AC_Block(AC_out, name = 'conv_2', filters = 32, kernel_size = 5, padding = 'same', momentum=config['BN_momentum'], epsilon=config['BN_eps'])
    AC_out = tf.keras.layers.ReLU()(AC_out)
    AC_out = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2)(AC_out)
    AC_out = AC_Block(AC_out, name = 'conv_3', filters = 64, kernel_size = 5, padding = 'same', momentum=config['BN_momentum'], epsilon=config['BN_eps'])
    AC_out = tf.keras.layers.ReLU()(AC_out)
    AC_out = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2)(AC_out)

    out = tf.keras.layers.Flatten()(AC_out)
    out = tf.keras.layers.Dense(units=64)(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Dense(units=10)(out)

    model = tf.keras.Model(img_input, out, name='cifar_quick')
    return model

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = create_cifar_quick()
    model.summary()
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[10,20,30], values=[0.1, 0.01, 0.001, 0.0001])
    highest_accuracy = 0
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = (x_train - 0.5) / 0.5
    x_eval = x_eval.astype('float32') / 255.
    x_eval = (x_eval - 0.5) / 0.5
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_eval = tf.keras.utils.to_categorical(y_eval, 10)
    for i in range(1, config['epoch']+1):
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule(i), 
                                                            momentum=config['optimizer_momentum']),
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        print("Epoch: {} lr: {}".format(i, model.optimizer.lr))
        model.fit(x_train, y_train, batch_size=64, epochs=1)
        if i%2 == 0:        
            test_scores = model.evaluate(x_eval, y_eval, verbose=2)
            print('Test loss:', test_scores[0])
            print('Test accuracy:', test_scores[1])
            if test_scores[1] > highest_accuracy:
                model.save('AC_cfqk.h5') 
                highest_accuracy = test_scores[1]
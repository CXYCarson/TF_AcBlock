import tensorflow as tf
from Ac_Block_utils import kernel_fusion, deploy

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.cifar10.load_data()
    x_eval = x_eval.astype('float32') / 255.
    x_eval = (x_eval - 0.5) / 0.5
    y_eval = tf.keras.utils.to_categorical(y_eval, 10)

    model = tf.keras.models.load_model('AC_cfqk.h5')
    test_scores = model.evaluate(x_eval, y_eval, verbose=2)
    print('Evaluating original model-------------------')
    print('Original Test Loss:', test_scores[0])
    print('Original Test Accuracy:', test_scores[1])
    print('Evaluation complete-------------------')

    AC_Block_names = ['conv_1', 'conv_2', 'conv_3']
    ckpt_file = 'AC_cfqk.h5'
    deploy(ckpt_file, AC_Block_names)
    print('Fusion complete-------------------')
    model = tf.keras.models.load_model('fused_model.h5', compile = False)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.summary()
    test_scores = model.evaluate(x_eval, y_eval, verbose=2)
    print('Evaluating fused model------------------')
    print('Fused Test Loss:', test_scores[0])
    print('Fused Test Accuracy:', test_scores[1])
    print('Evaluation complete-------------------')
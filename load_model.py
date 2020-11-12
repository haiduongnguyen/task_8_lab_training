import tensorflow as tf


model1 = tf.keras.models.load_model('resource/seed3.h5')
print('done')

def predict(img):
    z = model1.predict(img)
    if z > 0.5:
        return 1
    else:
        return 0

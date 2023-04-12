from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input

weight_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

img_size = 32

base_model = keras.applications.vgg16.VGG16(
    weights='imagenet',
    input_shape=(img_size,img_size,3),
    include_top=False
)

vgg_layers = []
for layer in base_model.layers[:4]:
    layer.trainable = False
    vgg_layers.append(layer)

vgg_layers

# Composed generator
def build_gen_untrainable():
    """
    here an input shape (0,100) is passed thourh vgg
    this part of the net remains untrainable
    """
    model = keras.models.Sequential(name='gen_untrainbale')

    model.add(keras.layers.Dense(img_size*img_size*3, input_dim=100))
    model.add(keras.layers.ReLU())
    
    model.add(keras.layers.Reshape([img_size,img_size,3]))

    for layer in vgg_layers:
        model.add(layer)

    return model

gen_in = build_gen_untrainable()
for layer in gen_in.layers:
    layer.trainable = False

def build_gen_trainable():
    """
    here an input from the vgg part is passed
    the trainbale layers
    """
    model = keras.models.Sequential(name='gen_trainbale')
    model.add(keras.layers.InputLayer(input_shape = (16, 16, 64)))
    model.add(keras.layers.Dense(16*16*64))
    model.add(keras.layers.Dense(8*8*32))
    model.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2),
                                           padding='same',kernel_initializer=weight_init,
                                           activation = keras.layers.ReLU()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(3, (4, 4), padding="same", activation="tanh"))
    return model

gen_out = build_gen_trainable()

def build_dis_untrainable():
    """
    here an input shape (0,100) is passed thourh vgg
    this part of the net remains untrainable
    """
    model = keras.models.Sequential(name = 'dis_untrainbale')
    model.add(keras.layers.InputLayer(input_shape = (img_size, img_size, 3)))
    for layer in vgg_layers:
        model.add(layer)
    
    return model

dis_in = build_dis_untrainable()
for layer in dis_in.layers:
    layer.trainable = False

def build_dis_untrainable():
    """
    here an input from the vgg part is passed
    the trainbale layers
    """
    model = keras.models.Sequential(name='trainbale')
    model.add(keras.layers.InputLayer(input_shape = (16,16,64)))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.Conv2D(128, (4,4), padding='same', strides=(2,2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model

dis_out = build_dis_untrainable()
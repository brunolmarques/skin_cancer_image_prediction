### Algorithm that recognizes skin cancer by analysing photos

# Import statements
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import GlobalAveragePooling2D
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint
import numpy as np

img_width, img_height = 256, 256
train_data_dir = "data/train"
validation_data_dir = "data/valid"
test_data_dir = "data/test"
batch_size = 10
epochs = 10
seed = 7

# Define augumentation and normalization instances
train_valid_datagen = ImageDataGenerator(featurewise_center=True,
                                         rescale=1./255,
                                         zoom_range=0.2,
                                         horizontal_flip=False,
                                         vertical_flip=False,
                                         fill_mode='reflect',
                                         rotation_range=270,
                                         width_shift_range=0.15,
                                         height_shift_range=0.15)

test_datagen = ImageDataGenerator(rescale=1./255)

# Fit the imageset with the instances
train_generator = train_valid_datagen.flow_from_directory(
                                            train_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            seed = seed,
                                            class_mode='categorical')

valid_generator = train_valid_datagen.flow_from_directory(
                                            validation_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            seed = seed,
                                            class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
                                    test_data_dir,
                                    target_size=(img_width, img_height),
                                    batch_size=batch_size,
                                    seed = seed,
                                    class_mode='categorical')


# Model architecture
base_model = VGG16(weights='imagenet', include_top=False,
                    input_shape = (img_width, img_height, 3))

# Freeze the original model so it keeps the weights
for layer in base_model.layers:
    layer.trainable = False

# Create numpy zeros arrays to hold the training and valid features
train_features = np.zeros(shape=(2000, 7, 7, 512))
train_labels = np.zeros(shape=(2000,3))
valid_features = np.zeroes(shape=(150, 7, 7, 512))
valid_labels = np.zerps(shape=(150, 3))

# Use base model to predict the output of the imageset, output will be a
# tensor of dimension ( , 7, 7, 512)
i=0
for train_inputs, train_labels_batch in train_generator:
    train_features_batch = base_model.predict(train_inputs)
    train_features[i * batch_size : (i+1) * batch_size] = train_features_batch
    train_labels[i * batch_size : (i+1) * batch_size] = train_labels_batch
    i += 1
    if i * batch_size >= 2000:
        break
train_features = np.reshape(train_features, (2000, 7, 7, 512))

j=0
for valid_inputs, valid_labels_batch in valid_generator:
    valid_features_batch = base_model.predict(valid_inputs)
    valid_features[i * batch_size : (i+1) * batch_size] = valid_features_batch
    valid_labels[i * batch_size : (i+1) * batch_size] = valid_labels_batch
    i += 1
    if i * batch_size >= 2000:
        break
valid_features = np.reshape(valid_features, (2000, 7, 7, 512))

# Adding top model classifier
top_model = Sequential()
top_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 512)))
top_model.add(Dropout(rate=0.5, seed=seed))
top_model.add(Dense(1024)))
top_model.add(Activation('relu'))
top_model.add(Dropout(rate=0.5, seed=seed))
top_model.add(Dense(3))
top_model.add(Activation('softmax'))

# Compile top model
top_model.compile(loss="categorical_crossentropy", optimizer='RMSprop',
                    metrics=["accuracy"])

# Fit the top model with the generated data
history = top_model.fit(train_features,
                        train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(valid_features, valid_labels))

# Final model
final_model = Model(input=base_model.input, output=predictions)

# Compile Model
final_model.compile(loss="categorical_crossentropy", optimizer='RMSprop',
                    metrics=["accuracy"])

# Create a checkpoint to save best Model
checkpointer = ModelCheckpoint(filepath='dermo_vgg16.weights.best.hdf5',
                            verbose=1, save_best_only=True)

# Train the Model
final_model.fit_generator(train_generator,
            epochs=epochs,
            validation_data=valid_generator,
            verbose=1,
            callbacks=[checkpointer])

# Load the weights that yielded the best validation accuracy
final_model.load_weights('dermo_vgg16.weights.best.hdf5')

# Fine tune Model by unfreezing base model and freezing the added top layers
# Requires GPU (CUDA) to process
'''
for layer in final_model.layers[-4:]:
    layer.trainable = False
for layer in final_model.layers[:-4]:
    layer.trainable = True

# Compile Model
final_model.compile(loss="categorical_crossentropy",
                    optimizer=SGD(lr=0.0001, momentum=0.9),
                    metrics=["accuracy"])

# Create a checkpoint to save best Model
checkpointers = ModelCheckpoint(filepath='dermo2_vgg16.weights.best.hdf5',
                            verbose=1, save_best_only=True)

# Train the fine tunning
final_model.fit_generator(train_generator,
                            epochs=epochs,
                            validation_data=valid_generator,
                            verbose=1,
                            callbacks=[checkpointer2])
'''

# Save final model load_weights
final_model.save_weights("trained_dermo.h5")

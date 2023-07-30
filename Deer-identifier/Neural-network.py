import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Set image dimensions
img_width, img_height = 224, 224
batch_size = 32

# Load images using ImageDataGenerator
train_data_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_data_gen.flow_from_directory(
    '/Users/macbookair/Desktop/Combined-data-set',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Load MobileNetV2 model (pre-trained on ImageNet)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the base model's layers
for layer in base_model.layers:
    layer.trainable = False

# Build the model
x = base_model.output
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=x)

# Compile the model
opt = Adam(lr=1e-4)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 20
model.fit(
    train_data,
    epochs=epochs,
    steps_per_epoch=train_data.samples // batch_size
)

# Save the model
model.save('/Users/macbookair/Desktop/Combined-data-set/deer_identifier_model.h5')

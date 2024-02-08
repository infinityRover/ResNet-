from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.resnet import ResNet101
#from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np


def model_trainer(epochs = None, **kwargs):
    train_data_dir = r"/workspace/data_DIR/gaussian_filtered_images/gaussian_filtered_images_processed/train"
    test_data_dir = r"/workspace/data_DIR/gaussian_filtered_images/gaussian_filtered_images_processed/test"
    valid_data_dir = r"/workspace/data_DIR/gaussian_filtered_images/gaussian_filtered_images_processed/val"

    train_generator, test_generator, valid_generator = preprocess(
        train_data_dir=train_data_dir,
        test_data_dir=test_data_dir,
        valid_data_dir=valid_data_dir)

    x, y = test_generator.next()
    print(x.shape)
    print(f'Number of classification classes : {train_generator.num_classes}')

    base_model = ResNet101(include_top=False,
                          weights="imagenet")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(train_generator.num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint_callback = ModelCheckpoint('/workspace/data_DIR/models/resnet/best_model.h5',
                                          monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    model.fit(
        train_generator,
        epochs=30,
        # steps_per_epoch=steps_per_epoch,
        validation_data=valid_generator,  # Replace with your validation dataset
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    '''
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, (x_batch, y_batch) in enumerate(train_generator):
            try:
                # Train on the current batch
                loss, accuracy = model.train_on_batch(x_batch, y_batch)
                print(f"Batch {i + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            except Exception as e:
                # Handle the error (e.g., skip the problematic image)
                print(f"Error in batch {i + 1}: {str(e)}")

                problem_image_bytes = x_batch[0]  # Assuming the problematic image is the first in the batch
                problem_image = Image.open(io.BytesIO(problem_image_bytes))
                problem_image.save(f'problematic_image_{epoch + 1}_{i + 1}.png')

                continue
    '''
    return model
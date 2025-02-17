import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, models
import json
import pandas as pd
from tensorflow import keras

data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomRotation(0.2),
    layers.RandomWidth(0.2),
    layers.RandomHeight(0.2),
    layers.RandomFlip('horizontal')
])


# Diccionario para guardar resultados
results = {
    "step": [],
    "train_accuracy": [],
    "train_loss": [],
    "val_accuracy": [],
    "val_loss": [],
    "test_accuracy": []  # Añadimos test_accuracy
}

def log_results(step, history, test_acc=None):
    results["step"].append(step)
    results["train_accuracy"].append(history.history['accuracy'][-1])
    results["train_loss"].append(history.history['loss'][-1])
    results["val_accuracy"].append(history.history['val_accuracy'][-1])
    results["val_loss"].append(history.history['val_loss'][-1])
    if test_acc is not None:
        results["test_accuracy"].append(test_acc)
    else:
        results["test_accuracy"].append(None)  # Para mantener las listas del mismo tamaño

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])
# Configuración básica
batch_size = 32  # Número de imágenes por lote
img_height = 180  # Altura de las imágenes redimensionadas
img_width = 180   # Ancho de las imágenes redimensionadas

# Ruta a los datasets
train_dir = "../data/train_data"
test_dir = "../data/test_data"

# Cargar los datos
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True  # Mezcla los datos
)

class_names = train_ds.class_names

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))


test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    batch_size=batch_size,
    image_size=(img_height, img_width)
)

AUTOTUNE = tf.data.AUTOTUNE

# Optimización del rendimiento
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

'''
# Definir la arquitectura de la CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.53),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.53),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])'''

def Model_Convolutional(shapes):
    inputs = keras.Input(shape=shapes)

    x = layers.Conv2D(filters=64, kernel_size=3, padding = 'same')(inputs)
    
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.MaxPool2D(pool_size = 2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, padding = 'same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, padding = 'same', activation='leaky_relu')(x)
    
    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.MaxPool2D(pool_size = 2)(x)

    x = layers.Conv2D(filters=256, kernel_size=3, padding = 'same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=256, kernel_size=3, padding = 'same', activation='leaky_relu')(x)

    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.MaxPool2D(pool_size = 2)(x)

    x = layers.Conv2D(filters=512, kernel_size=3, padding = 'same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=512, kernel_size=3, padding = 'same', activation='leaky_relu')(x)

    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(100, activation='leaky_relu')(x)

    outputs = layers.Dense(len(class_names), activation='softmax')(x)

    model = keras.Model(inputs = inputs, outputs = outputs)
    
    return model

model = Model_Convolutional(shapes = (180,180,3))

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

epochs = 10
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs
)

# Guardar el modelo entrenado
step = 5  # Cambia este número según el paso
model.save(f"../models/model_step{step}.h5")
print(f"Modelo guardado como: models/model_step{step}.h5")


# Evaluar el modelo en el conjunto de test
test_loss, test_acc = model.evaluate(test_ds)
print(f"Precisión en el conjunto de test: {test_acc:.2f}")

# Guardar en results
log_results(f"Step {step}", history)

# Convertir a DataFrame para mejor visualización
results_df = pd.DataFrame(results)
print(results_df)

plt.plot(history.history['accuracy'], label = 'Train')
plt.plot(history.history['val_accuracy'], label = 'Val')
plt.legend()
plt.show()

# Guardar el historial de resultados en un archivo JSON
results_path = f"../logs/results_model_step{step}.json"
os.makedirs("logs", exist_ok=True)  # Asegura que el directorio logs exista
with open(results_path, "w") as f:
    json.dump(results, f)
print(f"Resultados guardados como: {results_path}")


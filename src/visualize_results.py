import json
import os
import matplotlib.pyplot as plt

# Ruta donde están los archivos JSON
json_dir = "../logs"  # Cambia esto por la ruta a tus archivos JSON

# Inicializar listas para almacenar los datos
steps = []
train_accuracy = []
train_loss = []
val_accuracy = []
val_loss = []

# Cargar los archivos JSON y extraer los datos
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        with open(os.path.join(json_dir, filename), "r") as f:
            data = json.load(f)
            steps.append(data["step"])
            train_accuracy.append(data["train_accuracy"])
            train_loss.append(data["train_loss"])
            val_accuracy.append(data["val_accuracy"])
            val_loss.append(data["val_loss"])

# Crear un gráfico de la evolución de las métricas
plt.figure(figsize=(12, 6))

# Gráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(steps, train_accuracy, label="Train Accuracy", marker='o', linestyle='-', color='b')
plt.plot(steps, val_accuracy, label="Val Accuracy", marker='o', linestyle='--', color='g')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Evolución de la Precisión')
plt.legend()

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(steps, train_loss, label="Train Loss", marker='o', linestyle='-', color='r')
plt.plot(steps, val_loss, label="Val Loss", marker='o', linestyle='--', color='y')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Evolución de la Pérdida')
plt.legend()

# Mostrar los gráficos
plt.tight_layout()
plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = r"C:\Users\hii\Desktop\skin cancer ai\data\ISIC_2019_Training_Input"
GROUNDTRUTH_CSV = r"c:\Users\hii\Desktop\skin cancer ai\data\ISIC_2019_Training_GroundTruth.csv"

# Load your CSV
df = pd.read_csv(GROUNDTRUTH_CSV)

# ✅ Keep only numeric label columns (ignore 'image' and any text)
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# ✅ Exclude 'UNK' if it exists
label_cols = [col for col in numeric_cols if col != "UNK"]

# ✅ Create label index and map to string labels
df["label_idx"] = df[label_cols].values.argmax(axis=1)
label_map = {i: label for i, label in enumerate(label_cols)}
df["label"] = df["label_idx"].map(label_map)

# ✅ Add .jpg to image names if missing
df["image"] = df["image"].astype(str).apply(
    lambda x: x + ".jpg" if not x.lower().endswith(".jpg") else x
)

print("✅ Label columns used:", label_cols)
print("✅ Label mapping:", label_map)


# ✅ Add .jpg to all image names
df["image"] = df["image"].astype(str).apply(lambda x: x + ".jpg" if not x.lower().endswith(".jpg") else x)

print("Label mapping:", label_map)
print("Sample filenames:", df["image"].head())


# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Image size and batch
IMG_SIZE = 128
BATCH_SIZE = 32

# Data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
)

# Train generator
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=DATA_DIR,
    x_col="image",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Validation generator
val_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=DATA_DIR,
    x_col="image",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Number of classes (from your CSV columns)
NUM_CLASSES = len(label_cols)

# Base model (transfer learning)
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers initially
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
#EPOCHS = 10
#history = model.fit(
   # train_generator,
   # validation_data=val_generator,
   # epochs=EPOCHS
#)

# Optionally unfreeze some layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

#history_finetune = model.fit(
  #  train_generator,
   # validation_data=val_generator,
    #epochs=5
#)

# Save trained model
import os
os.makedirs("model", exist_ok=True)

model.save("model/skin_cancer_model.keras")
print("✅ Model saved successfully!")

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
import os

save_path = "model/skin_cancer_model.h5"
model.save(save_path)

if os.path.exists(save_path):
    size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ Model saved successfully at '{save_path}' ({size:.2f} MB)")
else:
    print("❌ Model save failed — file not found!")

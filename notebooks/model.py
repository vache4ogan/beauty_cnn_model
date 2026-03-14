import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==================== ПАРАМЕТРЫ ====================
IMG_SIZE = 128
BATCH_SIZE = 32
RANDOM_SEED = 42
EPOCHS = 50
NUM_CLASSES = 5

# Устанавливаем seed
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==================== ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ ====================
def load_data(data_path='D:/Projects/Crokodilo/Sorted_data', img_size=IMG_SIZE):
    """
    Загружает изображения из папок 1-5
    Возвращает X (изображения) и y (метки 0-4)
    """
    images = []
    labels = []
    data_path = Path(data_path)
    
    for rating in range(1, 6):
        folder = data_path / str('rating_'+str(rating))
        if not folder.exists():
            print(f"Папка {folder} не найдена")
            continue
            
        for img_path in folder.glob('*.*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (img_size, img_size))
                    images.append(img)
                    labels.append(rating - 1)
    
    return np.array(images), np.array(labels)

# ==================== АУГМЕНТАЦИЯ ДАННЫХ ====================
def create_augmentation():
    """
    Создает слой аугментации данных
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),  # Случайный переворот
        tf.keras.layers.RandomRotation(0.1),       # Поворот до 10%
        tf.keras.layers.RandomZoom(0.1),           # Зум до 10%
        tf.keras.layers.RandomContrast(0.1),       # Контраст
        tf.keras.layers.RandomTranslation(0.1, 0.1), # Сдвиг
    ])

# ==================== НОРМАЛЬНАЯ CNN АРХИТЕКТУРА ====================
def create_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Нормальная CNN архитектура для классификации изображений
    """
    # Создаем слой аугментации
    data_augmentation = create_augmentation()
    
    model = keras.models.Sequential([
        # Аугментация применяется только при обучении
        tf.keras.layers.Input(shape=input_shape),
        data_augmentation,
        
        # Первый сверточный блок
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Второй сверточный блок
        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Третий сверточный блок
        keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Четвертый сверточный блок
        keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Выходная часть
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# ==================== ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ ====================
def train_beauty_model(data_path='D:/Projects/Crokodilo/Sorted_data'):
    """
    Главная функция: загружает данные и обучает модель
    """
    print("="*50)
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("="*50)
    
    # Загружаем данные
    X, y = load_data(data_path)
    print(f"Загружено {len(X)} изображений")
    
    # Нормализация
    X = X / 255.0
    
    # Разделение на train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    print("\n" + "="*50)
    print("2. СОЗДАНИЕ МОДЕЛИ")
    print("="*50)
    
    # Создаем модель
    model = create_model()
    model.summary()
    
    # Компиляция
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    
    print("\n" + "="*50)
    print("3. ОБУЧЕНИЕ С АУГМЕНТАЦИЕЙ")
    print("="*50)
    
    # Callbacks
    checkpoint = keras.callbacks.ModelCheckpoint(
        "best_model.keras", 
        save_best_only=True, 
        monitor='val_accuracy',
        mode='max'
    )
    
    early_stop = keras.callbacks.EarlyStopping(
        patience=15, 
        restore_best_weights=True,
        monitor='val_loss'
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Обучение
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    print("\n" + "="*50)
    print("4. ОЦЕНКА")
    print("="*50)
    
    # Оценка на тесте
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Сохраняем финальную модель
    model.save("D:/Projects/Crokodilo/models/beauty_model_final.keras")
    print("Модель сохранена как 'beauty_model_final.keras'")
    
    return model, history

# ==================== ФУНКЦИЯ ПРЕДСКАЗАНИЯ ====================
def predict_score(image_path, model):
    """
    Предсказывает оценку для одного изображения
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    pred = model.predict(img, verbose=0)[0]
    score = np.argmax(pred) + 1
    confidence = pred[score - 1]
    
    return score, confidence, pred

# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    # Обучаем модель
    model, history = train_beauty_model()
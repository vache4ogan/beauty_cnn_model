# 👸 Beauty CNN Model

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
  ![Keras](https://img.shields.io/badge/Keras-2.13%2B-red)
  ![License](https://img.shields.io/badge/License-MIT-green)
  
  **Модель для оценки привлекательности на фотографиях (оценки от 1 до 5)**  
  *на основе сверточных нейросетей (CNN)*
  
</div>

---

## 📋 О проекте

Данный проект использует **сверточную нейросеть с 4 сверточными слоями** для классификации фотографий по шкале красоты от 1 до 5. Модель обучена на датасете SCUT-FBP5500 и показывает хорошие результаты в задаче субъективной оценки внешности.

**Особенности модели:**
- 🏗️ 4 сверточных блока с нарастающим количеством фильтров (32→64→128→256)
- 🔄 Аугментация данных (повороты, отражения, зум, контраст) для улучшения обобщения
- 📊 Batch Normalization после каждого сверточного слоя
- 🎯 Dropout для борьбы с переобучением
- ⚡ GlobalAveragePooling2D вместо Flatten для уменьшения параметров

---

## 📊 Датасет

**SCUT-FBP5500 Database-Release**

Это открытый репозиторий, содержащий:
- 🖼️ 5500 фотографий лиц
- 📝 Усредненные оценки красоты от 1 до 5 от нескольких экспертов
- 🌍 Разнообразные лица (разные возрасты, расы, ракурсы)

Датасет доступен по [ссылке](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)

---

## 🚀 Быстрый старт

### 1. Клонирование репозитория
```bash
git clone https://github.com/vache4ogan/beauty_cnn_model.git
cd beauty_cnn_model
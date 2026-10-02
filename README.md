# DeepVision-CIFAR10: CIFAR-10 Evrişimsel Sinir Ağları (CNN) Modeli

Keras ve TensorFlow kullanarak CIFAR-10 veri seti üzerinde görüntü sınıflandırması gerçekleştiren derin öğrenme (CNN) projesi.

## 📌 Proje İçeriği ve Özellikler

- **Veri Kümesi:** CIFAR-10 (32x32 boyutlarında, 10 farklı sınıfa ait 60.000 renkli görsel: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*)
- **Veri Ön İşleme (Preprocessing):**
  - Piksel normalizasyonu (`[0, 1]` aralığı)
  - One-hot encoding (`to_categorical`)
  - Örnek veri görselleştirmesi (Matplotlib)
- **Veri Çoğaltma (Data Augmentation):**
  - Rastgele döndürme (`rotation_range=20`)
  - Yatay ve dikey kaydırma (`width_shift_range=0.2`, `height_shift_range=0.2`)
  - Kesme ve yakınlaştırma (`shear_range=0.2`, `zoom_range=0.2`)
  - Yatay çevirme (`horizontal_flip=True`)
- **Model Mimarisi:**
  - **Blok 1:** Conv2D (32 filtre, 3x3) + ReLU + Conv2D (32 filtre, 3x3) + ReLU + MaxPooling2D (2x2) + Dropout (%25)
  - **Blok 2:** Conv2D (64 filtre, 3x3) + ReLU + Conv2D (64 filtre, 3x3) + ReLU + MaxPooling2D (2x2) + Dropout (%25)
  - **Sınıflandırma (Classifier):** Flatten + Dense (512 nöron, ReLU) + Dropout (%50) + Dense (10 nöron, Softmax)
- **Optimizasyon ve Kayıp Fonksiyonu:** RMSprop (lr=0.001, decay=1e-6) ve Categorical Crossentropy.

## 🚀 Kurulum ve Çalıştırma

### 1. Sanal Ortamı Oluşturma ve Aktifleştirme

```bash
# Sanal ortam oluşturma
python -m venv .venv

# Sanal ortamı aktifleştirme (Windows)
.venv\Scripts\activate

# Sanal ortamı aktifleştirme (Linux/macOS)
source .venv/bin/activate
```

### 2. Bağımlılıkların Yüklenmesi

```bash
pip install -r requirements.txt
```

### 3. Modeli Eğitme ve Çalıştırma

```bash
python main.py
```

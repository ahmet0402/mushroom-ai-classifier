# 🍄 Mushroom AI Classifier

AI destekli mantar sınıflandırma web uygulaması. Derin öğrenme algoritması ile mantar türlerini (yenilebilir/zehirli) anlık olarak analiz eder.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Teknolojiler](#teknolojiler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Model Bilgileri](#model-bilgileri)
- [Lisans](#lisans)

## ✨ Özellikler

- 🖼️ **Tekli & Çoklu Tahmin**: Tek seferde 1-10 arası mantar fotoğrafı analiz edin
- 🎯 **Yüksek Doğruluk**: %87 doğruluk oranı ile güvenilir tahminler
- 📊 **Detaylı İstatistikler**: Model performans metrikleri ve grafikler
- 🎨 **Modern UI/UX**: Gradient tasarım, animasyonlar ve responsive arayüz
- 🖱️ **Drag & Drop**: Dosyaları sürükle-bırak ile yükleme
- 📈 **Canlı Grafikler**: Chart.js ile interaktif veri görselleştirme
- 🔍 **Grafik Önizleme**: Model performans grafiklerine tıklayarak büyütme
- 💾 **Tahmin Geçmişi**: Yapılan tüm tahminleri kaydetme ve görüntüleme

## 🛠️ Teknolojiler

### Backend
- **Flask 3.0** - Web framework
- **TensorFlow/Keras 2.20** - Deep learning
- **NumPy 1.26** - Numerical computing
- **Pillow 10.1** - Image processing

### Frontend
- **Bootstrap 5.3** - UI framework
- **Chart.js 4.4** - Data visualization
- **Font Awesome 6.4** - Icons
- **Jinja2** - Template engine

### AI/ML
- **CNN (Convolutional Neural Network)** - Image classification
- **Softmax Activation** - Multi-class output
- **Input Size**: 128x128 RGB
- **Classes**: 2 (Edible, Poisonous)

## 📦 Kurulum

### Gereksinimler
- Python 3.11 veya üzeri
- pip (Python package manager)
- Git (opsiyonel)

### Adım 1: Projeyi İndirin
```bash
git clone https://github.com/KULLANICI_ADIN/mushroom-ai.git
cd mushroom-ai
```

### Adım 2: Sanal Ortam Oluşturun (Önerilen)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### Adım 3: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### Adım 4: Uygulamayı Başlatın
```bash
python app.py
```

Tarayıcınız otomatik olarak `http://127.0.0.1:5000` adresini açacaktır.

## 🚀 Kullanım

### Tekli Tahmin
1. Ana sayfada **"Çoklu Tahmin"** menüsüne tıklayın
2. Tek bir mantar fotoğrafı yükleyin (sürükle-bırak veya dosya seç)
3. **"Tümünü Analiz Et"** butonuna tıklayın
4. Sonuçları ve güven skorlarını görüntüleyin

### Çoklu Tahmin
1. **"Çoklu Tahmin"** sayfasına gidin
2. Birden fazla mantar fotoğrafı seçin (max 10)
3. Tüm fotoğraflar aynı anda analiz edilir
4. Grid görünümünde sonuçları inceleyin

### İstatistikler
1. **"Analiz"** sayfasına gidin
2. Model performans metriklerini görün:
   - Doğruluk (Accuracy)
   - Precision
   - Recall
   - F1-Score
3. Eğitim grafiklerini inceleyin
4. Son tahminler tablosunu kontrol edin

## 📁 Proje Yapısı

```
mushroom_web/
│
├── app.py                      # Ana Flask uygulaması
├── mushroom_model.keras        # Eğitilmiş CNN modeli
├── requirements.txt            # Python bağımlılıkları
├── README.md                   # Proje dokümantasyonu
│
├── static/                     # Statik dosyalar
│   ├── css/
│   │   └── style.css          # Ana CSS dosyası
│   ├── plots/                 # Model grafikleri
│   │   ├── train_accuracy.png
│   │   ├── train_loss.png
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   └── classification_report.txt
│   └── uploads/               # Yüklenen görseller
│
└── templates/                 # HTML şablonları
    ├── index.html            # Ana sayfa (sonuç gösterimi)
    ├── batch.html            # Çoklu tahmin sayfası
    └── analytics.html        # İstatistik sayfası
```

## 🧠 Model Bilgileri

### Performans Metrikleri
```
                  precision    recall  f1-score   support

      edible       0.90      0.85      0.87       150
   poisonous       0.84      0.89      0.86       132

    accuracy                           0.87       282
   macro avg       0.87      0.87      0.87       282
weighted avg       0.87      0.87      0.87       282
```

### Model Özellikleri
- **Mimari**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Girdi Boyutu**: 128x128x3 (RGB)
- **Çıktı**: 2 sınıf (Softmax)
- **Eğitim Veri Seti**: 282 örnek
- **Test Doğruluğu**: %87

### Tahmin Süreci
1. Görsel 128x128 boyutuna yeniden boyutlandırılır
2. Piksel değerleri normalize edilir (0-1 arası)
3. Model forward pass yapar
4. Softmax aktivasyonu ile olasılıklar hesaplanır
5. En yüksek olasılıklı sınıf seçilir


## ⚙️ Yapılandırma

### Debug Modunu Kapatma (Production)
`app.py` dosyasında:
```python
if __name__ == "__main__":
    app.run(debug=False)  # debug=True yerine False yapın
```

### Port Değiştirme
```python
app.run(debug=True, port=8080)  # Varsayılan: 5000
```

### Maksimum Dosya Sayısını Değiştirme
`templates/batch.html` içinde:
```javascript
if (files.length > 20) {  // 10 yerine 20
    alert('Maksimum 20 dosya seçebilirsiniz!');
}
```

## 🌐 Deployment

### Render.com (Ücretsiz)
1. GitHub'a projeyi yükleyin
2. [Render.com](https://render.com) hesabı oluşturun
3. New Web Service → GitHub repo'yu seçin
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `gunicorn app:app`

### PythonAnywhere
1. [PythonAnywhere.com](https://pythonanywhere.com) hesabı oluşturun
2. Dosyaları yükleyin
3. Web tab → Flask uygulaması ekleyin
4. WSGI dosyasını yapılandırın

### Railway / Fly.io
GitHub ile otomatik deploy seçenekleri mevcuttur.

## 🐛 Bilinen Sorunlar

- **AVIF Format**: Pillow AVIF formatını desteklemiyor. JPEG, PNG, JPG kullanın.
- **Büyük Model**: `mushroom_model.keras` dosyası büyük olabilir (GitHub LFS gerektirebilir)
- **Browser Compatibility**: Modern tarayıcılar önerilir (Chrome, Firefox, Edge)

## 🤝 Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📝 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 👤 Geliştirici

**Ahmet**

- GitHub: [@KULLANICI_ADIN](https://github.com/KULLANICI_ADIN)
- E-mail: your.email@example.com

## 🙏 Teşekkürler

- TensorFlow & Keras ekibine
- Flask topluluğuna
- Bootstrap ve Chart.js'e
- Mantar veri seti sağlayıcılarına

## ⚠️ Yasal Uyarı

Bu uygulama **sadece eğitim ve araştırma amaçlıdır**. Gerçek hayatta mantar tüketimi için profesyonel bir uzman veya mikolog danışmanız gerekmektedir. Tahminler %100 doğru olmayabilir ve yanlış tanımlama ciddi sağlık sorunlarına neden olabilir.

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

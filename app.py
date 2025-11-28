import os
import uuid
import numpy as np
import webbrowser
from threading import Timer
from datetime import datetime
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

# -------------------------
# 1) Flask uygulamasını oluştur
# -------------------------
app = Flask(__name__)

# Yüklemeler için klasör
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Tahmin geçmişini saklamak için basit liste
prediction_history = []

# -------------------------
# 2) Modeli belleğe yükle
# -------------------------
MODEL_PATH = "mushroom_model.keras"
model = load_model(MODEL_PATH)

# Bizim sınıf isimlerimiz (edible klasör index 0, poisonous klasör index 1 idi)
CLASS_NAMES = ["edible", "poisonous"]

IMG_SIZE = (128, 128)  # modeli eğitirken kullandığımız boyut

# -------------------------
# 3) Ana sayfa: form göster
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Formdan gelen dosyayı al
        if "file" not in request.files:
            return render_template("index.html", error="Dosya bulunamadı!")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="Lütfen bir mantar resmi seçin.")

        # Dosyayı sunucuya kaydet
        ext = os.path.splitext(file.filename)[1].lower()
        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # Tahmin yap
        pred_label, edible_prob, poisonous_prob = predict_image(save_path)

        # Web'de kullanmak için yol
        image_url = save_path.replace("\\", "/")

        # Geçmişe ekle
        add_to_history(image_url, pred_label, edible_prob, poisonous_prob)

        return render_template(
            "index.html",
            image_url=image_url,
            pred_label=pred_label,
            edible_prob=edible_prob,
            poisonous_prob=poisonous_prob
        )

    # GET ise sadece boş sayfa göster
    return render_template("index.html")


# -------------------------
# 3.1) Çoklu fotoğraf yükleme sayfası
# -------------------------
@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "POST":
        files = request.files.getlist("files")
        
        if not files or files[0].filename == "":
            return render_template("batch.html", error="Lütfen en az bir dosya seçin.")
        
        if len(files) > 10:
            return render_template("batch.html", error="Maksimum 10 dosya yükleyebilirsiniz.")
        
        results = []
        for file in files:
            if file.filename == "":
                continue
                
            # Dosyayı kaydet
            ext = os.path.splitext(file.filename)[1].lower()
            filename = f"{uuid.uuid4().hex}{ext}"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            
            # Tahmin yap
            pred_label, edible_prob, poisonous_prob = predict_image(save_path)
            
            # Web yolu
            image_url = save_path.replace("\\", "/")
            
            # Geçmişe ekle
            add_to_history(image_url, pred_label, edible_prob, poisonous_prob)
            
            results.append({
                "image_url": image_url,
                "pred_label": pred_label,
                "edible_prob": edible_prob,
                "poisonous_prob": poisonous_prob
            })
        
        return render_template("batch.html", results=results)
    
    return render_template("batch.html")


# -------------------------
# 3.2) Analiz sayfası
# -------------------------
@app.route("/analytics")
def analytics():
    stats = calculate_statistics()
    recent = prediction_history[-10:] if len(prediction_history) >= 10 else prediction_history
    recent.reverse()  # En yeniden eskiye
    
    # Model performans metriklerini oku
    model_metrics = read_model_metrics()
    
    return render_template("analytics.html", stats=stats, recent_predictions=recent, metrics=model_metrics)


# -------------------------
# 4) Görüntüyü modele verip tahmin yapan fonksiyon
# -------------------------
def predict_image(img_path):
    # Görseli yükle
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0   # eğitimdeki gibi normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Modelden ileri besleme
    preds = model.predict(img_array)[0]  # shape: (2,)
    edible_p = float(preds[0])
    poisonous_p = float(preds[1])

    # En büyük olasılığı sınıf etiketi olarak seç
    if edible_p >= poisonous_p:
        label = "edible"
    else:
        label = "poisonous"

    # Yüzdelik hale getir
    edible_percent = round(edible_p * 100, 2)
    poisonous_percent = round(poisonous_p * 100, 2)

    return label, edible_percent, poisonous_percent


# -------------------------
# 5) Geçmişe ekleme fonksiyonu
# -------------------------
def add_to_history(image_url, pred_label, edible_prob, poisonous_prob):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_history.append({
        "image_url": image_url,
        "pred_label": pred_label,
        "edible_prob": edible_prob,
        "poisonous_prob": poisonous_prob,
        "timestamp": timestamp
    })


# -------------------------
# 6) İstatistik hesaplama fonksiyonu
# -------------------------
def calculate_statistics():
    if not prediction_history:
        return {
            "total_predictions": 0,
            "edible_count": 0,
            "poisonous_count": 0,
            "avg_confidence": 0,
            "confidence_distribution": [0, 0, 0, 0, 0],
            "trend_labels": [],
            "trend_edible": [],
            "trend_poisonous": []
        }
    
    total = len(prediction_history)
    edible_count = sum(1 for p in prediction_history if p["pred_label"] == "edible")
    poisonous_count = total - edible_count
    
    # Ortalama güven skoru (en yüksek olasılık)
    confidences = []
    for p in prediction_history:
        max_conf = max(p["edible_prob"], p["poisonous_prob"])
        confidences.append(max_conf)
    avg_confidence = round(sum(confidences) / len(confidences), 2)
    
    # Güven dağılımı
    conf_dist = [0, 0, 0, 0, 0]
    for conf in confidences:
        if conf <= 20:
            conf_dist[0] += 1
        elif conf <= 40:
            conf_dist[1] += 1
        elif conf <= 60:
            conf_dist[2] += 1
        elif conf <= 80:
            conf_dist[3] += 1
        else:
            conf_dist[4] += 1
    
    # Son 10 tahmin trendi
    recent = prediction_history[-10:] if len(prediction_history) >= 10 else prediction_history
    trend_labels = [f"#{i+1}" for i in range(len(recent))]
    trend_edible = [p["edible_prob"] for p in recent]
    trend_poisonous = [p["poisonous_prob"] for p in recent]
    
    return {
        "total_predictions": total,
        "edible_count": edible_count,
        "poisonous_count": poisonous_count,
        "avg_confidence": avg_confidence,
        "confidence_distribution": conf_dist,
        "trend_labels": trend_labels,
        "trend_edible": trend_edible,
        "trend_poisonous": trend_poisonous
    }


# -------------------------
# 7) Model metriklerini oku
# -------------------------
def read_model_metrics():
    """classification_report.txt dosyasından model performans metriklerini okur"""
    metrics = {
        "accuracy": 87,
        "precision": 87,
        "recall": 87,
        "f1_score": 87
    }
    
    try:
        report_path = os.path.join("static", "plots", "classification_report.txt")
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Accuracy satırını bul
        for line in content.split('\n'):
            if "accuracy" in line.lower() and "0." in line:
                # Format: "    accuracy                           0.87       282"
                parts = [p for p in line.split() if p]  # Boşlukları temizle
                for part in parts:
                    try:
                        val = float(part)
                        if 0 < val <= 1:  # 0-1 arası değer accuracy olmalı
                            metrics["accuracy"] = round(val * 100)
                            break
                    except:
                        continue
            
            # Weighted avg satırını bul
            if "weighted avg" in line.lower():
                # Format: "weighted avg       0.87      0.87      0.87       282"
                parts = [p for p in line.split() if p]
                # İlk "weighted" ve "avg" sonrası 3 değer: precision, recall, f1-score
                try:
                    precision_val = float(parts[2])
                    recall_val = float(parts[3])
                    f1_val = float(parts[4])
                    
                    metrics["precision"] = round(precision_val * 100)
                    metrics["recall"] = round(recall_val * 100)
                    metrics["f1_score"] = round(f1_val * 100)
                except Exception as e:
                    print(f"Weighted avg parse hatası: {e}")
                break
    except Exception as e:
        print(f"Metrik okuma hatası: {e}")
        # Hata durumunda varsayılan değerler kullanılır
    
    return metrics


# -------------------------
# 8) Uygulamayı başlat
# -------------------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Flask başladıktan 1 saniye sonra tarayıcıyı aç
    # use_reloader=False ile sadece bir kere açılır
    import os
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        Timer(1, open_browser).start()
    # debug=True → hata olursa ekranda görebilirsin
    app.run(debug=True)

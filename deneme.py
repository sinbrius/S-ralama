from ultralytics import YOLO
import cv2
import os

# 1. Kendi eğittiğin, etiketleri bilen beyni yükle
# 'yolov8n.pt' kullanma! 'kitap_beyni.pt' kullan.
model = YOLO('kitap-beyni.pt') 

# 2. Test edeceğin raf fotoğrafının adını yaz
test_resmi = 'raf.png' 

if os.path.exists(test_resmi):
    print("🎯 Model etiketleri arıyor...")
    
    # 3. Tahmin yap ve işaretlenmiş resmi kaydet!
    # 'save=True' parametresi işaretlenmiş resmi kaydeder.
    # 'conf=0.2' modelin %20 emin olduğu her şeyi gösterir.
    results = model.predict(source=test_resmi, save=True, conf=0.2, imgsz=640)
    
    # Çıktı klasörünü terminale yazdıralım
    for r in results:
        print(f"👉 İşaretlenmiş resim buraya kaydedildi: {r.save_dir}")
        print(f"✅ Toplam {len(r.boxes)} etiket bulundu.")

else:
    print(f"HATA: '{test_resmi}' dosyası bulunamadı, dosya adını kontrol et.")
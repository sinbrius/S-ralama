from ultralytics import YOLO
import cv2
import os

# 1. Eğittiğin modeli yükle
model = YOLO('kitap-beyni.pt')

# 2. Test resmini oku
img_path = 'raf1.png'
img = cv2.imread(img_path)

# 3. Tahmin yap
results = model.predict(source=img_path, conf=0.25)

# Kırpılan resimler için klasör oluştur (isteğe bağlı)
if not os.path.exists('kirpilmis_etiketler'):
    os.makedirs('kirpilmis_etiketler')

for r in results:
    # Kutuları soldan sağa x koordinatına göre sıralıyoruz (ÖNEMLİ!)
    # Bu sayede raftaki sırayı korumuş oluruz
    boxes = sorted(r.boxes, key=lambda x: x.xyxy[0][0])
    
    print(f"Toplam {len(boxes)} kitap etiketi tespit edildi. Kırpma başlıyor...")
    
    for i, box in enumerate(boxes):
        # Koordinatları al (x1, y1: sol üst | x2, y2: sağ alt)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Orijinal resimden o bölgeyi kes (Crop)
        # OpenCV'de [y_aralığı, x_aralığı] şeklinde kesilir
        etiket_resmi = img[y1:y2, x1:x2]
        
        # Kırpılan her etiketi kaydet (Kontrol etmek için)
        dosya_adi = f'kirpilmis_etiketler/etiket_{i}.jpg'
        cv2.imwrite(dosya_adi, etiket_resmi)
        
        print(f"Kitap {i} başarıyla kesildi: {dosya_adi}")

print("✅ Tüm etiketler raf sırasına göre ayrıştırıldı!")
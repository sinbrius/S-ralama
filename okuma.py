import easyocr
import os
import cv2
import re
import numpy as np

#Tanımlamalar
kitap_klasoru='kirpilmis_etiketler'
reader=easyocr.Reader(['en'], gpu=False)

#türkçe karakter bulunduran dosyaları okuyabilmesi için
def turkce_oku(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8),cv2.IMREAD_COLOR)

#easyocr kolay anlasın diye resim önişlemesi
def preprocessin_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resmi 2 kat büyüt (Küçük yazılar için kritik)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Kontrastı artır (Eşiği otomatik belirleyen Otsu yöntemi)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def akilli_etiket_duzelt(raw_text):
    """Senin geliştirdiğin düzeltme mantığı: Karıştırılan karakterleri onar."""
    # OCR hatalarını kütüphane kod standartlarına göre düzelt
    duzeltmeler = { 'Z': '2', 'O': '0', 'B': '8', 'I': '1'}
    
    clean = raw_text.upper()
    for yanlis, dogru in duzeltmeler.items():
        clean = clean.replace(yanlis, dogru)
    
    # Sadece Harf, Rakam ve Nokta kalsın, gürültüyü sil
    clean = re.sub(r'[^A-Z0-9.\s]', '', clean)
    return clean.strip()

# 2. İşleme Başla
# Dosya adındaki rakama göre sıralıyoruz ki raf sırası bozulmasın (etiket_0, etiket_1...)
kitap_listesi = sorted([f for f in os.listdir(kitap_klasoru) if f.endswith('.jpg')], 
                       key=lambda x: int(re.findall(r'\d+', x)[0]))

sonuclar = []

print("🎯 Kütüphane etiketleri analiz ediliyor...")

for dosya in kitap_listesi:
    yol = os.path.join(kitap_klasoru, dosya)
    img = turkce_oku(yol)
    
    if img is not None:
        temiz_img = preprocessin_img(img)
        # allowlist ile OCR'ı sadece kütüphane karakterlerine zorla
        result = reader.readtext(temiz_img, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ. ')
        
        if result:
            ham_metin = " ".join(result)
            duzeltilmis_metin = akilli_etiket_duzelt(ham_metin)
            
            sonuclar.append({"dosya": dosya, "kod": duzeltilmis_metin})
            print(f"✅ {dosya}: {duzeltilmis_metin}")
        else:
            print(f"⚠️ {dosya}: Hiçbir şey okunamadı.")

# 3. Sıralama Analizi (Mühendislik Kararı)
print("\n--- 🧐 Hatalı Kitap Tespiti ---")
for i in range(len(sonuclar) - 1):
    if sonuclar[i]['kod'] > sonuclar[i+1]['kod']:
        print(f"🚨 DİKKAT: '{sonuclar[i+1]['dosya']}' yanlış yerde duruyor!")
        print(f"👉 Olması gereken yer: {sonuclar[i]['kod']} öncesi değil, sonrası.")
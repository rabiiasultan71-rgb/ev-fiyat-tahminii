import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans


dosya_adi = 'data.csv'

if os.path.exists(dosya_adi):
    df = pd.read_csv(dosya_adi)
    print("✅ Veri seti başarıyla yüklendi!")
    
    
    def oda_temizle(x):
        try:
            if '+' in str(x):
                p = str(x).split('+')
                return float(p[0]) + float(p[1])
            return float(x)
        except: return 0

    df['Oda_S'] = df['Oda_Sayisi'].apply(oda_temizle)

    
    X = df[['Metrekare', 'Oda_S']]
    y = df['fiyat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    print(f"📊 Denetimli Öğrenme Başarı Skoru (R2): {reg_model.score(X_test, y_test):.2f}")

    
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    df['Grup'] = kmeans.fit_predict(df[['Metrekare', 'fiyat']])
    print("🤖 Denetimsiz Öğrenme: Evler 3 gruba ayrıldı.")
    

    print("\n--- Analiz Sonuçlarından Örnek ---")
    print(df[['Metrekare', 'Oda_Sayisi', 'fiyat', 'Grup']].head())

else:
    print(f"❌ HATA: '{dosya_adi}' dosyası bulunamadı! odev.py ile aynı klasörde olmalı.")

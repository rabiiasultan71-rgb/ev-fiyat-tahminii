from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)


df = pd.read_csv('data.csv')

def oda_temizle(x):
    try:
        if '+' in str(x):
            p = str(x).split('+')
            return float(p[0]) + float(p[1])
        return float(x)
    except: return 0

df['Oda_S'] = df['Oda_Sayisi'].apply(oda_temizle)

le_sehir = LabelEncoder()
df['Sehir_Kod'] = le_sehir.fit_transform(df['il'])

le_ilce = LabelEncoder()
df['Ilce_Kod'] = le_ilce.fit_transform(df['Ilce'])

le_mahalle = LabelEncoder()
df['Mahalle_Kod'] = le_mahalle.fit_transform(df['Mahalle'])

le_satici = LabelEncoder()
df['Satici_Kod'] = le_satici.fit_transform(df['satici_tip'])


X = df[['Metrekare', 'Oda_S', 'Sehir_Kod', 'Ilce_Kod', 'Mahalle_Kod', 'Satici_Kod']]
y = df['fiyat']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, y)

sehirler = sorted(df['il'].unique())
ilceler = sorted(df['Ilce'].unique())
mahalleler = sorted(df['Mahalle'].unique())
satici_tipleri = sorted(df['satici_tip'].unique())

@app.route('/')
def index():
    return render_template('index.html', sehirler=sehirler, ilceler=ilceler, 
                           mahalleler=mahalleler, satici_tipleri=satici_tipleri)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        mkare = float(request.form['metrekare'])
        oda = float(request.form['oda'])
        
        
        sehir_k = le_sehir.transform([request.form['sehir']])[0]
        ilce_k = le_ilce.transform([request.form['ilce']])[0]
        mahalle_k = le_mahalle.transform([request.form['mahalle']])[0]
        satici_k = le_satici.transform([request.form['satici']])[0]
        
        input_data = scaler.transform([[mkare, oda, sehir_k, ilce_k, mahalle_k, satici_k]])
        sonuc = model.predict(input_data)[0]
        
        return render_template('index.html', tahmin_sonucu=f"{sonuc:,.0f}", 
                               sehirler=sehirler, ilceler=ilceler, 
                               mahalleler=mahalleler, satici_tipleri=satici_tipleri)
    except Exception as e:
        return f"Sistem hatası: {e}. Lütfen tüm alanları doldurduğunuzdan emin olun."

if __name__ == '__main__':
    app.run(debug=True)
  

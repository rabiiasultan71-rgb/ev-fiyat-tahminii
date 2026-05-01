from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# --- VERİ SETİ VE MODEL HAZIRLIĞI ---
df = pd.read_csv('data.csv')

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# İki modeli de eğitiyoruz (Ensemble için)
model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, y)
model_lr = LinearRegression().fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    mkare = float(request.form['metrekare'])
    oda = float(request.form['oda'])
    
    input_data = scaler.transform([[mkare, oda]])
    tahmin1 = model_rf.predict(input_data)[0]
    tahmin2 = model_lr.predict(input_data)[0]
    
    # İki tahminin ortalamasını alıyoruz (Ensemble Tahmin)
    sonuc = (tahmin1 + tahmin2) / 2
    
    return render_template('index.html', tahmin_sonucu=f"{sonuc:,.0f}")

if __name__ == '__main__':
    app.run(debug=True)
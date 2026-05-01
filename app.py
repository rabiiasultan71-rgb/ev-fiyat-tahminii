from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

# Veri Hazırlama
df['Oda_S'] = df['Oda_Sayisi'].apply(oda_temizle)


le = LabelEncoder()
df['Sehir_Kod'] = le.fit_transform(df['Sehir']) 


X = df[['Metrekare', 'Oda_S', 'Sehir_Kod']]
y = df['fiyat']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, y)
model_lr = LinearRegression().fit(X_scaled, y)


sehirler = sorted(df['Sehir'].unique())

@app.route('/')
def index():
    return render_template('index.html', sehirler=sehirler)

@app.route('/predict', methods=['POST'])
def predict():
    mkare = float(request.form['metrekare'])
    oda = float(request.form['oda'])
    secilen_sehir = request.form['sehir']
    
    sehir_kod = le.transform([secilen_sehir])[0]
    
    input_data = scaler.transform([[mkare, oda, sehir_kod]])
    tahmin1 = model_rf.predict(input_data)[0]
    tahmin2 = model_lr.predict(input_data)[0]
    
    sonuc = (tahmin1 + tahmin2) / 2
    
    return render_template('index.html', tahmin_sonucu=f"{sonuc:,.0f}", sehirler=sehirler)

if __name__ == '__main__':
    app.run(debug=True)

import os
import shutil
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_CICD_model():
    mlflow.sklearn.autolog()

    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("[CI DETECTED] Menggunakan tracking URI absolut dari GitHub Actions Environment...")
    else:
        # JIKA dijalankan di laptopmu (Lokal), tetap gunakan port 5000 sesuai kriteria tugas
        print("[LOKAL DETECTED] Menggunakan tracking URI localhost port 5000...")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("UMKM_Pondok_Gede_Modelling")
    
    print("=== [CICD PIPELINE] Memulai Proses Training Model Baseline ===")
    
    data_path = "data_penjualan_siap_ml.csv"
    
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset berhasil dimuat. Ukuran: {df.shape}")
    except FileNotFoundError:
        print("[ERROR] File tidak ditemukan di folder preprocessing!")
        print("Pastikan di folder '../preprocessing/' sudah ada file 'data_penjualan_siap_ml.csv'")
        return

    target_column = 'Total_Penjualan'

    if target_column not in df.columns:
        target_column = [col for col in df.columns if 'Total' in col or 'Penjualan' in col][0]
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="Baseline_Model_CICD"):
        print("Melatih model RandomForestRegressor tanpa hyperparameter tuning...")
        
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared: {r2:.4f}")
        print("=== Seluruh parameter, metrik, dan artifak model dicatat otomatis oleh Autolog! ===")

        target_dir = "model_dir/model"
        
        # Bersihkan folder lama jika ada agar struktur artifact tidak tumpang tindih
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            
        # Simpan model secara lokal agar bisa dibaca perintah COPY di Dockerfile
        mlflow.sklearn.save_model(sk_model=model, path=target_dir)
        print(f"[SUCCESS] Model berhasil diekspor ke folder '{target_dir}'!")

if __name__ == "__main__":
    train_CICD_model()
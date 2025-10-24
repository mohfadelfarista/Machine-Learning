## Machine Learning - Lembar kerja 4-7
Moh. Fadel Farista - 231011401386

Repository ini dibuat untuk mendokumentasikan seluruh proses pembelajaran dan praktikum mata kuliah **Machine Learning**.  
Proyek ini berfokus pada penerapan **algoritma Random Forest** dan **Artificial Neural Network (ANN)** untuk menyelesaikan kasus **klasifikasi biner** menggunakan data tabula
1. **Data Collection**  
   Mengumpulkan dataset yang berisi data kelulusan mahasiswa, kemudian menyimpannya dalam format CSV agar mudah diolah.

2. **Data Cleaning & Preprocessing**  
   Melakukan pembersihan data dengan menghapus duplikasi, menangani missing value, serta mengubah tipe data agar sesuai untuk pemrosesan model.

3. **Exploratory Data Analysis (EDA)**  
   Melakukan analisis statistik dan visualisasi data (menggunakan `seaborn` dan `matplotlib`) untuk memahami pola, distribusi, dan hubungan antar variabel.

4. **Feature Engineering & Selection**  
   Membuat fitur baru yang lebih representatif dan memilih fitur terbaik untuk meningkatkan performa model.

5. **Data Splitting**  
   Memisahkan data menjadi **train**, **validation**, dan **test set** agar model dapat dievaluasi dengan benar.

6. **Model Training dan Evaluation**
   - **Random Forest:** Melatih model ensemble untuk klasifikasi data tabular.  
   - **ANN (Artificial Neural Network):** Membangun jaringan saraf sederhana untuk klasifikasi biner menggunakan TensorFlow/Keras.  
   - Menggunakan metrik evaluasi seperti **accuracy**, **F1-score**, dan **ROC-AUC** untuk menilai performa model.

7. **Model Tuning & Regularization**  
   Menerapkan teknik tuning hyperparameter serta regularisasi seperti **L2**, **Dropout**, dan **Batch Normalization** untuk mencegah overfitting.

8. **Visualization**  
   Menampilkan **learning curve**, **confusion matrix**, dan **ROC curve** untuk menganalisis hasil pelatihan.

9. *(Opsional)* **Deployment API**  
   Menyiapkan model yang telah dilatih agar bisa digunakan kembali untuk prediksi secara otomatis melalui API.

---

### ⚙️ Teknologi yang Digunakan
- **Python 3.x**
- **Pandas**, **NumPy** – untuk pengolahan data  
- **Matplotlib**, **Seaborn** – untuk visualisasi  
- **Scikit-learn** – untuk Random Forest dan evaluasi  
- **TensorFlow / Keras** – untuk membangun ANN  
- **Joblib / Pickle** – untuk menyimpan model  

---

### 🧩 Hasil Akhir
- Model **Random Forest** dan **ANN** yang sudah dilatih dan siap digunakan.  
- File model tersimpan dalam format `.pkl`.  
- Visualisasi hasil pelatihan (learning curve, ROC curve, confusion matrix).  
- File `requirement.txt` berisi daftar library yang digunakan agar mudah direproduksi.
**Moh. Fadel Farista** 
2025knkb

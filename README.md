# Proyek Rawat Inap & Stok Obat (Demo)

Isi paket:
- generate_dataset.py : script untuk membuat dataset sintetik
- train_models.py : script untuk melatih model (inpatient, ward, stay)
- app.py : aplikasi Streamlit untuk demo pengisian pasien, prediksi, dan manajemen stok obat
- data/medicine_stock.csv : contoh stok obat
- data/patients_sample.csv : (tidak termasuk by default, bisa digenerate)
- models/ : target output model (.pkl) setelah training

Cara cepat:
1. (Opsional) Generate dataset: `python generate_dataset.py --n 1000 --out data/patients_sample.csv`
2. Train models: `python train_models.py --data data/patients_sample.csv --out_dir models`
3. Jalankan app Streamlit: `streamlit run app.py`

Catatan: ini adalah demo / prototype. Untuk produksi, tambahkan validasi klinis, audit log, autentikasi, dan integrasi DB.

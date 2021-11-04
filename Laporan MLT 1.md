# Laporan Proyek Machine Learning - Muhammad Rionando D

## Domain Proyek
Domain yang akan digunakan dalam proyek ini adalah di bidang keungan khususnya dunia perbankan, salah satu produk utama dari bank adalah pemberian pinjaman kredit dari customer, tidak sedikit customer yang gagal bayar atau telat dalam melakukan pembayaran hutang kepada bank, dimana tentunya akan menggangu cahflow atau perputaran uang. oleh karena itu perlunya melakukan filtering terhadap calon pelanggan yang akan diberikan pinjaman, filtering ini dapat dilakukan dengan menggunakan data data karakteristikdari setiap pelanggan yang nantinya dapat di klasifikasikan berdasarkan variabel variabel yang ada seperti apa karakteristik pelanggan yang meliki kemungkinan membayar lebih besar. 
- Berikut ini merupakan referensi yang saya ambil dari world bank mengenai classification bank loan [Referensi World Bank](https://openknowledge.worldbank.org/handle/10986/15157) 

## Business Understanding
Pada kasus ini sebuah bank ingin menambah atau memperbanyak pelanggan yang meminjam kepada mereka akan tetapi mereka menginginkan pelanggan yang memiliki kemungkinan bayar yang tinggi. Mereka memiliki data dengan barbagai macam karakteristik pelanggan yang ingin digunakan untuk memaksimalkan campaign ini.

### Problem Statements
1. Dari berbagai macam data karakteristik pelanggan variabel mana yang paling berpengaruh terhadap kemungkinan bayar sukses
2. Dari 3 model yaitu Decision Tree, Random Forest dan KNN mana yang memiliki akurasi paling tinggi

### Goals
1. Mengetahui variabel karakteristik apa yang paling berpengaruh terhadap kesuksesan bayar pelanggan.
2. Membuat model machine learning terbaik untuk memprediksi kasus ini

### Solution statements
Pada kasus kali ini saya menggunakan 3 model machine learning yaitu:
- **Decision tree**. Decision tree merupakan salah satu bentuk algoritma pembelajaran terbaik berdasarkan berbagai metode pembelajaran. Mereka meningkatkan model prediktif dengan akurasi, kemudahan dalam interpretasi, dan stabilitas.
- **Random Forest**. Algoritma random forest adalah salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest juga merupakan algoritma yang sering digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni.
- **KNN**. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k-tetangga terdekat. Nah, itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat). 

## Data Understanding
Dataset yang saya gunakan pada kasus ini bersumber dari kaggle [Bank Loan Modelling](https://www.kaggle.com/itsmesunil/bank-loan-modelling/code) yang memiliki dimensi 5000 X 14 variabel-variabelnya antaralain:

- id : ID pelanggan
- age : Usia
- experience : Pengalaman Kerja
- income : Pendapatan 
- zip_code : Kode Pos
- family : Status Keluarga
- ccavg : Rata-rata pengeluaran kartu kredit
- education : Pendidikan
- mortgage : Tanggungan KPR.
- personal_loan : Apakah pelanggan menerima tawaran campaing sebelumnya
- securities_account : Apakah memiliki akun pengamab?
- cd_account : Apakah Pelanggan memiliki deposio?
- online : Apakah Menggunakan Internet Banking?
- creditcard : apakah menggunakan credit card?

### Data Visualization
![alternate text](https://github.com/rionando/MLT-1/blob/main/image%201.jpg?raw=true)
1. variabel **'age'** dan **'experience'** berdistribusi normal
2. varibael **'income''CCAvg' dan 'mortgage'** memiliki kemiringan positif
3. variabel **ZIP code** memiliki kemiringan negatif

![alternate text](https://github.com/rionando/MLT-1/blob/main/image%202.jpg?raw=true)
1. Kebanyakan pelanggan tidak memiliki **Securities Account, CD Account dan CreditCard**
2. Kebanyakan pelanggan menggunakan **internet banking**
3. Persebaran **tipe keluarga** di dominasi tipe 1 dan 2

![alternate text](https://github.com/rionando/MLT-1/blob/main/image%203.jpg?raw=true)
1. Variable **'age'** beridistribusi normal dengan rata rata usia antara 30-60 tahun
2. Variabel **'Experience'** berdistribusi normal dan banyak customer yang memiliki pengalaman mulai 8 tahun. 
3. Variabel **Income** memiliki kemiringan positif. mayoritas pelanggan memiliki pemasukan antara 45K-55K. 
4. Variabel **CCAvg** juga memiliki kemiringan positif dan rata rata pengeluaran antara 0K-10K dan mayoritas kurang dari 2.5K.
5. Variabel **Mortgage** 70% pelanggan memiliki mortgage kurang dari 40K. Namun nilai maksimalnya adalah 635K.


![alternate text](https://github.com/rionando/MLT-1/blob/main/image%204.jpg?raw=true)
1. Variabel **Income dan CCAvg** memiliki korelasi yang cukup tinggi
2. Variabel **Age dan Experience** memiliki korelasi yang sangat tinggi

## Data Preparation
### Melakukan Split dataset menjadi dua bagian dengan rasio 80% untuk train set dan 20% untuk test set

Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. Kita perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. Ketahuilah bahwa setiap transformasi yang kita lakukan pada data sebelum pemodelan juga merupakan bagian dari model. Karena data uji (test set) berperan sebagai data baru, kita perlu melakukan semua proses transformasi dalam data latih. Inilah alasan mengapa langkah awal adalah membagi dataset sebelum melakukan transformasi apa pun. Tujuannya adalah agar kita tidak mengotori data uji dengan informasi yang kita dapat dari data latih. 

### Melakukan standarisasi data

Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn, StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1. Untuk menghindari kebocoran informasi pada data uji, kita akan menerapkan fitur standarisasi pada data latih. Kemudian, pada tahap evaluasi, kita akan melakukan standarisasi pada data uji. Untuk lebih jelasnya, mari kita terapkan StandardScaler pada data. 

## Modeling
Pada Kasus ini saya menggunakan 3 model machine learning yaitu Decision tree, Random forest dan KNN dan metric yang saya gunakan kali ini hanyalah Accuracy

### Decision Tree
Cara kerja model decision tree dimulai dengan satu node atau simpul. Kemudian, node tersebut bercabang untuk menyatakan pilihan-pilihan yang ada. Selanjutnya, setiap cabang tersebut akan memiliki cabang-cabang baru. Oleh karenanya, metode ini disebut 'tree' karena bentuknya menyerupai pohon yang memiliki banyak cabang.

Untuk Hasil Accuracynya
![alternate text](https://github.com/rionando/MLT-1/blob/main/iamge%205.jpg?raw=true)

### Random Forest
Random forest termasuk ke dalam kelompok model ensemble (group). Apa itu model ensemble? Sederhananya, ia merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. 

Untuk Hasil Accuracynya
![alternate text](https://github.com/rionando/MLT-1/blob/main/iamge%206.jpg?raw=true)


### KNN
KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k-tetangga terdekat. Nah, itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat). KNN bisa digunakan untuk kasus klasifikasi dan regresi. Pada modul ini, kita akan menggunakannya untuk kasus regresi.

Untuk Hasil Accuracynya
![alternate text](https://github.com/rionando/MLT-1/blob/main/image%207.jpg?raw=true)

## Evaluation
### Confusion Matrics
Confusion matrix terdapat 4 hasil kesimpulan yang bisa kita ambil yaitu
- True Positive (TP) :
Interpretasi: Anda memprediksi positif dan itu benar.
- True Negative (TN):
Interpretasi: Anda memprediksi negatif dan itu benar.
- False Positive (FP): (Kesalahan Tipe 1)
Interpretasi: Anda memprediksi positif dan itu salah.
- False Negative (FN): (Kesalahan Tipe 2, kesalahan tipe 2 ini sangat berbahaya)
Interpretasi: Anda memprediksi negatif dan itu salah.

### Metrik evaluasi
Pada kasus klasifikasi kali ini saya hanya menggunakan 1 metrik yaitu akurasi, alasan saya menggunkan akurasi adalah karena akurasi merupakan metrik klasifikasi klasik. dan cukup mudah untuk dipahami serta cocok untuk masalah klasifikasi multiclass.
Rumus dari akurasi sendiri adalah sebagai berikut:
Accuracy = (TP+TN)/(TP+FP+FN+TN)
Akurasi adalah proporsi hasil yang benar di antara jumlah total kasus yang diperiksa.

Berikut merupakan perbandingan hasil akurasi dari 3 model yang digunakan
![alternate text](https://github.com/rionando/MLT-1/blob/main/image%208.jpg?raw=true)

dari hasil tersebut model Random forest memiliki tingkat akurasi paling tinggi


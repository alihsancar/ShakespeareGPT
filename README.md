Character-Level Shakespeare GPT
Bu proje, PyTorch kullanılarak sıfırdan inşa edilmiş, karakter bazlı (character-level) bir Generative Pre-trained Transformer (GPT) dil modelidir. Model, William Shakespeare'in eserlerinden oluşan bir veri seti üzerinde eğitilerek onun yazım tarzını, diyalog yapılarını ve kelime dağarcığını taklit eden metinler üretmeyi amaçlamaktadır.

🚀 Kullanılan Teknolojiler
Dil: Python

Derin Öğrenme Kütüphanesi: PyTorch (torch, torch.nn)

Mimari: Transformer (Sadece Decoder yapısı)

📂 Proje Yapısı
dataset.py: Veri işleme aşamasını yönetir. Shakespeare metnini uzak sunucudan otomatik olarak indirir. Metni kelimeler yerine karakter düzeyinde sayılara çeviren (harf harf tokenizasyon) özel CharacterTokenizer sınıfını içerir. Eğitim için verileri rastgele yığınlar (batch) halinde hazırlayan get_batch fonksiyonunu barındırır.

model.py: GPT modelinin çekirdek mimarisidir. Çoklu Dikkat Mekanizması (Multihead Attention), İleri Beslemeli Sinir Ağları (MLP), Causal Masking (gelecek tokenleri görmeyi engelleyen diagonal maskeleme) ve Transformer Blokları burada PyTorch ile baştan tanımlanmıştır.

train.py: Eğitim döngüsünü çalıştırır. AdamW optimizasyon algoritması, learning rate warmup (öğrenme oranını yavaş yavaş artırma), gradyan kırpma (gradient clipping) ve periyodik model değerlendirmesi işlemlerini içerir. Eğitilen modelin parametrelerini hiperparametre konfigürasyonlarıyla birlikte checkpoints/model.pt olarak kaydeder.

generate.py: Eğitilmiş modeli yükleyerek kullanıcıya etkileşimli bir terminal arayüzü sunar. Kullanıcıdan alınan başlangıç metnine (prompt) göre, istatistiksel olasılıklara (multinomial sampling & temperature) dayanarak yeni metinler üretir.

🧠 Model Mimarisi ve Hiperparametreler
Embedding Boyutu (d_model): 384

Dikkat Başlıkları (Attention Heads): 6

Transformer Katmanları (Layers): 6

Bağlam Penceresi (Block Size): 256 Karakter

Dropout Oranı: 0.1

Kayıp Fonksiyonu (Loss): Cross Entropy Loss

⚙️ Kurulum ve Kullanım
1. Eğitimi Başlatın:
Veri setini indirecek ve belirlenen iterasyon sayısı kadar modeli eğitecektir. (Eğitim sonunda ağırlıklar checkpoints/model.pt dizinine kaydedilir).

Bash
python train.py
2. Metin Üretin:
Eğitilmiş modeli yükleyip üretim (inference) moduna geçer.

Bash
python generate.py
Çalıştırdıktan sonra terminale "ROMEO:" gibi bir başlangıç cümlesi (prompt) girerek modelin metni nasıl sürdürdüğünü test edebilirsiniz. Çıkmak için q veya quit yazabilirsiniz.

# Yap 470 Project
main.py: Projenin çalıştırıldığı ana modül. Model optimizasyonu için gerekli parametreler burada girilir.
dataset.py: Diskten verileri okur, sütun adlarını standartlaştırır, gereksiz öznitelikleri temizler.
dataSplitter.py: dataset.py modülünden çıkan veriyi train, validation ve test olarak üç parçaya böler.
featureExtractor.py: Metinsel verileri sayısal vektör haline getirir, TF-IDF kullanır.
pca.py: featureExtractor'dan çıkan veriyi küçültüp yoğunlaştırır.
KNN.py: Sınıflandırma işlemini gerçekleştiren model.

Not: Detaylı bilgi için dökümandaki "Çalıştırılacak Dosyalar ve Ne Yaptıkları" bölümü incelenebilir.

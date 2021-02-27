# Deep Learning Course 1

## 1. HAFTA
### Görev Kodu : 
Pytorch kütüphanesinin kurulumunu gerçekleştirin. Kaggle’den Avustralyadaki hava durumu ile alakalı şu dataseti indirin. Pandas Kütüphanesini kullanarak dataseti düzenleyip, kullanılabilir hale getirin. Bunun için ilk olarak eksik data olan satırları çıkarın. Ayrıca max temperature dışındaki sütunları da çıkarın. Datasetteki şehir sayısı çok fazlaysa bu şehirlerin sayısını 10-15’e indirebilirsiniz. Ondan sonra bir şehir seçin. Input, seçilen ve diğer şehirlerdeki max hava sıcaklığıdır. Output seçilen şehir için ertesi günkü max hava sıcaklığıdır. böylece input ve output listelerinizi(matrislerini oluşturun.) Bu dataset bir nöral ağ ile eğitilerek input olarak seçilen şehir ve diğer şehirlerin max hava sıcaklıkları girildiğinde seçilen şehrin bir sonraki günkü max hava sıcaklığı tahmin edilmeye çalışılacaktır.

## 3. HAFTA
### Görev Kodu : 
Pytorch kütüphanesi ile 3-5 layer’lik bir neural network oluşturun. Bu neural network’i oluştururken, kütüphanede mevcut olan loss fonksiyonları, aktivasyon fonksiyonlarını listeleyin ve içlerinden birer tane seçin. Videoda anlatıldığı şekle uygun random initialization yapın. Optimzasyon yöntemi olarak Gradient Descent’i tercih edin. Nöral ağın sonunda sigmoid function kullanmayın. Sigmod function’ı neden kullanmamanız gerektiğini düşünün. Avustralyadaki şehirlerin max sıcaklık değerleri dataseti ile bu nöral ağı train edin.
Layer Sayısını, kullandığınız aktivasyon fonksiyonu ve loss fonksiyonunu değiştirerek denemeler yapın.
Layer Sayısı, aktivasyon fonksiyonu ve loss fonksiyonu tercih edilirken nelere dikkat edilmesi gerektiğini düşünün ve bu konuları daha sonra mentorunuzla tartışın.
# Machine-Learning
## 1. HAFTA
Görev Kodu : Python Sklearn Kütüphanesini kullanarak covid_19_data_tr.csv dosyası içerisindeki #Confirmed sayılarını gün ile lineer regression ediniz. 75 gün sonraki Confirmed Cases sayısını tespit ediniz.

## 2. HAFTA
Görev Kodu 1 : Python Sklearn Kütüphanesini kullanarak covid_19_data_tr.csv dosyası içerisindeki #Confirmed sayılarını gün ile polynomial regression ediniz. Hangi polinim fonksiyonu kullanacağınızı Sağlıkçıların öngörülerine göre (peak,plato mevuzu) belirleyiniz. 75 gün sonraki Confirmed Cases sayısını tespit ediniz.
Görev Kodu 2 : Python Sklearn Kütüphanesini kullanarak car_fuel_consumption.csv dosyası içerisindeki distance, speed, temp_outside inputlarını consume outputu ile multiple regression yapınız. Sadece E10 gas_type olan araçlar ile yapınız, diğerlerni droplayınız. Sonuçta, distance,speed, temp_outside inputları verilince consume outpunu verecek bir fonksiyon, model oluşturunuz.

## 3. HAFTA
Görev Kodu 1 : Dron resimlerinden (bu dron resimleri başka bir şey içermemelidir yani görüntünün çoğunu dron oluşturmalıdır) ve dron olmayan resimlerinden (kuş,uçak vb nesneler, manzaralar) oluşan bir dataset toplayın. Dron resimleri ile dron olmayanların oran eşit olsa iyi olabilir, bu konuda net bir şey söyleyemeyiz.
Bu datasetteki fotoğrafları grayscale’ye çevirip Pythonda numpy kütüphanesi ile vektörel hale getirin. Sklearn kütüphanesi ile Logisctic regression yapın.
Bu sırada datasetinizin bir kısmını traininge vermeyin, ayırın bunun için Sklearnde test-training oluşturma fonksiyonları vardır onlara bakınız. Logistic regressionda elde ettiğiniz modeli Test dataseti üzerinde yine sklearn kütüphanesinde bir fonksiyon ile değerlendirin. Modeliniz Test datasetinde dron olan ve olmayan görselleri doğru tespit edebildi mi? Hangi oranda doğru değerlendirebildi ?

## 5. HAFTA
Görev Kodu 1 : Python Sklearn kütüphanesini kullanarak 3. HAFTADA yaptığınız görev kodunu kendi oluşturacağınız sırasıyla 1 ve 2 ara layerin bulunduğu Neural Network yapılarında yazınız. Dron tespit etmede başarı oranda bir değişim gözlemlendi mi, gözlemlendiyse layer sayısının etkisi ne oldu, değerlendiriniz.
NOT : Neural network yapısında layer sayısına bağlı olarak data sayısı ve özelliklerinin logistic regressiona göre daha önemli olduğuna dikkat ediniz.

## 6. HAFTA
Görev Kodu 1: 5. Haftada gerçekleştirilen training ve test işlemlerinin iteration sayısına bağlı Cost değeri grafiklerini çıkarınız. Bu grafiklerin underfit,overfit durumlarını değerlendiriniz.

## 8. HAFTA
Görev Kodu : Eni, Boyu 400 metre olacak şekilde Merkezi (0,0) olan iki boyutlu bir mekan (şehir) düşünün. Az önce bu şehirde büyük bir afet oldu (deprem,sel, çığ vs) ve bu şehirde 40 afetzede var. Bu afetzedelerin konumlarını random kütüphanesini kullanarak ve uniform distribution yaparak türetin.
Bu afetzedeler telefon ile 112 aradı ve kurtarılmalarını istedi. AFAD bu afetzedelerin konumlarını baz istasyonları sayesinde öğrendi ve afetzedeleri konumlarına göre K-means algortimasına göre kümelendirerek 6 ekibi kurtarma işi için görevlendirdi.
Her bir kümede hangi yaralıların olduğunu (konumlarını) belirtiniz, bir şekilde görselleştiriniz.

## 9. HAFTA
GÖREV KODU : random kütüphanesi gaussian distrubtion kullanarak 100 sayı üretin. Bu sayıların ortalamalarını, varyanslarını manuel olarak hesaplayacak bir kod geliştirin ve gaussian distribution fonksiyonuna parametre olarak girdiğiniz ortalama ve varyans değerleriyle karşılaştırın.

## 11. HAFTA
GÖREV KODU : OCR algoritmasını Cam-Scanner’dan veya internetten bulup bilgisayarınıza kurarak deneyin. İyi çalışıp çalışmadığını buraya, foruma yazın.
GÖREV KODU 2 (Optinal) : İsterseniz kendiniz bir OCR algoritması geliştirmeye çalışın. Bu konu hakkında benimle iletişime geçin, yapılabilir mi yapılamaz mı bir konuşalım


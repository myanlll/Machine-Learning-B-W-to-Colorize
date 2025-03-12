# Colorize-Photos-V2 - Siyah-Beyaz Fotoğrafları Renklendirme Macerası

Selamlar agalar! Bugün sizlere bayağı bir emek verdiğim, uykusuz gecelerimin meyvesi olan "Colorize-Photos-V2" projemi anlatacağım. Tabii V2 olmasının sebebi, bunun bir de V1’i oluşu. Eğer sonuçları görüp "Bu ne ya, hiç olmadı" derseniz, V1’i sakın görmeyin, o daha boktan, benden söylemesi! :D Hadi gelin, projeyi baştan sona masaya yatıralım, her detayıyla anlatayım, ne dersiniz?

Tamam, şimdi her şeyiyle her detayıyla anlatayım projemizi. Dataset olarak COCO Train 2017’yi kullandık, belki ondandır, belki de benim sistemimin yetersizliği yüzünden, sadece 3. epoch sonucuyla modelimizi kullanabildik. Elimden gelenin fazlasını yaptım, optimize etmek için bayağı uğraştım. Eğer siz de bu projeyi denemek isterseniz, `train_2017` klasörünün içine büyük çapta fotoğraflarınızı koyarak pekala adım adım kendi modelinizi eğitip bu projeyi kullanabilirsiniz. Bayağı keyifli bir iş, deneyin derim.

Model eğitimi konusunda şöyle bir şey var: Eğer GPU’nuz CUDA destekliyse, işlemleri oldukça hızlı tamamlayabilirsiniz, CPU’ya kıyasla resmen uçarsınız. GPU’nuz 5090 da olsa, 3070Ti da olsa, ya da CUDA destekli her ne olursa olsun, sistem kullanımları en düşükte başlayıp sisteminizin kullanımına göre CPU workers’larını, I/O işlemlerini veya image batch işlemlerini, tıpkı eski arabam Honda Civic’deki V-Tec teknolojisinden ilham alarak, kademeli olarak kod çalışırken duruma göre artırıp azaltıyor. Yani sisteminiz dandik bile olsa bir şekilde optimize oluyor, mis gibi çalışıyor. Tabii CUDA kullanmıyorsa CPU’ya geçiyor, ama o zaman işiniz bayağı uzar, belki bir 3-5 yıl sürer, o yüzden bence CUDA destekli bir GPU şart. :D

Kullandığım teknolojiler ve dataset’ten bahsedeyim biraz. Bu projede PyTorch’u ana framework olarak kullandım, görüntü işleme için de OpenCV ve skimage kütüphaneleriyle uğraştım. Dataset olarak COCO Train 2017’yi tercih ettim, ama siz farklı fotoğraflarla da deneyebilirsiniz. Kodlar zaten GitHub’da açık kaynak, alın, kurcalayın, kendi modelinizi eğitin, oh mis! Sistem özelliklerime gelince, MSI GL76 kullanıyorum, içinde i7-12700H ve RTX 3070Ti 8 GB var. Fena değil gibi görünüyor ama 3. epoch’a gelmem 8 saat sürdü, düşünün artık! Gece 2’de "Şu batch bitsin de yatayım" dedim, sabah 6’da monitörün başında uyuklarken buldum kendimi, uykusuzluktan ağlayacaktım. :') 3. epoch’ta durmak zorunda kaldım, çünkü sistem daha fazla kaldırmadı, dataset de bayağı büyük. Ama bu haliyle bile fena olmadı, siyah-beyaz fotoğrafları renklendirmede vasatın iki tuk üstü işler çıkarıyor.

Sizler de bu modeli farklı foto setleriyle ve/veya daha güçlü cihazlarla eğitebilir, kısacası ağlenebilirsiniz! Ben profesyonelce geliştirmekten daha çok, kişisel bir merakım olan görüntü işleme ve makina öğrenmesi hakkında bir çalışma yaparak bu konuda bir deneme yapmak istedim. Bence fena da olmadı, ne dersiniz? İlerleyen zamanlarda belki QtPy ile güzel bir GUI de hazırlarım, herkes kolayca kullanıversin. Ama şu anda üzerinde uğraşmam gereken diğer freelance işler ve projelerim var, o yüzden bu fikir biraz rafa kalktı. Yine de takipte kalın agalar, çünkü bir sonraki projem eskiden severek oynadığım CS:GO hakkında bir makina öğrenmesi çalışması olacak. Derin öğrenme ile ilgili birkaç fikrimi uygulayacağım, bayağı eğlenceli bir proje olacak, takipte kalmanızı öneririm! :D

## Kurulum

Bu projeyi çalıştırmadan önce aşağıdaki adımları izlemeniz lazım, yoksa kod çalışmaz, baştan söyleyeyim! :D

1. **Gerekli Kütüphaneleri Kurun:**

   Aşağıdaki komutu terminale yapıştırıp çalıştırın, tüm bağımlılıklar yüklenecek:

   ```bash
   pip install torch torchvision torchaudio numpy opencv-python scikit-image pillow psutil keyboard
   ```

   Eğer CUDA destekli bir GPU’nuz varsa, PyTorch’un CUDA versiyonunu yüklediğinizden emin olun, yoksa CPU’ya geçer ve eğitim 3-5 yıl sürebilir, benden söylemesi! :D Alternatif olarak, projedeki `requirements.txt` dosyasını kullanarak şu komutla da kurabilirsiniz:

   ```bash
   pip install -r requirements.txt
   ```

2. **Projeyi Klonlayın:**

   Klonlamak için:

   ```bash
   git clone https://github.com/kullaniciadi/colorize-photos-v2.git
   cd colorize-photos-v2
   ```

3. **Gerekli Dosyaları Hazırlayın:**

   - `train_2017` klasörüne siyah-beyaz fotoğraflarınızı koyun (eğitim için).
   - `input_user` klasörüne renklendirmek istediğiniz kullanıcı fotoğraflarınızı ekleyin.

## Kullanım

Projenin farklı aşamalarını çalıştırmak için aşağıdaki adımları izleyin:

1. **Modeli Eğitin:**

   Eğitimi başlatmak için:

   ```bash
   python 3-train.py
   ```

   Bu, `train_2017` klasöründeki fotoğraflarla modeli eğitir ve `models/generator.pth` dosyasına kaydeder. 3 epoch sonra durur, benim sistemimde 8 saat sürdü! :D

2. **Kullanıcı Fotoğraflarını Renklendirin:**

   Kullanıcıdan gelen fotoğrafları renklendirmek için:

   ```bash
   python 5-user_inputs.py --input_dir input_user --output_dir output_user
   ```

   Bu, `input_user` klasöründeki fotoğrafları alır, renklendirir ve `output_user` klasörüne `_predicted.png` ve `_collage.png` dosyaları olarak kaydeder. Kolaj, orijinal siyah-beyaz ve renklendirilmiş hali yan yana gösterir.

3. **Diğer Adımlar:**

   - `1-preprocess.py`: Verileri ön işlemek için kullanılır (isteğe bağlı).
   - `2-move_val.py`: Validation setini ayırmak için.
   - `3-train.py`: Modeli eğitip generate.pth dosyanızı elde edebilmeniz için.
   - `4-run-model.py`: Modeli test etmek için (genelde eğitim sonrası test için).
   - `5-user_inputs.py`: Kullanıcı tarafından girilen fotoğrafları test etmek ve rneklendirmek için.
## Kodun Teorik Yapısı

Bu proje, siyah-beyaz fotoğrafları renklendirmek için U-Net mimarisine dayalı bir derin öğrenme modeli kullanıyor. İşte temel mantık:

- **Veri İşleme:** Fotoğraflar RGB’den LAB renk uzayına çevriliyor. L kanalı (parlaklık) modelin girişi, AB kanalları (renk bilgileri) ise çıkış olarak tahmin ediliyor.
- **Model Mimarisi:** U-Net, encoder-decoder yapısıyla çalışıyor. Encoder, görüntüyü özellik haritalarına indirgerken, decoder bu haritaları yeniden renklendirme için AB kanallarına dönüştürüyor. Skip bağlantıları, detay kaybını azaltır.
- **Eğitim:** COCO Train 2017 dataset’iyle 3 epoch eğitildi. Model, LAB uzayında AB kanallarını tahmin ederken, Tanh aktivasyonu ve histogram eşitleme gibi tekniklerle sepia gibi renk hatalarını düzeltiyor.
- **Çıkış:** Tahmin edilen AB kanalları, L kanalıyla birleştirilip RGB’ye dönüştürülüyor ve renk doygunluğu artırılarak sonuç elde ediliyor.

Bu, makina öğrenmesiyle görüntü renklendirmenin temel bir örneği. Daha fazla epoch ve güçlü bir sistemle sonuçları daha da iyileştirebilirsiniz!

## Örnek Çıktılar

Projenin nasıl çalıştığını görmek için ana dizine 3. epoch’tan çıkan 7 tane örnek çıktı koydum. Siyah-beyaz fotoğrafları renklendirme konusunda fena iş çıkarmadı, siz ne dersiniz? Aşağıda görebilirsiniz:

- ![Renklendirilen Bazı Fotoğraflar](./1.jpg)
- ![Renklendirilen Bazı Fotoğraflar](./2.jpg)
- ![Renklendirilen Bazı Fotoğraflar](./3.jpg)
- ![Renklendirilen Bazı Fotoğraflar](./4.jpg)
- ![Renklendirilen Bazı Fotoğraflar](./5.jpg)
- ![Renklendirilen Bazı Fotoğraflar](./6.jpg)
- ![Renklendirilen Bazı Fotoğraflar](./7.jpg)

## Kullanılan Teknolojiler ve Dataset

- **Framework:** PyTorch (ana framework olarak kullandım, model eğitimi için olmazsa olmaz).
- **Görüntü İşleme:** OpenCV ve scikit-image (renk dönüşümleri ve işleme için).
- **Diğer Kütüphaneler:** PIL (görüntü dosyalarını yönetmek için), numpy (sayısal işlemler), psutil (sistem kaynaklarını izlemek için), keyboard (eğitim sırasında durdurma tuşu için).
- **Dataset:** COCO Train 2017 (bayağı büyük bir dataset, ama farklı fotoğraflarla da deneyebilirsiniz).

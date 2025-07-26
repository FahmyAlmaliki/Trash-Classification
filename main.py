import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class TrashClassifier:
    def __init__(self, img_width=224, img_height=224):
        """
        Inisialisasi klasifikator sampah
        """
        self.img_width = img_width
        self.img_height = img_height
        self.model = None
        self.label_encoder = LabelEncoder()
        
        # Kategori sampah yang akan diklasifikasi
        self.trash_categories = {
            0: 'Plastik',
            1: 'Logam/Besi', 
            2: 'Kertas',
            3: 'Kaca',
            4: 'Organik',
            5: 'Kardus'
        }
        
    def create_sample_data(self, num_samples=1000):
        """
        Membuat data sampel untuk demonstrasi
        """
        print("Membuat data sampel untuk pelatihan model...")
        
        # Simulasi fitur-fitur sampah
        np.random.seed(42)
        
        # Fitur untuk setiap kategori sampah
        features = []
        labels = []
        
        for category_id, category_name in self.trash_categories.items():
            for _ in range(num_samples // len(self.trash_categories)):
                if category_id == 0:  # Plastik
                    # Plastik: fleksibel, ringan, berbagai warna
                    sample = [
                        np.random.uniform(0.1, 0.3),  # density
                        np.random.uniform(0.7, 0.9),  # flexibility  
                        np.random.uniform(0.2, 0.8),  # transparency
                        np.random.uniform(0.1, 0.4),  # hardness
                        np.random.uniform(0.3, 0.9),  # color_variance
                    ]
                elif category_id == 1:  # Logam/Besi
                    # Logam: keras, berat, konduktif
                    sample = [
                        np.random.uniform(2.5, 8.0),  # density
                        np.random.uniform(0.1, 0.2),  # flexibility
                        np.random.uniform(0.0, 0.1),  # transparency
                        np.random.uniform(0.8, 1.0),  # hardness
                        np.random.uniform(0.1, 0.3),  # color_variance
                    ]
                elif category_id == 2:  # Kertas
                    # Kertas: ringan, mudah terbakar, menyerap air
                    sample = [
                        np.random.uniform(0.3, 0.9),  # density
                        np.random.uniform(0.4, 0.7),  # flexibility
                        np.random.uniform(0.0, 0.2),  # transparency
                        np.random.uniform(0.1, 0.3),  # hardness
                        np.random.uniform(0.2, 0.6),  # color_variance
                    ]
                elif category_id == 3:  # Kaca
                    # Kaca: keras, berat, transparan
                    sample = [
                        np.random.uniform(2.0, 2.8),  # density
                        np.random.uniform(0.0, 0.1),  # flexibility
                        np.random.uniform(0.7, 1.0),  # transparency
                        np.random.uniform(0.9, 1.0),  # hardness
                        np.random.uniform(0.0, 0.4),  # color_variance
                    ]
                elif category_id == 4:  # Organik
                    # Organik: bervariasi, mudah busuk
                    sample = [
                        np.random.uniform(0.3, 1.5),  # density
                        np.random.uniform(0.2, 0.8),  # flexibility
                        np.random.uniform(0.0, 0.3),  # transparency
                        np.random.uniform(0.1, 0.6),  # hardness
                        np.random.uniform(0.4, 1.0),  # color_variance
                    ]
                else:  # Kardus
                    # Kardus: ringan, fleksibel
                    sample = [
                        np.random.uniform(0.1, 0.5),  # density
                        np.random.uniform(0.6, 0.9),  # flexibility
                        np.random.uniform(0.0, 0.1),  # transparency
                        np.random.uniform(0.2, 0.4),  # hardness
                        np.random.uniform(0.2, 0.5),  # color_variance
                    ]
                
                features.append(sample)
                labels.append(category_id)
        
        return np.array(features), np.array(labels)
    
    def create_cnn_model(self):
        """
        Membuat model CNN untuk klasifikasi gambar sampah
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(len(self.trash_categories), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_simple_model(self):
        """
        Membuat model sederhana untuk klasifikasi berdasarkan fitur
        """
        model = Sequential([
            Dense(128, activation='relu', input_shape=(5,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(len(self.trash_categories), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, features, labels, test_size=0.2, epochs=50):
        """
        Melatih model klasifikasi
        """
        print("Memulai pelatihan model...")
        
        # Membagi data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Membuat model
        self.model = self.create_simple_model()
        
        # Callback untuk early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        # Melatih model
        history = self.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluasi model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nAkurasi model pada data test: {test_accuracy:.4f}")
        
        return history, X_test, y_test
    
    def predict(self, features):
        """
        Memprediksi kategori sampah
        """
        if self.model is None:
            print("Error: Model belum dilatih!")
            return None
        
        # Reshape jika input hanya satu sample
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        predictions = self.model.predict(features)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        results = []
        for i, (pred_class, confidence) in enumerate(zip(predicted_classes, confidence_scores)):
            results.append({
                'kategori': self.trash_categories[pred_class],
                'confidence': confidence,
                'probabilitas': predictions[i]
            })
        
        return results
    
    def plot_training_history(self, history):
        """
        Menampilkan grafik pelatihan model
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot akurasi
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, X_test, y_test):
        """
        Menampilkan confusion matrix
        """
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Prediksi
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.trash_categories.values()),
                   yticklabels=list(self.trash_categories.values()))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=list(self.trash_categories.values())))
    
    def classify_trash_interactive(self):
        """
        Interface interaktif untuk klasifikasi sampah
        """
        print("\n" + "="*50)
        print("SISTEM KLASIFIKASI SAMPAH")
        print("="*50)
        print("Masukkan karakteristik sampah (0-1 untuk sebagian besar fitur):")
        print("1. Density (kepadatan): 0.1-8.0")
        print("2. Flexibility (fleksibilitas): 0.0-1.0") 
        print("3. Transparency (transparansi): 0.0-1.0")
        print("4. Hardness (kekerasan): 0.0-1.0")
        print("5. Color variance (variasi warna): 0.0-1.0")
        print("-"*50)
        
        try:
            density = float(input("Masukkan density: "))
            flexibility = float(input("Masukkan flexibility: "))
            transparency = float(input("Masukkan transparency: "))
            hardness = float(input("Masukkan hardness: "))
            color_variance = float(input("Masukkan color variance: "))
            
            # Membuat array fitur
            features = np.array([density, flexibility, transparency, hardness, color_variance])
            
            # Prediksi
            results = self.predict(features)
            
            if results:
                result = results[0]
                print(f"\n{'='*50}")
                print("HASIL KLASIFIKASI:")
                print(f"{'='*50}")
                print(f"Kategori sampah: {result['kategori']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"\nProbabilitas untuk setiap kategori:")
                for i, prob in enumerate(result['probabilitas']):
                    print(f"  {self.trash_categories[i]}: {prob:.2%}")
                
                # Rekomendasi penanganan
                self.give_disposal_recommendation(result['kategori'])
                
        except ValueError:
            print("Error: Masukkan nilai numerik yang valid!")
        except Exception as e:
            print(f"Error: {e}")
    
    def give_disposal_recommendation(self, category):
        """
        Memberikan rekomendasi penanganan sampah
        """
        recommendations = {
            'Plastik': [
                "🔄 Pisahkan berdasarkan jenis plastik",
                "🧽 Bersihkan dari sisa makanan",
                "♻️ Masukkan ke tempat sampah daur ulang",
                "💡 Tip: Cek kode daur ulang di kemasan"
            ],
            'Logam/Besi': [
                "🧲 Pisahkan logam dari bahan lain",
                "🧽 Bersihkan dari karat atau kotoran",
                "♻️ Jual ke pengepul logam bekas",
                "💡 Tip: Logam memiliki nilai ekonomi tinggi"
            ],
            'Kertas': [
                "📄 Pisahkan kertas bersih dari yang kotor",
                "🗂️ Hindari kertas berlapis lilin/plastik",
                "♻️ Masukkan ke bank sampah",
                "💡 Tip: Kertas koran memiliki nilai lebih rendah"
            ],
            'Kaca': [
                "⚠️ Hati-hati saat menangani",
                "🧽 Bersihkan dari label dan tutup",
                "♻️ Pisahkan berdasarkan warna",
                "💡 Tip: Kaca dapat didaur ulang berulang kali"
            ],
            'Organik': [
                "🍂 Buat kompos di rumah",
                "🪣 Gunakan tempat khusus sampah organik",
                "🌱 Manfaatkan untuk pupuk tanaman",
                "💡 Tip: Proses pengomposan membutuhkan 2-3 bulan"
            ],
            'Kardus': [
                "📦 Ratakan dan lipat kardus",
                "🧽 Hilangkan selotip dan staples",
                "♻️ Kumpulkan untuk dijual ke pengepul",
                "💡 Tip: Kardus kering lebih bernilai"
            ]
        }
        
        print(f"\n🌿 REKOMENDASI PENANGANAN - {category.upper()}:")
        print("-" * 40)
        for rec in recommendations.get(category, ["Konsultasi dengan petugas kebersihan"]):
            print(rec)

def main():
    """
    Fungsi utama program
    """
    print("🗑️ PROGRAM KLASIFIKASI SAMPAH")
    print("=" * 50)
    
    # Inisialisasi classifier
    classifier = TrashClassifier()
    
    while True:
        print("\nPilih menu:")
        print("1. Latih model dengan data sampel")
        print("2. Klasifikasi sampah interaktif")
        print("3. Keluar")
        
        choice = input("\nMasukkan pilihan (1-3): ").strip()
        
        if choice == '1':
            print("\nMemulai pelatihan model...")
            
            # Membuat data sampel
            features, labels = classifier.create_sample_data(num_samples=2000)
            
            # Melatih model
            history, X_test, y_test = classifier.train_model(features, labels, epochs=100)
            
            # Menampilkan grafik
            classifier.plot_training_history(history)
            classifier.plot_confusion_matrix(X_test, y_test)
            
            print("\n✅ Model berhasil dilatih!")
            
        elif choice == '2':
            if classifier.model is None:
                print("\n⚠️ Model belum dilatih! Silakan latih model terlebih dahulu.")
                continue
            
            classifier.classify_trash_interactive()
            
            # Tanya apakah ingin melanjutkan
            if input("\nKlasifikasi sampah lain? (y/n): ").lower() != 'y':
                continue
                
        elif choice == '3':
            print("\nTerima kasih telah menggunakan program klasifikasi sampah! 🌍")
            break
            
        else:
            print("\n❌ Pilihan tidak valid! Silakan pilih 1-3.")

if __name__ == "__main__":
    main()
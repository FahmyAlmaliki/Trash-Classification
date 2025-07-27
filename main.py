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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
import threading
import time
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
        Membuat model CNN untuk klasifikasi gambar sampah menggunakan transfer learning
        """
        # Menggunakan MobileNetV2 sebagai base model untuk transfer learning
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        
        # Unfreeze beberapa layer terakhir untuk fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(len(self.trash_categories), activation='softmax')
        ])
        
        # Menggunakan learning rate yang lebih rendah untuk fine-tuning
        from tensorflow.keras.optimizers import Adam
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
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
    
    def detect_objects_with_contours(self, frame):
        """
        Deteksi objek menggunakan contour detection untuk bounding box
        """
        # Convert ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold untuk mendapatkan objek
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations untuk membersihkan noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours berdasarkan area
        min_area = 2000  # Minimum area untuk objek
        max_area = frame.shape[0] * frame.shape[1] * 0.8  # Maksimum 80% dari frame
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Approximate contour to reduce points
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                valid_contours.append(contour)
        
        return valid_contours
    
    def get_bounding_boxes(self, frame):
        """
        Mendapatkan bounding boxes dari objek yang terdeteksi
        """
        contours = self.detect_objects_with_contours(frame)
        bounding_boxes = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter berdasarkan aspect ratio dan ukuran
            aspect_ratio = w / h
            if 0.3 < aspect_ratio < 3.0 and w > 50 and h > 50:  # Reasonable aspect ratio and size
                bounding_boxes.append((x, y, w, h))
        
        # Sort by area (largest first)
        bounding_boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
        
        # Return maksimal 3 objek terbesar
        return bounding_boxes[:3]
    
    def classify_region(self, frame, bbox):
        """
        Klasifikasi objek dalam region tertentu (bounding box)
        """
        x, y, w, h = bbox
        
        # Extract region of interest
        roi = frame[y:y+h, x:x+w]
        
        # Resize ROI if too small
        if roi.shape[0] < 50 or roi.shape[1] < 50:
            roi = cv2.resize(roi, (100, 100))
        
        # Predict using the region
        predicted_class, confidence = self.predict_from_image(roi)
        
        return predicted_class, confidence
    
    def preprocess_image(self, image):
        """
        Preprocess gambar untuk prediksi
        """
        # Resize gambar ke dimensi yang dibutuhkan model
        image_resized = cv2.resize(image, (self.img_width, self.img_height))
        
        # Normalisasi pixel values ke range [0, 1]
        image_normalized = image_resized.astype('float32') / 255.0
        
        # Tambahkan batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def predict_from_image(self, image):
        """
        Prediksi kategori sampah dari gambar dengan improved accuracy
        """
        if self.model is None:
            return None, 0.0
        
        # Multiple predictions dengan augmentasi untuk akurasi lebih tinggi
        predictions_list = []
        
        # Prediksi original
        processed_image = self.preprocess_image(image)
        pred = self.model.predict(processed_image, verbose=0)
        predictions_list.append(pred[0])
        
        # Prediksi dengan augmentasi ringan untuk ensemble
        augmented_images = self.create_augmented_images(image, num_augments=4)
        for aug_img in augmented_images:
            processed_aug = self.preprocess_image(aug_img)
            pred_aug = self.model.predict(processed_aug, verbose=0)
            predictions_list.append(pred_aug[0])
        
        # Ensemble prediction (rata-rata)
        ensemble_pred = np.mean(predictions_list, axis=0)
        predicted_class = np.argmax(ensemble_pred)
        confidence = np.max(ensemble_pred)
        
        # Confidence threshold - jika terlalu rendah, return sebagai "tidak dikenal"
        if confidence < 0.3:
            return None, confidence
        
        return predicted_class, confidence
    
    def create_augmented_images(self, image, num_augments=4):
        """
        Membuat variasi gambar untuk ensemble prediction
        """
        augmented = []
        
        for _ in range(num_augments):
            aug_img = image.copy()
            
            # Random brightness
            brightness_factor = np.random.uniform(0.8, 1.2)
            aug_img = np.clip(aug_img * brightness_factor, 0, 255).astype(np.uint8)
            
            # Random rotation (kecil)
            angle = np.random.uniform(-10, 10)
            center = (aug_img.shape[1]//2, aug_img.shape[0]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (aug_img.shape[1], aug_img.shape[0]))
            
            # Random noise
            noise = np.random.normal(0, 5, aug_img.shape).astype(np.uint8)
            aug_img = np.clip(aug_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            augmented.append(aug_img)
        
        return augmented
    
    def draw_prediction_info_with_bbox(self, frame, bounding_boxes, predictions, fps):
        """
        Menggambar informasi prediksi dengan bounding boxes
        """
        height, width = frame.shape[:2]
        
        # Background untuk informasi utama
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 160), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Gambar bounding boxes dan klasifikasi
        for i, (bbox, prediction) in enumerate(zip(bounding_boxes, predictions)):
            x, y, w, h = bbox
            predicted_class, confidence = prediction
            
            if predicted_class is not None:
                category_name = self.trash_categories[predicted_class]
                
                # Color coding berdasarkan confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # Hijau
                    status = "SANGAT YAKIN"
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Kuning
                    status = "YAKIN"
                elif confidence > 0.4:
                    color = (0, 165, 255)  # Orange
                    status = "RAGU-RAGU"
                else:
                    color = (0, 0, 255)  # Merah
                    status = "TIDAK YAKIN"
                
                # Gambar bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Label dengan background
                label = f"{category_name} ({confidence:.1%})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background untuk label
                cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                             (x + label_size[0] + 10, y), color, -1)
                
                # Text label
                cv2.putText(frame, label, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Informasi di sidebar untuk objek pertama (terbesar)
                if i == 0:
                    cv2.putText(frame, f"Objek Utama: {category_name}", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f"Confidence: {confidence:.1%} ({status})", (20, 75),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Progress bar untuk confidence
                    bar_width = width - 40
                    bar_height = 10
                    bar_x, bar_y = 20, 85
                    
                    # Background bar
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                    # Confidence bar
                    confidence_width = int(bar_width * confidence)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
            
            else:
                # Objek tidak dikenal
                cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
                cv2.putText(frame, "Unknown", (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Jika tidak ada objek terdeteksi
        if not bounding_boxes:
            cv2.putText(frame, "Tidak ada objek terdeteksi", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Coba dekatkan objek ke kamera", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Info jumlah objek terdeteksi
        cv2.putText(frame, f"Objek terdeteksi: {len(bounding_boxes)}", (20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS dengan background
        fps_text = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (width - text_size[0] - 20, 10), (width - 10, 40), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (width - text_size[0] - 15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instruksi dengan background
        instructions = [
            "Tekan 'q' untuk keluar",
            "Tekan 's' untuk screenshot", 
            "Tekan 'h' untuk help",
            "Tekan 'b' untuk toggle bbox"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = height - 80 + (i * 18)
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(frame, (10, y_pos - 12), (text_size[0] + 20, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(frame, instruction, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return frame
    
    def realtime_classification(self):
        """
        Klasifikasi sampah secara realtime menggunakan webcam dengan bounding box detection
        """
        if self.model is None:
            print("⚠️ Model belum dilatih! Silakan latih model CNN terlebih dahulu (pilihan 2).")
            return
        
        print("🎥 Memulai klasifikasi realtime dengan bounding box...")
        print("📝 Instruksi:")
        print("   - Tekan 'q' untuk keluar")
        print("   - Tekan 's' untuk mengambil screenshot")
        print("   - Tekan 'h' untuk menampilkan help")
        print("   - Tekan 'p' untuk pause/resume")
        print("   - Tekan 'b' untuk toggle bounding box detection")
        print("   - Tunjukkan sampah ke kamera untuk klasifikasi")
        print("\n🔄 Membuka kamera...")
        
        # Inisialisasi kamera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Tidak dapat mengakses kamera!")
            print("💡 Tips:")
            print("   - Pastikan kamera tidak digunakan aplikasi lain")
            print("   - Coba restart aplikasi")
            print("   - Periksa permission kamera")
            return
        
        # Set resolusi kamera yang lebih baik
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Variabel untuk tracking
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0
        screenshot_counter = 0
        paused = False
        show_help = False
        use_bbox = True  # Default menggunakan bounding box
        
        # Buffer untuk stabilisasi prediksi
        prediction_buffer = []
        buffer_size = 5
        
        print("✅ Kamera berhasil dibuka! Menampilkan feed...")
        print("🎯 Mode: Bounding Box Detection AKTIF")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error: Tidak dapat membaca frame dari kamera!")
                    break
                
                # Flip frame horizontally untuk mirror effect
                frame = cv2.flip(frame, 1)
                
                if use_bbox:
                    # Deteksi objek dengan bounding box
                    bounding_boxes = self.get_bounding_boxes(frame)
                    
                    # Klasifikasi setiap objek yang terdeteksi
                    predictions = []
                    for bbox in bounding_boxes:
                        pred_class, confidence = self.classify_region(frame, bbox)
                        predictions.append((pred_class, confidence))
                    
                    # Stabilisasi prediksi untuk objek utama (terbesar)
                    if bounding_boxes and predictions:
                        main_prediction = predictions[0]  # Objek terbesar
                        prediction_buffer.append(main_prediction)
                        if len(prediction_buffer) > buffer_size:
                            prediction_buffer.pop(0)
                        
                        # Gunakan prediksi yang paling sering muncul
                        if len(prediction_buffer) >= 3:
                            classes = [p[0] for p in prediction_buffer if p[0] is not None]
                            if classes:
                                most_common_class = max(set(classes), key=classes.count)
                                confidences = [p[1] for p in prediction_buffer if p[0] == most_common_class]
                                avg_confidence = np.mean(confidences)
                                predictions[0] = (most_common_class, avg_confidence)
                
                else:
                    # Mode klasik tanpa bounding box
                    predicted_class, confidence = self.predict_from_image(frame)
                    bounding_boxes = []
                    predictions = [(predicted_class, confidence)] if predicted_class is not None else []
                
                # Hitung FPS
                fps_counter += 1
                if fps_counter >= 30:  # Update FPS setiap 30 frame
                    fps_end_time = time.time()
                    fps = fps_counter / (fps_end_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = fps_end_time
            
            # Gambar informasi prediksi
            if show_help:
                frame_with_info = self.draw_help_screen_with_bbox(frame)
            else:
                if use_bbox:
                    frame_with_info = self.draw_prediction_info_with_bbox(frame, bounding_boxes, predictions, fps)
                else:
                    # Fallback ke method lama jika bbox dimatikan
                    pred_class, confidence = predictions[0] if predictions else (None, 0.0)
                    frame_with_info = self.draw_prediction_info_classic(frame, pred_class, confidence, fps)
            
            # Tambahkan status mode
            mode_text = "BBOX MODE" if use_bbox else "CLASSIC MODE"
            cv2.putText(frame_with_info, mode_text, (frame_with_info.shape[1] - 150, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Tambahkan status paused
            if paused:
                cv2.putText(frame_with_info, "PAUSED - Tekan 'p' untuk melanjutkan", 
                           (20, frame_with_info.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Tampilkan frame
            cv2.imshow('Trash Classification - Realtime with Bounding Box', frame_with_info)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Keluar dari mode realtime...")
                break
            elif key == ord('s'):
                # Screenshot
                screenshot_counter += 1
                filename = f"screenshot_bbox_{screenshot_counter:03d}.jpg"
                cv2.imwrite(filename, frame_with_info)
                print(f"📸 Screenshot disimpan: {filename}")
                
                # Tampilkan hasil klasifikasi di terminal
                if use_bbox and bounding_boxes and predictions:
                    print(f"📊 Hasil klasifikasi screenshot (Bounding Box Mode):")
                    for i, (bbox, prediction) in enumerate(zip(bounding_boxes, predictions)):
                        pred_class, confidence = prediction
                        if pred_class is not None:
                            category_name = self.trash_categories[pred_class]
                            x, y, w, h = bbox
                            print(f"   Objek {i+1}: {category_name} ({confidence:.1%}) - Area: {w}x{h}")
                            if i == 0 and confidence > 0.6:  # Hanya untuk objek utama dengan confidence tinggi
                                self.give_disposal_recommendation(category_name)
                elif predictions and predictions[0][0] is not None:
                    pred_class, confidence = predictions[0]
                    category_name = self.trash_categories[pred_class]
                    print(f"📊 Hasil klasifikasi screenshot (Classic Mode):")
                    print(f"   Kategori: {category_name} ({confidence:.1%})")
                    if confidence > 0.6:
                        self.give_disposal_recommendation(category_name)
                else:
                    print("   ⚠️ Tidak ada objek terdeteksi atau confidence rendah")
                        
            elif key == ord('p'):
                paused = not paused
                if paused:
                    print("⏸️ Video di-pause")
                else:
                    print("▶️ Video dilanjutkan")
                    
            elif key == ord('h'):
                show_help = not show_help
                
            elif key == ord('b'):
                use_bbox = not use_bbox
                prediction_buffer.clear()  # Clear buffer saat ganti mode
                if use_bbox:
                    print("🎯 Bounding Box Detection: AKTIF")
                else:
                    print("🎯 Bounding Box Detection: NONAKTIF (Classic Mode)")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Kamera ditutup.")
        print(f"📊 Total screenshot diambil: {screenshot_counter}")
        print(f"🎯 Mode terakhir: {'Bounding Box' if use_bbox else 'Classic'}")
    
    def draw_help_screen_with_bbox(self, frame):
        """
        Menampilkan help screen dengan informasi bounding box
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (frame.shape[1]-50, frame.shape[0]-50), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        help_text = [
            "=== BANTUAN PENGGUNAAN (BOUNDING BOX) ===",
            "",
            "Kontrol:",
            "  'q' - Keluar dari aplikasi",
            "  's' - Ambil screenshot",
            "  'p' - Pause/Resume video",
            "  'h' - Toggle help screen",
            "  'b' - Toggle bounding box detection",
            "",
            "Mode Bounding Box:",
            "  - Mendeteksi hingga 3 objek sekaligus",
            "  - Kotak warna menunjukkan confidence",
            "  - Hijau: Sangat yakin (>80%)",
            "  - Kuning: Yakin (60-80%)",
            "  - Orange: Ragu-ragu (40-60%)",
            "  - Merah: Tidak yakin (<40%)",
            "",
            "Tips untuk hasil terbaik:",
            "  - Gunakan background kontras",
            "  - Objek minimal 5cm dari kamera",
            "  - Hindari objek terlalu kecil",
            "  - Pencahayaan merata",
            "",
            "Tekan 'h' lagi untuk menutup help"
        ]
        
        y_start = 70
        for i, text in enumerate(help_text):
            cv2.putText(frame, text, (60, y_start + i*22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_prediction_info_classic(self, frame, predicted_class, confidence, fps):
        """
        Menggambar informasi prediksi mode klasik (tanpa bounding box)
        """
        height, width = frame.shape[:2]
        
        # Background untuk informasi utama
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Informasi kategori
        if predicted_class is not None:
            category_name = self.trash_categories[predicted_class]
            
            # Status confidence dengan color coding
            if confidence > 0.8:
                color = (0, 255, 0)  # Hijau
                status = "SANGAT YAKIN"
            elif confidence > 0.6:
                color = (0, 255, 255)  # Kuning
                status = "YAKIN"
            elif confidence > 0.4:
                color = (0, 165, 255)  # Orange
                status = "RAGU-RAGU"
            else:
                color = (0, 0, 255)  # Merah
                status = "TIDAK YAKIN"
            
            # Teks kategori
            cv2.putText(frame, f"Kategori: {category_name}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Teks confidence dan status
            cv2.putText(frame, f"Confidence: {confidence:.1%} ({status})", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Progress bar untuk confidence
            bar_width = width - 40
            bar_height = 10
            bar_x, bar_y = 20, 85
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            # Confidence bar
            confidence_width = int(bar_width * confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
            
        else:
            cv2.putText(frame, "Objek tidak dikenal atau confidence rendah", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # FPS dengan background
        fps_text = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (width - text_size[0] - 20, 10), (width - 10, 40), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (width - text_size[0] - 15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_help_screen(self, frame):
        """
        Menampilkan help screen klasik
        """
        return self.draw_help_screen_with_bbox(frame)  # Gunakan help screen yang lebih lengkap
    
    def draw_help_screen(self, frame):
        """
        Menampilkan help screen
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (frame.shape[1]-50, frame.shape[0]-50), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        help_text = [
            "=== BANTUAN PENGGUNAAN ===",
            "",
            "Kontrol:",
            "  'q' - Keluar dari aplikasi",
            "  's' - Ambil screenshot",
            "  'p' - Pause/Resume video",
            "  'h' - Toggle help screen",
            "",
            "Tips untuk akurasi lebih baik:",
            "  - Pastikan pencahayaan cukup",
            "  - Pegang objek dengan stabil",
            "  - Dekatkan objek ke kamera",
            "  - Hindari background yang ramai",
            "  - Tunggu confidence > 60%",
            "",
            "Kategori yang dapat dideteksi:",
            "  Plastik, Logam/Besi, Kertas,",
            "  Kaca, Organik, Kardus",
            "",
            "Tekan 'h' lagi untuk menutup help"
        ]
        
        y_start = 80
        for i, text in enumerate(help_text):
            cv2.putText(frame, text, (70, y_start + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def save_model(self, filename="trash_classifier_model.h5"):
        """
        Simpan model yang sudah dilatih
        """
        if self.model is not None:
            self.model.save(filename)
            print(f"✅ Model disimpan ke: {filename}")
        else:
            print("❌ Tidak ada model untuk disimpan!")
    
    def load_model(self, filename="trash_classifier_model.h5"):
        """
        Load model yang sudah dilatih sebelumnya
        """
        try:
            if os.path.exists(filename):
                from tensorflow.keras.models import load_model
                self.model = load_model(filename)
                print(f"✅ Model berhasil dimuat dari: {filename}")
                return True
            else:
                print(f"❌ File model tidak ditemukan: {filename}")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def train_cnn_model_with_sample_images(self):
        """
        Melatih model CNN dengan data gambar sampel (simulasi) - Improved Version
        """
        print("🖼️ Membuat dataset gambar sampel untuk CNN...")
        
        # Simulasi dataset gambar dengan lebih banyak variasi
        num_samples_per_class = 300  # Lebih banyak data
        images = []
        labels = []
        
        for class_id, class_name in self.trash_categories.items():
            print(f"   Membuat sampel untuk {class_name}...")
            
            for i in range(num_samples_per_class):
                # Buat gambar sintetis dengan pola yang lebih realistis
                img = np.random.rand(self.img_height, self.img_width, 3) * 0.3 + 0.1
                
                # Tambahkan pola spesifik untuk setiap kelas dengan lebih banyak variasi
                if class_id == 0:  # Plastik - berbagai pola plastik
                    # Pola garis vertikal/horizontal
                    if i % 3 == 0:
                        img[:, ::8, :] = [0.7 + np.random.rand()*0.3, 0.1, 0.1]
                    elif i % 3 == 1:
                        img[::8, :, :] = [0.2, 0.7 + np.random.rand()*0.3, 0.2]
                    else:
                        # Pola dot untuk plastik transparan
                        img[::15, ::15, :] = [0.8, 0.8, 0.9]
                        
                elif class_id == 1:  # Logam - pola metalik yang lebih realistis
                    # Efek metalik dengan gradient
                    gradient = np.linspace(0.3, 0.8, self.img_width)
                    for j in range(self.img_height):
                        img[j, :, 0] = gradient * (0.6 + np.random.rand()*0.2)
                        img[j, :, 1] = gradient * (0.6 + np.random.rand()*0.2)
                        img[j, :, 2] = gradient * (0.4 + np.random.rand()*0.2)
                    # Tambahkan refleksi metalik
                    img[::20, :, :] = np.minimum(img[::20, :, :] + 0.3, 1.0)
                        
                elif class_id == 2:  # Kertas - tekstur kertas yang lebih realistis
                    # Base warna kertas
                    base_color = [0.8 + np.random.rand()*0.2, 0.7 + np.random.rand()*0.2, 0.6 + np.random.rand()*0.2]
                    img = np.ones_like(img) * base_color
                    # Tambahkan noise untuk tekstur kertas
                    noise = np.random.normal(0, 0.15, img.shape)
                    img = np.clip(img + noise, 0, 1)
                    # Garis untuk kertas bergaris (kadang-kadang)
                    if i % 4 == 0:
                        img[::25, :, :] *= 0.7
                        
                elif class_id == 3:  # Kaca - pola transparan dan reflektif
                    # Base transparan
                    img = np.ones_like(img) * [0.85, 0.9, 0.95]
                    # Refleksi dan highlight
                    for _ in range(np.random.randint(3, 8)):
                        x = np.random.randint(0, self.img_height-20)
                        y = np.random.randint(0, self.img_width-20)
                        img[x:x+15, y:y+15, :] = [0.95, 0.98, 1.0]
                    # Shadow dan depth
                    img[::30, ::30, :] *= 0.6
                        
                elif class_id == 4:  # Organik - pola organik yang lebih natural
                    # Warna-warna organik yang bervariasi
                    organic_colors = [
                        [0.2, 0.6, 0.1],  # Hijau daun
                        [0.4, 0.2, 0.1],  # Coklat tanah
                        [0.8, 0.6, 0.2],  # Kuning buah
                        [0.6, 0.3, 0.1],  # Coklat kayu
                        [0.7, 0.1, 0.1]   # Merah buah
                    ]
                    base_color = organic_colors[i % len(organic_colors)]
                    img = np.ones_like(img) * base_color
                    
                    # Tambahkan variasi natural
                    for _ in range(np.random.randint(5, 15)):
                        x = np.random.randint(0, self.img_height-10)
                        y = np.random.randint(0, self.img_width-10)
                        size = np.random.randint(5, 15)
                        variation = np.random.rand(3) * 0.4 - 0.2
                        img[x:x+size, y:y+size, :] = np.clip(base_color + variation, 0, 1)
                        
                else:  # Kardus - pola kardus yang lebih realistis
                    # Warna dasar kardus
                    cardboard_color = [0.6 + np.random.rand()*0.2, 0.4 + np.random.rand()*0.2, 0.2 + np.random.rand()*0.1]
                    img = np.ones_like(img) * cardboard_color
                    
                    # Garis-garis kardus bergelombang
                    for line in range(0, self.img_height, 12):
                        if line < self.img_height:
                            img[line:line+2, :, :] *= 0.8
                    
                    # Lipatan dan tekstur
                    fold_pos = np.random.randint(20, self.img_width-20)
                    img[:, fold_pos:fold_pos+3, :] *= 0.6
                
                # Tambahkan noise realistis untuk semua kategori
                realistic_noise = np.random.normal(0, 0.05, img.shape)
                img = np.clip(img + realistic_noise, 0, 1)
                
                images.append(img)
                labels.append(class_id)
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"📊 Dataset dibuat: {len(images)} gambar")
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print("🧠 Membuat model CNN...")
        self.model = self.create_cnn_model()
        
        # Data augmentation yang lebih agresif
        train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.2,
            fill_mode='nearest'
        )
        
        # Validation data generator (tanpa augmentasi)
        val_datagen = ImageDataGenerator()
        
        # Callbacks yang lebih baik
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=8,
            min_lr=0.00001,
            verbose=1
        )
        
        # Learning rate scheduler
        from tensorflow.keras.callbacks import LearningRateScheduler
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * 0.95
        
        lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
        
        print("🚀 Memulai pelatihan model CNN...")
        
        # Training dengan epochs yang lebih banyak
        history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=16),  # Batch size lebih kecil
            steps_per_epoch=len(X_train) // 16,
            epochs=50,  # Lebih banyak epochs
            validation_data=val_datagen.flow(X_test, y_test, batch_size=16),
            validation_steps=len(X_test) // 16,
            callbacks=[early_stopping, reduce_lr, lr_scheduler],
            verbose=1
        )
        
        # Evaluasi
        test_loss, test_accuracy, test_top2_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n✅ Pelatihan selesai!")
        print(f"📈 Akurasi model pada data test: {test_accuracy:.4f}")
        print(f"📈 Top-2 Akurasi: {test_top2_accuracy:.4f}")
        
        return history, X_test, y_test

def main():
    """
    Fungsi utama program - Improved Version
    """
    print("🗑️ PROGRAM KLASIFIKASI SAMPAH - IMPROVED VERSION")
    print("=" * 60)
    print("🎯 Fitur Baru:")
    print("   - Model CNN yang lebih akurat dengan transfer learning")
    print("   - Ensemble prediction untuk akurasi lebih tinggi")
    print("   - UI realtime yang lebih baik dengan progress bar")
    print("   - Stabilisasi prediksi dan confidence threshold")
    print("   - Model save/load untuk konsistensi")
    print("=" * 60)
    
    # Inisialisasi classifier
    classifier = TrashClassifier()
    
    # Coba load model yang sudah ada
    model_file = "trash_classifier_model.h5"
    if os.path.exists(model_file):
        print(f"\n🔍 Ditemukan model tersimpan: {model_file}")
        if input("Mau load model yang sudah ada? (y/n): ").lower() == 'y':
            classifier.load_model(model_file)
    
    while True:
        print("\n" + "="*50)
        print("📋 MENU UTAMA:")
        print("="*50)
        print("1. 🧠 Latih model dengan data sampel (fitur)")
        print("2. 🖼️  Latih model CNN dengan data gambar (RECOMMENDED)")
        print("3. ⌨️  Klasifikasi sampah interaktif (input manual)")
        print("4. 🎥 Klasifikasi sampah realtime (webcam)")
        print("5. 💾 Simpan model saat ini")
        print("6. 📂 Load model dari file")
        print("7. ❌ Keluar")
        print("="*50)
        
        choice = input("\nMasukkan pilihan (1-7): ").strip()
        
        if choice == '1':
            print("\n🧠 Memulai pelatihan model dengan data fitur...")
            
            # Membuat data sampel
            features, labels = classifier.create_sample_data(num_samples=2000)
            
            # Melatih model
            history, X_test, y_test = classifier.train_model(features, labels, epochs=100)
            
            # Menampilkan grafik
            classifier.plot_training_history(history)
            classifier.plot_confusion_matrix(X_test, y_test)
            
            print("\n✅ Model berhasil dilatih!")
            
            # Tanya apakah mau simpan model
            if input("\nSimpan model? (y/n): ").lower() == 'y':
                classifier.save_model("feature_model.h5")
            
        elif choice == '2':
            print("\n🖼️ Memulai pelatihan model CNN dengan data gambar...")
            print("⚠️ Proses ini membutuhkan waktu lebih lama tapi menghasilkan akurasi lebih tinggi")
            
            if input("Lanjutkan? (y/n): ").lower() != 'y':
                continue
            
            # Melatih model CNN
            history, X_test, y_test = classifier.train_cnn_model_with_sample_images()
            
            # Menampilkan grafik
            classifier.plot_training_history(history)
            
            print("\n✅ Model CNN berhasil dilatih!")
            print("🎥 Sekarang Anda dapat menggunakan fitur klasifikasi realtime!")
            
            # Auto-save model CNN
            classifier.save_model("trash_classifier_model.h5")
            
        elif choice == '3':
            if classifier.model is None:
                print("\n⚠️ Model belum dilatih! Silakan latih model terlebih dahulu.")
                continue
            
            classifier.classify_trash_interactive()
            
            # Tanya apakah ingin melanjutkan
            if input("\nKlasifikasi sampah lain? (y/n): ").lower() != 'y':
                continue
                
        elif choice == '4':
            if classifier.model is None:
                print("\n⚠️ Model belum dilatih! Silakan latih model CNN terlebih dahulu (pilihan 2).")
                continue
            
            print("\n🎥 Memulai mode realtime...")
            print("💡 Tips untuk hasil terbaik:")
            print("   - Pastikan pencahayaan cukup baik")
            print("   - Dekatkan objek ke kamera (jarak 20-50cm)")
            print("   - Hindari background yang ramai")
            print("   - Tunggu confidence > 60% untuk hasil akurat")
            
            if input("\nLanjutkan ke mode realtime? (y/n): ").lower() == 'y':
                classifier.realtime_classification()
            
        elif choice == '5':
            classifier.save_model()
            
        elif choice == '6':
            filename = input("Masukkan nama file model (default: trash_classifier_model.h5): ").strip()
            if not filename:
                filename = "trash_classifier_model.h5"
            classifier.load_model(filename)
            
        elif choice == '7':
            print("\n🌍 Terima kasih telah menggunakan program klasifikasi sampah!")
            print("💚 Mari bersama-sama menjaga lingkungan yang bersih!")
            break
            
        else:
            print("\n❌ Pilihan tidak valid! Silakan pilih 1-7.")

if __name__ == "__main__":
    main()
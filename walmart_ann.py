import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc, confusion_matrix, log_loss
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')  # macOS uyarılarını önlemek için
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Keras modelini sklearn uyumlu hale getirmek için özel bir wrapper sınıfı ()
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        # Eğitilmiş model, tekrar eğitmeye gerek yok
        return self

    def predict(self, X):
        # Modelin tahminlerini döndür 0 veya 1 e    
        return (self.model.predict(X) > 0.5).astype(int)

    def predict_proba(self, X):
        # Olasılık tahminleri için
        return self.model.predict(X)

    def score(self, X, y):
        # Skor hesapla (doğruluk)
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# Veri ve model ön işleme fonksiyonu
def prepare_data_and_model():
    print("TensorFlow sürümü:", tf.__version__)
    print("GPU mevcut mu:", tf.config.list_physical_devices('GPU'))
    
    # Veri setini yükleme ve eksik verileri doldurma
    data = pd.read_csv("/Users/girayakbulut/keras_project/walmart_sales.csv")
    data['Product_Category'].fillna(data['Product_Category'].mode()[0], inplace=True)
    print("Eksik veri dolduruldu, veri seti:", data.shape)

    # Kategorik değişkenleri one-hot encoding ile kodlama
    categorical_columns = ['Gender', 'Age', 'Occupation', 'City_Category', 
                          'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category']
    data = pd.get_dummies(data, columns=categorical_columns)
    print("One-hot encoding sonrası veri seti :", data.shape)

    # Gereksiz sütunları kaldırma
    data.drop(['User_ID', 'Product_ID'], axis=1, inplace=True)

    # Hedef değişkeni oluşturma (Purchase > 8000 ise 1, değilse 0)
    data['Target'] = (data['Purchase'] > 8000).astype(int)
    print("Purchase için benzersiz değer:", np.unique(daxta['Target']))
    data.drop('Purchase', axis=1, inplace=True)

    # Özellikler (X) ve hedef değişkeni (y) ayırma
    X = data.drop('Target', axis=1)
    y = data['Target']

    # Veriyi normalizasyon (0-1 aralığına ölçeklendirme)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    print("Normalizasyon sonrası X aralığı:", X.min(), X.max())

    # Eğitim ve test setlerine bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print("Eğitim seti şekli:", X_train.shape, "Test seti şekli:", X_test.shape)

    # Sınıf dengesini kontrol etme
    print("Eğitim seti sınıf oranları:", np.bincount(y_train) / len(y_train))
    print("Test seti sınıf oranları:", np.bincount(y_test) / len(y_test))

    # Sınıf ağırlıklarını hesaplama (dengesiz veri için)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print("Sınıf agırlıkları:", class_weight_dict)

    # Yapay sinir ağı modelini tanımlama
    model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    print("Optimizer öğrenme oranı:", 0.0001)

    # Erken durdurma 
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Modelin eğitimi
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
    epochs=100, 
    batch_size=128, 
    class_weight=class_weight_dict,callbacks=[early_stopping], verbose=1)

    # Test seti üzerinde tahmin yapma
    y_pred_prob = model.predict(X_test)
    print("Tahminler aralıgı:", y_pred_prob.min(), y_pred_prob.max())
    
    return model, X_test, y_test, y_pred_prob, history, data

# Doğruluk ve kayıp grafiklerini fonksiyonu
def plot_accuracy_loss(history):
    print("Doğruluk ve kayıp grafik")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu', color='blue')
    plt.plot(history.history['val_accuracy'], label='Test Doğruluğu', color='orange')
    plt.title('Doğruluk Grafiği')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı', color='blue')
    plt.plot(history.history['val_loss'], label='Test Kaybı', color='orange')
    plt.title('Kayıp Grafiği')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/Users/girayakbulut/keras_project/accuracy_loss_plot.png')

# ROC eğrisini fonksiyonu
def plot_roc_curve(y_test, y_pred_prob):
    print("ROC eğrisi grafik")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC eğrisi (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (FPR)')
    plt.ylabel('Doğru Pozitif Oranı (TPR)')
    plt.title('ROC Eğrisi')
    plt.legend(loc="lower right")
    plt.savefig('/Users/girayakbulut/keras_project/roc_curve.png')

# Karışıklık matrisini fonksiyonu
def plot_confusion_matrix(y_test, y_pred_prob):
    print("Karışıklık matris grafik")
    y_pred = (y_pred_prob > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Karışıklık Matrisi')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.savefig('/Users/girayakbulut/keras_project/confusion_matrix.png')

# Özellik önem sıralamasını fonksiyonu
def plot_feature_importance(X_test, y_test, model, data):
    print("Özellik önem sıralamasını çiziyorum...")
    # sklearn uyumlu hale getirme
    wrapped_model = KerasClassifierWrapper(model)

    # Permütasyon önemini hesapla
    perm_importance = permutation_importance(estimator=wrapped_model, 
    X=X_test, y=y_test, scoring='accuracy', n_repeats=10)

    sorted_idx = perm_importance.importances_mean.argsort()

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [data.drop('Target', axis=1).columns[i] for i in sorted_idx])
    plt.xlabel('Permütasyon Önemi')
    plt.title('Özellik Önem Sıralaması')
    plt.savefig('/Users/girayakbulut/keras_project/feature_importance.png')

# Aktivasyon fonksiyonlarını fonksiyonu
def plot_activation_functions():
    print("Aktivasyon fonksiyonu grafik")
    x = np.linspace(-5, 5, 100)
    sigmoid = 1 / (1 + np.exp(-x))
    relu = np.maximum(0, x)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, sigmoid, label='Sigmoid', color='blue')
    plt.title('Sigmoid Aktivasyon Fonksiyonu')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, relu, label='ReLU', color='orange')
    plt.title('ReLU Aktivasyon Fonksiyonu')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/Users/girayakbulut/keras_project/activation_functions.png')

# Ana kod
# tüm fonksiyonları sırayla çalıştır ve sonuçları yazdır
if __name__ == "__main__":
    model, X_test, y_test, y_pred_prob, history, data = prepare_data_and_model()
    plot_accuracy_loss(history)
    plot_roc_curve(y_test, y_pred_prob)
    plot_confusion_matrix(y_test, y_pred_prob)
    plot_feature_importance(X_test, y_test, model, data)
    plot_activation_functions()

    # Kayıp ve doğruluk sonuçları
    print(f"Eğitim Doğruluğu: {history.history['accuracy'][-1]:.2f}")
    print(f"Test Doğruluğu: {history.history['val_accuracy'][-1]:.2f}")
    print(f"Eğitim Kaybı: {history.history['loss'][-1]:.2f}")
    print(f"Test Kaybı: {history.history['val_loss'][-1]:.2f}")
    manual_loss = log_loss(y_test, y_pred_prob)
    print("Manuel hesaplanan test kaybı:", manual_loss)
    print("Tahminlerin aralığı:", y_pred_prob.min(), y_pred_prob.max())
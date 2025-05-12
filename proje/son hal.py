import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import stats
from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# --- KLASÖR OLUŞTUR ---
os.makedirs("resimler", exist_ok=True)

# 1. VERİ YÜKLEME VE ÖN İŞLEME
df = pd.read_csv("data.csv", sep=';')
df.columns = df.columns.str.strip().str.replace('\\t', '', regex=True)

target_col = "Target"
df = df.dropna(subset=[target_col])

le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

for col in df.select_dtypes(include='object'):
    if col != target_col:
        df[col] = le.fit_transform(df[col].astype(str))

df.fillna(df.median(numeric_only=True), inplace=True)

# 2. ÖLÇEKLENDİRME
X = df.drop(columns=[target_col])
y = df[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. SMOTE ile sınıf dengeleme
print("\n--- SMOTE Uygulaması ---")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("Orijinal sınıf dağılımı:")
print(y.value_counts())
print("SMOTE sonrası sınıf dağılımı:")
print(pd.Series(y_resampled).value_counts())

# 4. VERİYİ BÖL
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 5. KNN İÇİN HİPERPARAMETRE OPTİMİZASYONU
print("\n--- KNN Hiperparametre Arama ---")
param_grid_knn = {'n_neighbors': list(range(1, 21))}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_knn.fit(X_train, y_train)

best_knn = grid_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test)

print("En iyi k:", grid_knn.best_params_['n_neighbors'])
print("KNN Doğruluk: {:.2f}%".format(accuracy_score(y_test, y_pred_knn) * 100))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# 6. ANN MODELİ (SMOTE sonrası, Dropout ve EarlyStopping ile)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

ann = Sequential()
ann.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
ann.add(Dropout(0.3))
ann.add(Dense(64, activation='relu'))
ann.add(Dropout(0.3))
ann.add(Dense(3, activation='softmax'))

ann.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

ann.fit(X_train, y_train_cat, validation_split=0.2,
        epochs=100, batch_size=32, verbose=0, callbacks=[early_stop])

loss, acc = ann.evaluate(X_test, y_test_cat, verbose=0)
print("\n--- ANN Doğruluk ---")
print("Test doğruluğu: {:.2f}%".format(acc * 100))

# 7. KARAR AĞACI + HİPERPARAMETRE TUNING
print("\n--- Karar Ağacı Hiperparametre Arama ---")
param_grid_dt = {'max_depth': range(3, 11)}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
grid_dt.fit(X_train, y_train)

best_dt = grid_dt.best_estimator_
y_pred_dt = best_dt.predict(X_test)

print("En iyi derinlik:", grid_dt.best_params_['max_depth'])
print("Karar Ağacı Doğruluk: {:.2f}%".format(accuracy_score(y_test, y_pred_dt) * 100))
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Karar Ağacı Görselleştirmesi
plt.figure(figsize=(18, 8))
plot_tree(best_dt, feature_names=df.drop(columns=["Target"]).columns,
          class_names=["Dropout", "Enrolled", "Graduate"], filled=True)
plt.title("Karar Ağacı Görselleştirmesi")
plt.savefig("resimler/karar_agaci.png")
plt.close()

# 8. GÖRSEL ANALİZLERİ GRAFİKLERE KAYDET
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["Curricular units 1st sem (grade)"],
                y=df["Curricular units 2nd sem (grade)"],
                hue=df["Target"], palette='Set1')
plt.title("1. Dönem vs 2. Dönem Notları")
plt.xlabel("1. Dönem Not Ortalaması")
plt.ylabel("2. Dönem Not Ortalaması")
plt.grid(True)
plt.tight_layout()
plt.savefig("resimler/not_karsilastirma.png")
plt.close()

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["Curricular units 1st sem (approved)"],
                y=df["Curricular units 2nd sem (approved)"],
                hue=df["Target"], palette='Set2')
plt.title("Geçilen Ders Sayıları Karşılaştırması")
plt.xlabel("1. Dönem Geçilen Dersler")
plt.ylabel("2. Dönem Geçilen Dersler")
plt.grid(True)
plt.tight_layout()
plt.savefig("resimler/ders_karsilastirma.png")
plt.close()

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["Admission grade"],
                y=df["Age at enrollment"],
                hue=df["Target"], palette='Set3')
plt.title("Kabul Notu vs Kayıt Yaşı")
plt.xlabel("Kabul Notu")
plt.ylabel("Yaş")
plt.grid(True)
plt.tight_layout()
plt.savefig("resimler/kabul_yas.png")
plt.close()

# 9. KORELASYON ISI HARİTASI
plt.figure(figsize=(14, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Özellikler Arası Korelasyon Isı Haritası")
plt.tight_layout()
plt.savefig("resimler/korelasyon_haritasi.png")
plt.close()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from imblearn.over_sampling import SMOTE


# 1. Load Data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancy', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'Pedigree', 'Age', 'Outcome']
data = pd.read_csv(url, names=column_names)

print("Dataset shape:", data.shape)
print("Class distribution:\n", data['Outcome'].value_counts())


# 2. Preprocessing
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[zero_not_accepted] = data[zero_not_accepted].replace(0, np.nan)
data.fillna(data.mean(), inplace=True)

X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 3. Train / Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=1, stratify=y
)


# 4. SMOTE — applied only to training data
smote = SMOTE(random_state=1)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"After SMOTE — Train size: {X_train_res.shape[0]}")


# 5. Class Weights
class_weights_array = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train_res), y=y_train_res
)
class_weight_dict = dict(enumerate(class_weights_array))

y_train_cat = to_categorical(y_train_res, num_classes=2)
y_test_cat  = to_categorical(y_test,      num_classes=2)


# 6. Model
model = Sequential([
    Dense(128, input_dim=8, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(2, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0005),
    metrics=['accuracy']
)

model.summary()


# 7. Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)


# 8. Training
history = model.fit(
    X_train_res, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=200,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

print(f"Training stopped at epoch: {len(history.history['loss'])}")


# 9. Threshold Tuning
y_pred_prob = model.predict(X_test)
y_true      = np.argmax(y_test_cat, axis=1)

fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

f1_scores = [f1_score(y_true, (y_pred_prob[:, 1] >= t).astype(int)) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores)]

print(f"Optimal threshold: {optimal_threshold:.4f}  |  AUC: {roc_auc:.4f}")

y_pred_optimal = (y_pred_prob[:, 1] >= optimal_threshold).astype(int)


# 10. Evaluation
print(classification_report(y_true, y_pred_optimal, target_names=['Healthy', 'Diabetes']))


# 11. Plots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('PIMA Diabetes ANN - Results', fontsize=16, fontweight='bold')

axes[0].plot(history.history['accuracy'],     label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Test Accuracy',  linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(loc='lower right')
axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].plot(fpr, tpr, color='purple', lw=3, label=f'ROC Curve (AUC = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[1].axvline(x=fpr[np.argmax(f1_scores)], color='red', linestyle=':', linewidth=1.5,
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
axes[1].set_title('ROC Analysis', fontsize=13, fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

cm = confusion_matrix(y_true, y_pred_optimal)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2], annot_kws={"size": 14})
axes[2].set_title(f'Confusion Matrix (Threshold = {optimal_threshold:.2f})', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Actual', fontsize=12)
axes[2].set_xlabel('Predicted', fontsize=12)
axes[2].set_xticklabels(['Healthy', 'Diabetes'], fontsize=11)
axes[2].set_yticklabels(['Healthy', 'Diabetes'], fontsize=11, rotation=0)

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
plt.show()

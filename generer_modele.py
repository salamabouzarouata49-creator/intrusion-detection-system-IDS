import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 

# Ensure Scapy cache files are written to a local writable directory.
base_dir = os.path.dirname(__file__)
cache_dir = os.path.join(base_dir, '.scapy_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ.setdefault('XDG_CACHE_HOME', cache_dir)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Importing Machine Learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Ignore warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. LOADING AND PREPARING DATA (NSL-KDD)
# =====================================================================

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 
    'outcome', 'level'
]

print("1. Loading data...")
train_path = os.path.join(base_dir, "train.txt")
test_path = os.path.join(base_dir, "test.txt")
train_df = pd.read_csv(train_path, header=None, names=columns)
test_df = pd.read_csv(test_path, header=None, names=columns)
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Selecting important features
selected_features = [
    'protocol_type', 'service', 'logged_in', 'is_host_login', 'is_guest_login', 'dst_bytes', 
    'num_shells', 'num_outbound_cmds', 'srv_count', 'diff_srv_rate', 'srv_diff_host_rate', 
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
    'dst_host_srv_serror_rate', 'outcome'
]

# ADDED: .copy() to avoid SettingWithCopyWarning
df = combined_df[selected_features].copy()

# =====================================================================
# 2. DATA PREPROCESSING
# =====================================================================
print("2. Data preprocessing (cleaning, encoding, standardization)...")

df.dropna(inplace=True)

# Target encoding (0 = normal, 1 = attack)
df['outcome'] = df['outcome'].apply(lambda x: 0 if x == 'normal' else 1)

# Text encoding to numbers
le_service = LabelEncoder()
df['service'] = le_service.fit_transform(df['service'].astype(str))

le_protocol_type = LabelEncoder()
df['protocol_type'] = le_protocol_type.fit_transform(df['protocol_type'].astype(str))

# Separation (X = features, y = target)
X = df.drop('outcome', axis=1)
y = df['outcome']

# Split (Train 80% / Test 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality reduction with PCA
pca = PCA(n_components=min(10, X_train_scaled.shape[1]))
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# =====================================================================
# 3. MODELS INITIALIZATION
# =====================================================================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42),
    "Decision Tree": DecisionTreeClassifier(criterion='gini', random_state=42),
    "Logistic Regression": LogisticRegression(solver='lbfgs', penalty='l2', C=1.0, max_iter=100, random_state=42),
    "KNN (K-Nearest Neighbors)": KNeighborsClassifier(n_neighbors=5, algorithm='auto', weights='uniform'),
    "Naive Bayes": GaussianNB(var_smoothing=1e-09),
}

results = {}

# =====================================================================
# 4. TRAINING AND EVALUATION
# =====================================================================
print("\n--- 3. Starting Machine Learning algorithms training ---")

for name, model in models.items():
    print(f"Training in progress: {name}...")
    
    # Training
    model.fit(X_train_pca, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_pca)
    y_test_pred = model.predict(X_test_pca)

    # Metrics calculation
    train_acc = accuracy_score(y_train, y_train_pred) * 100
    test_acc = accuracy_score(y_test, y_test_pred) * 100
    train_precision = precision_score(y_train, y_train_pred, average='binary') * 100
    test_precision = precision_score(y_test, y_test_pred, average='binary') * 100
    train_recall = recall_score(y_train, y_train_pred, average='binary') * 100
    test_recall = recall_score(y_test, y_test_pred, average='binary') * 100
    train_f1 = f1_score(y_train, y_train_pred, average='binary') * 100
    test_f1 = f1_score(y_test, y_test_pred, average='binary') * 100

    # Saving results
    results[name] = {
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Train Precision': train_precision,
        'Test Precision': test_precision,
        'Train Recall': train_recall,
        'Test Recall': test_recall,
        'Train F1': train_f1,
        'Test F1': test_f1,
        'Confusion Matrix': confusion_matrix(y_test, y_test_pred),
        'Model Object': model 
    }

# =====================================================================
# 5. COMPARATIVE RESULTS DISPLAY
# =====================================================================
print("\n" + "=" * 135)
print("  ALGORITHMS PERFORMANCE COMPARISON TABLE")
print("=" * 135)
print(f"{'Algorithm':<30} | {'Train Acc':<10} | {'Test Acc':<10} | {'Train Prec':<11} | {'Test Prec':<10} | {'Train Rec':<9} | {'Test Rec':<8} | {'Train F1':<8} | {'Test F1':<7}")
print("-" * 135)
for name, metrics in results.items():
    print(f"{name:<30} | {metrics['Train Accuracy']:.3f}% | {metrics['Test Accuracy']:.3f}% | {metrics['Train Precision']:.3f}% | {metrics['Test Precision']:.3f}% | {metrics['Train Recall']:.3f}% | {metrics['Test Recall']:.3f}% | {metrics['Train F1']:.3f}% | {metrics['Test F1']:.3f}%")
print("=" * 135)

# =====================================================================
# 6. SAVING MODELS (JOBLIB)
# =====================================================================
print("\n4. Saving models and preprocessing tools (exporting as .pkl)...")

best_model_name = "Random Forest"
best_model_trained = results[best_model_name]['Model Object']
joblib.dump(best_model_trained, os.path.join(base_dir, 'ids_model_random_forest.pkl'))

joblib.dump(scaler, os.path.join(base_dir, 'scaler.pkl'))
joblib.dump(pca, os.path.join(base_dir, 'pca_transformer.pkl'))
joblib.dump(le_protocol_type, os.path.join(base_dir, 'label_encoder_protocol.pkl'))
joblib.dump(le_service, os.path.join(base_dir, 'label_encoder_service.pkl'))

print("✅ Save successful! You can now use these files in another script.")

# =====================================================================
# 7. CONFUSION MATRIX (PLOT)
# =====================================================================
if best_model_name in results:
    plt.figure(figsize=(6, 5))
    sns.heatmap(results[best_model_name]['Confusion Matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    print(f"\n[Graph generated] Close the plot window to exit the program.")
    plt.show()
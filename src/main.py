import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split, KFold, LearningCurveDisplay, ShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[-_()\"#!@$%^&*{}?.,:]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return text
def token_text(text):
    text = text.lower()
    text = re.sub(r"[-_()\"#!@$%^&*{}?.,:]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = ViTokenizer.tokenize(text)
    return text

# Load và xử lý dữ liệu
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    df = data[['is_fake', 'normalized_content']].copy()
    df.rename(columns={'normalized_content': 'text', 'is_fake': 'label'}, inplace=True)

    mapping = {
        True: 1, False: 0,
        'True': 1, 'False': 0,
        'TRUE': 1, 'FALSE': 0
    }

    if 'label' in df.columns:
        df['label'] = df['label'].map(mapping)
        df.dropna(subset=['label', 'text'], inplace=True)
        df['label'] = df['label'].astype(int)

    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].astype(str).str.split().str.len() >= 20]
    df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    df = df.reset_index(drop=True)

    # Tokenize văn bản
    df['text'] = df['text'].apply(token_text)
    return df



# CÁC HÀM HIỂN THỊ / VẼ ĐỒ THỊ

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', labels=['Tin Thật', 'Tin Giả']):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()


def sconfusion_matrix(trains, pred):
    cm = confusion_matrix(trains, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()



# CHIA DỮ LIỆU
def split_data(df):
    X_train, X_vol, y_train, y_vol = train_test_split(
        df['text'], df['label'], test_size=0.3, shuffle=True, stratify=df['label'], random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_vol, y_vol, test_size=0.5, shuffle=True, stratify=y_vol, random_state=42
    )

    print("Số lượng mẫu huấn luyện:", len(X_train))
    print("Số lượng mẫu kiểm tra:", len(X_val))
    print("Số lượng mẫu kiểm thử:", len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test



# HUẤN LUYỆN & ĐÁNH GIÁ CƠ BẢN
def train_and_eval_baseline(X_train, X_val, y_train, y_val):
    tfid = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        norm='l2',
        max_df=0.9,
        min_df=2
    )

    X_train_tfid = tfid.fit_transform(X_train)
    X_val_tfid = tfid.transform(X_val)

    svc_model = SVC(kernel='linear', C=1.0, random_state=42)
    svc_model.fit(X_train_tfid, y_train)

    y_pred = svc_model.predict(X_val_tfid)
    print("--- BÁO CÁO TẬP VALIDATION ---")
    print(classification_report(y_val, y_pred))
    print(f"Độ Chính Xác : {accuracy_score(y_val, y_pred)*100:.2f}%\n")

    plot_confusion_matrix(y_val, y_pred, title='Validation Confusion Matrix')
    
    return tfid, svc_model

# K-FOLD

def evaluate_kfold(X_train, X_val, y_train, y_val):
    sion_cl = []
    fold_accuracies_cl = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), norm='l2', max_df=0.9, min_df=2)),
        ('model', SVC(kernel='linear', C=1.0, random_state=42))
    ])

    X_kf = pd.concat([X_train, X_val], ignore_index=True)
    y_kf = pd.concat([y_train, y_val], ignore_index=True)

    for trains, tests in kf.split(X_kf, y_kf):
        X_train_fold, X_test_fold = X_kf[trains], X_kf[tests]
        y_train_fold, y_test_fold = y_kf[trains], y_kf[tests]

        pipeline.fit(X_train_fold, y_train_fold)

        predictions = pipeline.predict(X_test_fold)
        score = accuracy_score(y_test_fold, predictions)
        sconfusion_matrix(y_test_fold, predictions)
        print(classification_report(y_test_fold, predictions))
        fold_accuracies_cl.append(score)
        sion_cl.append(confusion_matrix(y_test_fold, predictions))

    print(f"Độ chính xác của từng Fold: {fold_accuracies_cl}")
    print(f"Độ chính xác trung bình: {np.mean(fold_accuracies_cl)* 100:.2f}%")
    print(f"Độ lệch chuẩn: {np.std(fold_accuracies_cl):.4f}")

    # Vẽ Confusion Matrix trung bình
    mean_confusion_matrix_cl = np.round(np.mean(sion_cl, axis=0)).astype(int)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        mean_confusion_matrix_cl, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=['Tin Thật (0)', 'Tin Giả (1)'],
        yticklabels=['Tin Thật (0)', 'Tin Giả (1)']
    )
    plt.xlabel('Dự báo (Predicted Label)')
    plt.ylabel('Thực tế (True Label)')
    plt.title('Ma trận nhầm lẫn trung bình qua 5 Folds')
    plt.show()

    # Biểu đồ Accuracy qua các Fold
    folds = [f"Fold {i+1}" for i in range(len(fold_accuracies_cl))]
    plt.figure(figsize=(8, 5))
    plt.plot(folds, fold_accuracies_cl, marker='o', color='#1f77b4', linewidth=2, markersize=8, label='Accuracy')
    mean_acc = np.mean(fold_accuracies_cl)
    plt.axhline(y=mean_acc, color='r', linestyle='--', linewidth=1.5, label=f'Mean Accuracy ({mean_acc:.4f})')
    for i, acc in enumerate(fold_accuracies_cl):
        plt.annotate(f"{acc:.4f}", (folds[i], fold_accuracies_cl[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    plt.title('Độ chính xác (Accuracy) qua 5 Folds Cross Validation', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('K-Fold', fontsize=11, labelpad=10)
    plt.ylabel('Accuracy Score', fontsize=11, labelpad=10)
    plt.ylim(min(fold_accuracies_cl) - 0.03, max(fold_accuracies_cl) + 0.04)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    return X_kf, y_kf



# VẼ LEARNING CURVE
def plot_learning_curve(X_kf, y_kf, model_name="SVC"):
    fig, ax = plt.subplots(figsize=(8, 6))
    pipeline_text = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), norm='l2', max_df=0.9, min_df=2)),
        ('model', SVC(kernel='linear', C=1.0, random_state=42))
    ])
    common_params = {
        "X": X_kf,
        "y": y_kf,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Accuracy",
    }

    LearningCurveDisplay.from_estimator(pipeline_text, **common_params, ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Training Score", "Validation Score"])
    ax.set_title(f"Learning Curve for {model_name}", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()



# ĐÁNH GIÁ TẬP TEST
def eval_on_test(tfid, svc_model, X_test, y_test):
    X_test_tfid = tfid.transform(X_test)
    pred_test = svc_model.predict(X_test_tfid)

    print("--- BÁO CÁO TẬP TEST ---")
    print(classification_report(y_test, pred_test))
    print(f"Độ Chính Xác : {accuracy_score(y_test, pred_test)*100:.2f}%\n")

    plot_confusion_matrix(y_test, pred_test, title='Test Confusion Matrix')


#========================================================
file_path = '/content/drive/MyDrive/KPDL/YTe.csv'

#Load dữ liệu
df = load_and_preprocess_data(file_path)
print("Load thành công !")
print('--*--' * 20)
# Chia dữ liệu
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
print('--*--' * 20)
#Huấn luyện mô hình đơn lẻ & Đánh giá Validation
tfid, svc_model = train_and_eval_baseline(X_train, X_val, y_train, y_val)
print('--*--' * 20)
#Đánh giá Cross-Validation (K-Fold)
X_kf, y_kf = evaluate_kfold(X_train, X_val, y_train, y_val)
print('--*--' * 20)
#Vẽ đường học (Learning Curve)
plot_learning_curve(X_kf, y_kf, model_name=svc_model.__class__.__name__)
print('--*--' * 20)
#Đánh giá cuối cùng trên tập Test
eval_on_test(tfid, svc_model, X_test, y_test)

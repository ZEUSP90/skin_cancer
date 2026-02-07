"""
EVALUATION SCRIPT FOR SKIN CANCER DETECTION MODEL
==================================================

This script provides comprehensive evaluation tools for  trained model:
- Training history visualization
- Confusion matrix
- ROC curves
- Classification reports
- Performance metrics

Compatible with InceptionV3 model (299x299 images)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
import os


def plot_training_history(history_path, save_path=None):
    """
    Plot training metrics over epochs
    
    Args:
        history_path: Path to training_history.json file
        save_path: Where to save the plot (optional)
    
    Creates plots for:
        - Accuracy (training vs validation)
        - Loss (training vs validation)
        - AUC, Precision, Recall
    """
    print(f"\n📊 Loading training history from: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    metrics = ['accuracy', 'loss', 'auc', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if metric in history:
            axes[idx].plot(history[metric], label=f'Training {metric}', linewidth=2)
            axes[idx].plot(history[f'val_{metric}'], label=f'Validation {metric}', linewidth=2)
            axes[idx].set_title(f'Model {metric.capitalize()}', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Epoch', fontsize=11)
            axes[idx].set_ylabel(metric.capitalize(), fontsize=11)
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)
    
    # Hide the last empty subplot
    axes[-1].axis('off')
    
    plt.suptitle('Training History - InceptionV3 Skin Cancer Detection', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Where to save the plot (optional)
    
    Shows:
        - How many predictions were correct (diagonal)
        - Where the model makes mistakes (off-diagonal)
    """
    print("\n🎯 Generating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Predictions'})
    plt.title('Confusion Matrix - InceptionV3 Model', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm


def plot_roc_curves(y_true, y_pred_proba, class_names, save_path=None):
    """
    Plot ROC curves for each class
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        class_names: List of class names
        save_path: Where to save the plot (optional)
    
    ROC Curve shows:
        - Model's ability to distinguish between classes
        - AUC closer to 1.0 = better performance
        - AUC = 0.5 = random guessing
    """
    print("\n📈 Generating ROC curves...")
    
    n_classes = len(class_names)
    
    # Binarize the labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Calculate ROC curve for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i, color in enumerate(colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves - Multi-class Classification', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves saved to: {save_path}")
    
    plt.show()
    
    return roc_auc


def generate_classification_report(y_true, y_pred, class_names, save_path=None):
    """
    Generate detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Where to save JSON report (optional)
    
    Provides per-class metrics:
        - Precision: Of predictions for this class, how many were correct?
        - Recall: Of actual cases, how many did we find?
        - F1-score: Harmonic mean of precision and recall
        - Support: Number of actual cases
    """
    print("\n📋 Generating classification report...")
    
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   digits=4, output_dict=True)
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"✅ Classification report saved to: {save_path}")
    
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("="*80)
    
    return report


def analyze_model_predictions(model, test_generator, class_names, output_dir='results/'):
    """
    Complete model evaluation pipeline
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        class_names: List of class names
        output_dir: Directory to save results
    
    Generates:
        - Confusion matrix plot
        - ROC curves plot
        - Classification report (JSON)
        - Metrics summary (JSON)
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    print("\n🔬 Running model predictions on test set...")
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    print(f"\n✅ Predictions completed!")
    print(f"   Total test samples: {len(y_true)}")
    print(f"   Number of classes: {len(class_names)}")
    
    # Generate visualizations and reports
    cm = plot_confusion_matrix(y_true, y_pred, class_names, 
                               save_path=os.path.join(output_dir, 'confusion_matrix.png'))
    
    roc_auc = plot_roc_curves(y_true, predictions, class_names,
                              save_path=os.path.join(output_dir, 'roc_curves.png'))
    
    report = generate_classification_report(y_true, y_pred, class_names,
                                           save_path=os.path.join(output_dir, 'classification_report.json'))
    
    # Create metrics summary
    metrics_summary = {
        'model_type': 'InceptionV3',
        'test_samples': int(len(y_true)),
        'num_classes': len(class_names),
        'overall_accuracy': float(report['accuracy']),
        'macro_avg_precision': float(report['macro avg']['precision']),
        'macro_avg_recall': float(report['macro avg']['recall']),
        'macro_avg_f1': float(report['macro avg']['f1-score']),
        'weighted_avg_precision': float(report['weighted avg']['precision']),
        'weighted_avg_recall': float(report['weighted avg']['recall']),
        'weighted_avg_f1': float(report['weighted avg']['f1-score']),
        'per_class_auc': {class_names[i]: float(roc_auc[i]) for i in range(len(class_names))},
        'per_class_metrics': {
            class_names[i]: {
                'precision': float(report[class_names[i]]['precision']),
                'recall': float(report[class_names[i]]['recall']),
                'f1-score': float(report[class_names[i]]['f1-score']),
                'support': int(report[class_names[i]]['support'])
            }
            for i in range(len(class_names))
        }
    }
    
    # Save metrics summary
    summary_path = os.path.join(output_dir, 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    print(f"\n✅ Metrics summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Overall Accuracy: {metrics_summary['overall_accuracy']:.4f}")
    print(f"Macro Avg F1-Score: {metrics_summary['macro_avg_f1']:.4f}")
    print(f"Weighted Avg F1-Score: {metrics_summary['weighted_avg_f1']:.4f}")
    print("\nPer-Class AUC Scores:")
    for class_name, auc_score in metrics_summary['per_class_auc'].items():
        print(f"  {class_name}: {auc_score:.4f}")
    print("="*80)
    
    return metrics_summary


def prepare_dataset_split(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test sets
    
    Args:
        data_dir: Directory containing class folders with images
        train_ratio: Fraction for training (default: 0.7 = 70%)
        val_ratio: Fraction for validation (default: 0.15 = 15%)
        test_ratio: Fraction for testing (default: 0.15 = 15%)
    
    Creates:
        split_dataset/
        ├── train/
        ├── val/
        └── test/
    """
    import shutil
    from pathlib import Path
    
    print("\n" + "="*80)
    print("DATASET SPLITTING")
    print("="*80)
    print(f"Ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")
    
    output_dir = os.path.join(os.path.dirname(data_dir), 'split_dataset')
    
    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Get class folders
    classes = [d for d in os.listdir(data_dir) 
              if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"\nFound {len(classes)} classes: {classes}")
    print("\nSplitting images...")
    
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle images
        np.random.shuffle(images)
        
        # Calculate split sizes
        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        # Copy images to respective folders
        for split, split_images in [('train', train_images), 
                                     ('val', val_images), 
                                     ('test', test_images)]:
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copy2(src, dst)
        
        print(f"  {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    print(f"\n✅ Dataset split completed!")
    print(f"✅ Output saved to: {output_dir}")
    print("="*80)
    
    return output_dir


# Example usage
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        EVALUATION TOOLS FOR SKIN CANCER DETECTION            ║
    ║                   InceptionV3 Model                          ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Available Functions:
    
    1. plot_training_history()
       - Visualize training metrics over epochs
       
    2. analyze_model_predictions()
       - Complete evaluation: confusion matrix, ROC curves, reports
       
    3. prepare_dataset_split()
       - Split your dataset into train/val/test
    
    Example Usage:
    -------------
    
    from evaluation import *
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Load model
    model = keras.models.load_model('models/best_model_inceptionv3_finetuned.h5')
    
    # Prepare test data
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        'data/test',
        target_size=(299, 299),  # InceptionV3 size!
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Run evaluation
    class_names = list(test_gen.class_indices.keys())
    results = analyze_model_predictions(model, test_gen, class_names, 'results/')
    
    # Plot training history
    plot_training_history('models/training_history_inceptionv3_finetuned.json', 
                         'results/training_history.png')
    """)

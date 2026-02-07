import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import InceptionV3
import json
import os


class SkinCancerDetector:
    """
    Skin Cancer Detection using InceptionV3 Transfer Learning
    
    InceptionV3 is particularly good for medical imaging because:
    - Multi-scale feature extraction (1x1, 3x3, 5x5 convolutions in parallel)
    - Efficient computation with factorized convolutions
    - Better at capturing fine-grained details in skin lesions
    """
    
    def __init__(self, img_height=299, img_width=299, num_classes=7):
        """
        Initialize the detector
        
        Args:
            img_height: InceptionV3 expects 299x299 (not 224x224)
            img_width: InceptionV3 expects 299x299
            num_classes: Number of skin cancer types (default: 7)
        """
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """
        Build the InceptionV3-based model architecture
        
        Architecture:
        1. InceptionV3 base (frozen initially)
        2. Global Average Pooling
        3. Dense layers with dropout for classification
        4. Softmax output for multi-class classification
        """
        print("\n🏗️  Building InceptionV3 architecture...")
        
        # Load InceptionV3 pre-trained on ImageNet
        base_model = InceptionV3(
            include_top=False,  # Exclude the final classification layer
            weights='imagenet',  # Use ImageNet pre-trained weights
            input_shape=(self.img_height, self.img_width, 3)
        )
        
        # Freeze the base model initially (Transfer Learning Phase 1)
        base_model.trainable = False
        
        # Build the custom classification head
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        
        # Pass through InceptionV3
        x = base_model(inputs, training=False)
        
        # Global Average Pooling - reduces spatial dimensions
        x = layers.GlobalAveragePooling2D()(x)
        
        # Batch Normalization - stabilizes learning
        x = layers.BatchNormalization()(x)
        
        # First Dense layer with Dropout
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        
        # Second Dense layer with Dropout
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.01))(x)
        
        # Final Dropout
        x = layers.Dropout(0.2)(x)
        
        # Output layer - softmax for multi-class classification
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        self.model = keras.Model(inputs, outputs, name='InceptionV3_SkinCancer')
        
        # Compile the model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
            ]
        )
        
        print(f"✅ Model built successfully!")
        print(f"✅ Total parameters: {self.model.count_params():,}")
        print(f"✅ Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        return self.model
    
    def get_data_generators(self, train_dir, val_dir, batch_size=32):
        """
        Create data generators with augmentation
        
        Data Augmentation helps prevent overfitting by creating variations:
        - Rotations: skin lesions can appear at any angle
        - Flips: lesions have no inherent orientation
        - Zoom/Shift: simulates different camera distances
        """
        print("\n📊 Setting up data generators...")
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,              # Normalize pixel values to [0,1]
            rotation_range=40,            # Random rotations up to 40 degrees
            width_shift_range=0.2,        # Random horizontal shifts
            height_shift_range=0.2,       # Random vertical shifts
            shear_range=0.2,              # Shear transformations
            zoom_range=0.2,               # Random zoom
            horizontal_flip=True,         # Random horizontal flips
            vertical_flip=True,           # Random vertical flips
            fill_mode='nearest',          # Fill strategy for new pixels
            brightness_range=[0.8, 1.2]   # Random brightness adjustment
        )
        
        # Validation data - only rescaling (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator, epochs=30, model_path='models/'):
        """
        Training Phase 1: Transfer Learning with frozen base
        
        Strategy: Train only the custom classification head while keeping
        InceptionV3 weights frozen. This is faster and prevents overfitting.
        """
        os.makedirs(model_path, exist_ok=True)
        
        print(f"\n🎯 Starting Training Phase 1...")
        print(f"   - Base model: FROZEN")
        print(f"   - Training: Custom classification head only")
        print(f"   - Epochs: {epochs}")
        
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(model_path, 'best_model_inceptionv3.h5'),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.save_training_history(history, model_path)
        
        return history
    
    def fine_tune(self, train_generator, val_generator, epochs=30, model_path='models/'):
        """
        Training Phase 2: Fine-tuning with unfrozen top layers
        
        Strategy: Unfreeze the top layers of InceptionV3 and train with
        a very low learning rate to adapt features to our specific task.
        """
        print(f"\n🔧 Starting Fine-Tuning Phase 2...")
        
        # Get the InceptionV3 base model
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Freeze all layers except the top 30
        # InceptionV3 has mixed layers - we unfreeze the higher-level ones
        print(f"   - Total layers in InceptionV3: {len(base_model.layers)}")
        
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"   - Unfrozen layers: {trainable_count}")
        print(f"   - Using very low learning rate: 1e-5")
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
            ]
        )
        
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(model_path, 'best_model_inceptionv3_finetuned.h5'),
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.save_training_history(history, model_path, 'finetuned')
        
        return history
    
    def save_training_history(self, history, model_path, suffix=''):
        """Save training history to JSON file"""
        history_dict = {key: [float(val) for val in values] 
                       for key, values in history.history.items()}
        
        filename = f'training_history_inceptionv3{"_" + suffix if suffix else ""}.json'
        with open(os.path.join(model_path, filename), 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"✅ Training history saved to: {os.path.join(model_path, filename)}")
    
    def load_model(self, model_path):
        """Load a saved model"""
        self.model = keras.models.load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
        return self.model
    
    def predict(self, image_array):
        """
        Make predictions on new images
        
        Args:
            image_array: Image array or batch of images
            
        Returns:
            Prediction probabilities for each class
        """
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        predictions = self.model.predict(image_array)
        return predictions
    
    def evaluate(self, test_generator):
        """
        Evaluate model on test set
        
        Returns:
            Dictionary of metrics
        """
        results = self.model.evaluate(test_generator)
        metrics = dict(zip(self.model.metrics_names, results))
        return metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("SKIN CANCER DETECTION MODEL - InceptionV3")
    print("=" * 80)
    
    print("\n🏥 Initializing Skin Cancer Detector...")
    # Note: InceptionV3 uses 299x299 images (not 224x224)
    detector = SkinCancerDetector(img_height=299, img_width=299, num_classes=7)
    
    print("\n🔬 Building InceptionV3 architecture...")
    model = detector.build_model()
    
    # Print model summary
    print("\n📋 Model Summary:")
    model.summary()
    
    print("\n" + "=" * 80)
    print("CONFIGURATION - UPDATE THESE PATHS")
    print("=" * 80)
    
    # ⚠️ CHANGE THESE PATHS TO YOUR DATA DIRECTORIES
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    BATCH_SIZE = 16  # Reduced for InceptionV3 (larger model)
    EPOCHS_PHASE1 = 30
    EPOCHS_PHASE2 = 30
    MODEL_SAVE_PATH = 'models/'
    
    print(f"\n📁 Training directory: {TRAIN_DIR}")
    print(f"📁 Validation directory: {VAL_DIR}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔄 Phase 1 epochs: {EPOCHS_PHASE1}")
    print(f"🔄 Phase 2 epochs: {EPOCHS_PHASE2}")
    print(f"💾 Model save path: {MODEL_SAVE_PATH}")
    
    # Validate directories exist
    if not os.path.exists(TRAIN_DIR):
        print(f"\n❌ ERROR: Training directory not found: {TRAIN_DIR}")
        print("Please update TRAIN_DIR or create the folder structure.")
        print("\nExpected structure:")
        print("data/")
        print("├── train/")
        print("│   ├── class1/")
        print("│   ├── class2/")
        print("│   └── ...")
        print("└── val/")
        print("    ├── class1/")
        print("    ├── class2/")
        print("    └── ...")
        exit(1)
    
    if not os.path.exists(VAL_DIR):
        print(f"\n❌ ERROR: Validation directory not found: {VAL_DIR}")
        print("Please update VAL_DIR or create the folder structure.")
        exit(1)
    
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    train_gen, val_gen = detector.get_data_generators(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        batch_size=BATCH_SIZE
    )
    
    print(f"\n✅ Training samples: {train_gen.n}")
    print(f"✅ Validation samples: {val_gen.n}")
    print(f"✅ Number of classes: {len(train_gen.class_indices)}")
    print(f"✅ Class names: {list(train_gen.class_indices.keys())}")
    
    print("\n" + "=" * 80)
    print("PHASE 1: TRANSFER LEARNING")
    print("=" * 80)
    print("🔒 InceptionV3 base: FROZEN")
    print("🎯 Training: Custom classification head only")
    print(f"⏱️  This will take several hours...")
    
    history_phase1 = detector.train(
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=EPOCHS_PHASE1,
        model_path=MODEL_SAVE_PATH
    )
    
    print(f"\n✅ Phase 1 completed!")
    print(f"💾 Best model saved to: {MODEL_SAVE_PATH}best_model_inceptionv3.h5")
    
    print("\n" + "=" * 80)
    print("PHASE 2: FINE-TUNING")
    print("=" * 80)
    print("🔓 Unfreezing top 30 layers of InceptionV3")
    print("📉 Using very low learning rate (1e-5)")
    
    # Reset generators
    train_gen.reset()
    val_gen.reset()
    
    history_phase2 = detector.fine_tune(
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=EPOCHS_PHASE2,
        model_path=MODEL_SAVE_PATH
    )
    
    print(f"\n✅ Phase 2 completed!")
    print(f"💾 Final model saved to: {MODEL_SAVE_PATH}best_model_inceptionv3_finetuned.h5")
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print("\n📊 Next steps:")
    print(f"   1. Review training history: {MODEL_SAVE_PATH}training_history_inceptionv3.json")
    print(f"   2. Make predictions on new images")
    print(f"   3. Evaluate on test set")
    
    print("\n💡 To make predictions:")
    print(f"   python predict.py {MODEL_SAVE_PATH}best_model_inceptionv3_finetuned.h5 <image_path>")
    
    print("\n" + "=" * 80)

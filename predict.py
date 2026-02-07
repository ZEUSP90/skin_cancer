import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json
import os


class SkinCancerPredictor:
    def __init__(self, model_path, class_names=None):
        self.model = keras.models.load_model(model_path)
        
        if class_names is None:
            self.class_names = [
                'Melanoma',
                'Basal Cell Carcinoma',
                'Actinic Keratosis',
                'Benign Keratosis',
                'Dermatofibroma',
                'Melanocytic Nevus',
                'Vascular Lesion'
            ]
        else:
            self.class_names = class_names
    
    def preprocess_image(self, image_path, target_size=(299, 299)):
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_single_image(self, image_path, top_k=3):
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array, verbose=0)
        
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'class': self.class_names[idx],
                'confidence': float(predictions[0][idx]),
                'percentage': float(predictions[0][idx] * 100)
            })
        
        return results
    
    def predict_batch(self, image_paths, batch_size=32):
        all_results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_arrays = [self.preprocess_image(path) for path in batch_paths]
            batch_arrays = np.vstack(batch_arrays)
            
            predictions = self.model.predict(batch_arrays, verbose=0)
            
            for j, pred in enumerate(predictions):
                top_idx = np.argmax(pred)
                all_results.append({
                    'image': batch_paths[j],
                    'predicted_class': self.class_names[top_idx],
                    'confidence': float(pred[top_idx]),
                    'all_probabilities': {
                        self.class_names[k]: float(pred[k]) 
                        for k in range(len(self.class_names))
                    }
                })
        
        return all_results
    
    def get_risk_assessment(self, prediction_results):
        primary_prediction = prediction_results[0]
        confidence = primary_prediction['confidence']
        predicted_class = primary_prediction['class']
        
        malignant_types = ['Melanoma', 'Basal Cell Carcinoma', 'Actinic Keratosis']
        
        if predicted_class in malignant_types:
            if confidence > 0.8:
                risk_level = 'HIGH'
                recommendation = 'Immediate consultation with dermatologist recommended'
            elif confidence > 0.6:
                risk_level = 'MODERATE'
                recommendation = 'Schedule appointment with dermatologist soon'
            else:
                risk_level = 'LOW-MODERATE'
                recommendation = 'Consider consulting dermatologist for evaluation'
        else:
            if confidence > 0.8:
                risk_level = 'LOW'
                recommendation = 'Lesion appears benign, monitor for changes'
            else:
                risk_level = 'UNCERTAIN'
                recommendation = 'Professional evaluation recommended for confirmation'
        
        return {
            'risk_level': risk_level,
            'recommendation': recommendation,
            'confidence': confidence,
            'note': 'This is a screening tool only. Always consult healthcare professionals for diagnosis.'
        }
    
    def generate_report(self, image_path, output_path=None):
        predictions = self.predict_single_image(image_path, top_k=3)
        risk_assessment = self.get_risk_assessment(predictions)
        
        report = {
            'image_analyzed': image_path,
            'predictions': predictions,
            'risk_assessment': risk_assessment,
            'disclaimer': 'This AI model is for screening purposes only and should not replace professional medical diagnosis.'
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
        
        return report


def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_path> <image_path>")
        print("Example: python predict.py models/best_model_finetuned.h5 test_image.jpg")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    
    predictor = SkinCancerPredictor(model_path)
    
    print(f"\nAnalyzing image: {image_path}")
    print("=" * 60)
    
    report = predictor.generate_report(image_path)
    
    print("\nTop Predictions:")
    for i, pred in enumerate(report['predictions'], 1):
        print(f"{i}. {pred['class']}: {pred['percentage']:.2f}%")
    
    print("\nRisk Assessment:")
    print(f"Risk Level: {report['risk_assessment']['risk_level']}")
    print(f"Recommendation: {report['risk_assessment']['recommendation']}")
    
    print(f"\n{report['disclaimer']}")
    print("=" * 60)
    
    output_json = image_path.rsplit('.', 1)[0] + '_analysis.json'
    predictor.generate_report(image_path, output_json)
    print(f"\nDetailed report saved to: {output_json}")


if __name__ == "__main__":
    main()

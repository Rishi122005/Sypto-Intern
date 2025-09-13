import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Disable eager execution for a potential slight performance boost,
# though it's often not needed for simple scripts.
tf.config.run_functions_eagerly(False)

class CropMonitoringSystem:
    def __init__(self, model_source="tensorflow"):
        """
        Initialize the crop monitoring system
        Args:
            model_source: "tensorflow" or "huggingface"
        """
        self.model_source = model_source
        self.feature_extractor = None
        self.disease_model = None
        self.advisory_dict = self._load_advisory_dict()
        
        # Initialize models
        self._load_pretrained_models()
        
        # Create directories for storing data
        self.data_dir = Path("crop_data")
        self.data_dir.mkdir(exist_ok=True)
        
    def _load_advisory_dict(self):
        """Load predefined advisory messages"""
        return {
            "healthy": "No action needed, crop is healthy. Continue current care routine.",
            "pest": "Pest detected! Apply neem oil spray in the evening. Monitor for 3-5 days.",
            "disease": "Disease symptoms detected! Apply appropriate fungicide within 2 days. Isolate affected plants if possible.",
            "stress": "Plant stress detected (possibly nutrient deficiency). Check soil moisture and consider balanced fertilizer application.",
            "unknown": "Unable to classify. Consider consulting a local agricultural expert."
        }
    
    def _load_pretrained_models(self):
        """Load pretrained models for feature extraction and disease detection"""
        try:
            if self.model_source == "tensorflow":
                # Load EfficientNetB0 for feature extraction (growth tracking)
                self.feature_extractor = EfficientNetB0(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg',
                    input_shape=(224, 224, 3)
                )
                
                # Simple disease classification model (using EfficientNet + custom head)
                # In production, this would be fine-tuned on a plant disease dataset.
                base_model = EfficientNetB0(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg',
                    input_shape=(224, 224, 3)
                )
                
                # Add classification head for 4 classes (healthy, pest, disease, stress)
                inputs = tf.keras.Input(shape=(224, 224, 3))
                x = base_model(inputs, training=False)
                x = tf.keras.layers.Dropout(0.2)(x)
                outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
                
                self.disease_model = tf.keras.Model(inputs, outputs)
                
                # Initialize with random weights (in production, load fine-tuned weights)
                self.disease_model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                print("✅ TensorFlow models loaded successfully")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("📝 Note: In production, ensure you have an internet connection for downloading pretrained weights.")
    
    def preprocess_image(self, img_path):
        """
        Preprocess image for model input
        Args:
            img_path: Path to image file
        Returns:
            Preprocessed image array
        """
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array
        except Exception as e:
            print(f"❌ Error preprocessing image {img_path}: {e}")
            return None
    
    def extract_features(self, img_path):
        """
        Extract features from image using a pretrained feature extractor
        Args:
            img_path: Path to image file
        Returns:
            Feature vector
        """
        if self.feature_extractor is None:
            print("❌ Feature extractor not loaded")
            return None
            
        img_array = self.preprocess_image(img_path)
        if img_array is None:
            return None
            
        try:
            features = self.feature_extractor.predict(img_array, verbose=0)
            return features.flatten()
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None
    
    def analyze_growth(self, new_photo_path, quadrant_id, past_photos_dir=None):
        """
        Analyze crop growth by comparing a new photo with past photos
        Args:
            new_photo_path: Path to new crop photo
            quadrant_id: ID of the quadrant (1-4)
            past_photos_dir: Directory containing past photos
        Returns:
            Growth analysis results
        """
        print(f"📊 Analyzing growth for quadrant {quadrant_id}...")
        
        # Extract features from new photo
        new_features = self.extract_features(new_photo_path)
        if new_features is None:
            return {"error": "Failed to process new photo"}
        
        # Save current photo features
        quadrant_dir = self.data_dir / f"quadrant_{quadrant_id}"
        quadrant_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_file = quadrant_dir / f"features_{timestamp}.json"
        
        # Load past features for comparison
        past_features_files = list(quadrant_dir.glob("features_*.npy"))
        
        if len(past_features_files) == 0:
            # First photo for this quadrant
            np.save(feature_file.with_suffix('.npy'), new_features)
            return {
                "status": "baseline",
                "message": "First photo recorded as baseline for future comparisons",
                "quadrant": quadrant_id,
                "timestamp": timestamp
            }
        
        # Compare with the most recent past photo
        latest_past_file = sorted(past_features_files)[-1]
        past_features = np.load(latest_past_file)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            new_features.reshape(1, -1),
            past_features.reshape(1, -1)
        )[0][0]
        
        # Save new features
        np.save(feature_file.with_suffix('.npy'), new_features)
        
        # Interpret similarity score
        if similarity > 0.95:
            growth_status = "minimal_change"
            message = "Minimal growth detected. Continue monitoring."
        elif similarity > 0.85:
            growth_status = "normal_growth"
            message = "Normal growth progression detected."
        elif similarity > 0.70:
            growth_status = "significant_growth"
            message = "Good growth progress! Crops are developing well."
        else:
            growth_status = "major_change"
            message = "Major changes detected. Verify if this is expected growth or potential issues."
        
        return {
            "status": growth_status,
            "similarity_score": float(similarity),
            "message": message,
            "quadrant": quadrant_id,
            "timestamp": timestamp,
            "comparison_with": latest_past_file.stem
        }
    
    def predict_disease(self, photo_path):
        """
        Predict pest/disease status from crop photo
        Args:
            photo_path: Path to crop photo
        Returns:
            Disease prediction results
        """
        print("🔍 Analyzing for pests and diseases...")
        
        if self.disease_model is None:
            print("❌ Disease model not loaded")
            return {"error": "Disease model not available"}
        
        img_array = self.preprocess_image(photo_path)
        if img_array is None:
            return {"error": "Failed to process image"}
        
        try:
            # Get prediction
            predictions = self.disease_model.predict(img_array, verbose=0)
            
            # Class labels
            class_labels = ["healthy", "pest", "disease", "stress"]
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = class_labels[predicted_class_idx]
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_probabilities": {
                    label: float(prob) for label, prob in zip(class_labels, predictions[0])
                }
            }
            
        except Exception as e:
            print(f"❌ Error in disease prediction: {e}")
            return {"error": f"Prediction failed: {e}"}
    
    def generate_advisory(self, prediction_result):
        """
        Generate an advisory message based on prediction
        Args:
            prediction_result: Result from predict_disease()
        Returns:
            Advisory message
        """
        if "error" in prediction_result:
            return self.advisory_dict["unknown"]
        
        predicted_class = prediction_result.get("predicted_class", "unknown")
        confidence = prediction_result.get("confidence", 0)
        
        base_advisory = self.advisory_dict.get(predicted_class, self.advisory_dict["unknown"])
        
        # Add confidence information
        if confidence > 0.8:
            confidence_note = f"High confidence ({confidence:.1%})"
        elif confidence > 0.6:
            confidence_note = f"Moderate confidence ({confidence:.1%})"
        else:
            confidence_note = f"Low confidence ({confidence:.1%}) - consider getting a second opinion"
        
        return f"{base_advisory} [{confidence_note}]"
    
    def full_analysis(self, photo_path, quadrant_id):
        """
        Perform complete analysis: growth tracking + disease detection + advisory
        Args:
            photo_path: Path to crop photo
            quadrant_id: Quadrant ID (1-4)
        Returns:
            Complete analysis results
        """
        print(f"🌱 Starting full analysis for quadrant {quadrant_id}...")
        print(f"📸 Processing photo: {photo_path}")
        
        # Growth analysis
        growth_result = self.analyze_growth(photo_path, quadrant_id)
        
        # Disease prediction
        disease_result = self.predict_disease(photo_path)
        
        # Generate advisory
        advisory = self.generate_advisory(disease_result)
        
        # Compile results
        complete_result = {
            "timestamp": datetime.now().isoformat(),
            "quadrant_id": quadrant_id,
            "photo_path": str(photo_path),
            "growth_analysis": growth_result,
            "disease_analysis": disease_result,
            "advisory": advisory,
            "system_status": "analysis_complete"
        }
        
        # Save results
        results_file = self.data_dir / f"analysis_results_{quadrant_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(complete_result, f, indent=2)
        
        return complete_result

if __name__ == "__main__":
    # Initialize the system
    cms = CropMonitoringSystem(model_source="tensorflow")

    # Path to your test image and a quadrant ID
    test_image_path = "test_crop.jpg"
    quadrant_id = 1

    # Check if the test image exists before running
    if os.path.exists(test_image_path):
        print(f"🚀 Starting full analysis for quadrant {quadrant_id} with image {test_image_path}...")
        results = cms.full_analysis(test_image_path, quadrant_id)
        print("\n✅ Analysis Complete. Results:")
        print(json.dumps(results, indent=2))
    else:
        print(f"❌ Error: Test image '{test_image_path}' not found. Please place an image with this name in the project directory.")

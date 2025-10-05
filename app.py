#!/usr/bin/env python3
"""
NASA Space Apps Challenge 2025 - Exoplanet Detection Web Application
Flask backend with fast-trained ML models for exoplanet detection
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import joblib
import os
import json
from datetime import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'nasa_space_apps_2025_secret_key'
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'txt'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

class ExoplanetDetectorFast:
    """Fast exoplanet detection system with optimized models"""
    
    def __init__(self):
        self.models = {}
        self.preprocessing_pipeline = None
        self.selected_features = []
        self.model_info = {}
        self.load_fast_models()
    
    def load_fast_models(self):
        """Load the fast-trained models and preprocessing tools"""
        try:
            # Load preprocessing pipeline
            if os.path.exists('models/preprocessing_pipeline_fast.pkl'):
                self.preprocessing_pipeline = joblib.load('models/preprocessing_pipeline_fast.pkl')
                print("✅ Fast preprocessing pipeline loaded")
            else:
                print("❌ Fast preprocessing pipeline not found")
            
            # Load selected features
            if os.path.exists('models/selected_features_fast.json'):
                with open('models/selected_features_fast.json', 'r') as f:
                    self.selected_features = json.load(f)
                print(f"✅ Selected features loaded: {len(self.selected_features)} features")
            
            # Load fast-trained models
            fast_model_files = {
                'SVM Fast': 'models/fast_svm_fast.pkl',
                'LightGBM Fast': 'models/fast_lightgbm_fast.pkl',
                'Random Forest Fast': 'models/fast_random_forest_fast.pkl',
                'Gradient Boosting Fast': 'models/fast_gradient_boosting_fast.pkl',
                'Ensemble Fast': 'models/fast_ensemble_fast.pkl'
            }
            
            # Model performance info
            self.model_info = {
                'SVM Fast': {
                    'name': 'SVM Fast',
                    'description': 'Support Vector Machine - Fast Training (Best Model)',
                    'f1_score': 0.329,
                    'accuracy': 0.530,
                    'precision': 0.288,
                    'recall': 0.383,
                    'is_best': True,
                    'training_time': '~4 seconds',
                    'loaded': True
                },
                'LightGBM Fast': {
                    'name': 'LightGBM Fast',
                    'description': 'Light Gradient Boosting - Fast Training',
                    'f1_score': 0.238,
                    'accuracy': 0.615,
                    'precision': 0.293,
                    'recall': 0.200,
                    'is_best': False,
                    'training_time': '~6 seconds',
                    'loaded': True
                },
                'Random Forest Fast': {
                    'name': 'Random Forest Fast',
                    'description': 'Random Forest - Fast Training',
                    'f1_score': 0.057,
                    'accuracy': 0.670,
                    'precision': 0.200,
                    'recall': 0.033,
                    'is_best': False,
                    'training_time': '~4 seconds',
                    'loaded': True
                },
                'Gradient Boosting Fast': {
                    'name': 'Gradient Boosting Fast',
                    'description': 'Gradient Boosting - Fast Training',
                    'f1_score': 0.145,
                    'accuracy': 0.645,
                    'precision': 0.261,
                    'recall': 0.100,
                    'is_best': False,
                    'training_time': '~5 seconds',
                    'loaded': True
                },
                'Ensemble Fast': {
                    'name': 'Ensemble Fast',
                    'description': 'Ensemble Voting - Fast Training',
                    'f1_score': 0.099,
                    'accuracy': 0.635,
                    'precision': 0.191,
                    'recall': 0.067,
                    'is_best': False,
                    'training_time': '~3 seconds',
                    'loaded': True
                },
                'Best Model Fast': {
                    'name': 'Best Model Fast',
                    'description': 'Best Performing Model (SVM Fast) - Optimized for Exoplanet Detection',
                    'f1_score': 0.329,
                    'accuracy': 0.530,
                    'precision': 0.288,
                    'recall': 0.383,
                    'is_best': True,
                    'training_time': '~4 seconds',
                    'loaded': True
                }
            }
            
            for name, path in fast_model_files.items():
                if os.path.exists(path):
                    try:
                        self.models[name] = joblib.load(path)
                        print(f"✅ {name} model loaded successfully")
                    except Exception as e:
                        print(f"❌ Error loading {name}: {e}")
                else:
                    print(f"⚠️ {name} model file not found: {path}")
            
            # Load the best model specifically
            if os.path.exists('models/best_model_fast.pkl'):
                try:
                    self.models['Best Model Fast'] = joblib.load('models/best_model_fast.pkl')
                    print("✅ Best fast model loaded successfully")
                except Exception as e:
                    print(f"❌ Error loading best fast model: {e}")
            
            print(f"📊 Total fast models loaded: {len(self.models)}")
            
            if len(self.models) == 0:
                print("⚠️ No fast models loaded! Please train models first.")
                
        except Exception as e:
            print(f"❌ Error loading fast models: {e}")
    
    def get_average_accuracy(self):
        """Calculate the average accuracy across all loaded models"""
        try:
            if not self.model_info:
                return 0.0
            
            accuracies = [info['accuracy'] for info in self.model_info.values() 
                         if 'accuracy' in info and info.get('loaded', True)]
            
            if not accuracies:
                return 0.0
                
            return sum(accuracies) / len(accuracies)
        except Exception as e:
            print(f"❌ Error calculating average accuracy: {e}")
            return 0.0
    
    def get_model_statistics(self):
        """Get comprehensive model statistics"""
        try:
            stats = {
                'total_models': len([m for m in self.model_info.values() if m.get('loaded', True)]),
                'average_accuracy': self.get_average_accuracy(),
                'best_model': None,
                'best_accuracy': 0.0
            }
            
            # Find best model by F1 score (most important for imbalanced datasets)
            for name, info in self.model_info.items():
                if info.get('loaded', True) and info.get('is_best', False):
                    stats['best_model'] = name
                    stats['best_accuracy'] = info.get('accuracy', 0.0)
                    break
            
            return stats
        except Exception as e:
            print(f"❌ Error calculating model statistics: {e}")
            return {'total_models': 0, 'average_accuracy': 0.0, 'best_model': None, 'best_accuracy': 0.0}
    
    def extract_features(self, flux, time):
        """Extract features from light curve data"""
        try:
            # Basic statistical features
            mean_flux = np.mean(flux)
            std_flux = np.std(flux)
            var_flux = np.var(flux)
            median_flux = np.median(flux)
            min_flux = np.min(flux)
            max_flux = np.max(flux)
            flux_range = max_flux - min_flux
            
            # Percentile features
            q25_flux = np.percentile(flux, 25)
            q75_flux = np.percentile(flux, 75)
            iqr_flux = q75_flux - q25_flux
            
            # Statistical moments
            skewness = stats.skew(flux)
            kurtosis = stats.kurtosis(flux)
            
            # Transit detection features
            median_abs_dev = np.median(np.abs(flux - median_flux))
            threshold = median_flux - 3 * median_abs_dev
            dip_mask = flux < threshold
            
            # Count dip events
            dip_events = 0
            in_dip = False
            dip_durations = []
            current_dip_duration = 0
            
            for is_dip in dip_mask:
                if is_dip and not in_dip:
                    dip_events += 1
                    in_dip = True
                    current_dip_duration = 1
                elif is_dip and in_dip:
                    current_dip_duration += 1
                elif not is_dip and in_dip:
                    dip_durations.append(current_dip_duration)
                    in_dip = False
                    current_dip_duration = 0
            
            if in_dip:
                dip_durations.append(current_dip_duration)
            
            num_dips = dip_events
            avg_dip_duration = np.mean(dip_durations) if dip_durations else 0
            max_dip_duration = np.max(dip_durations) if dip_durations else 0
            min_dip_depth = min_flux - median_flux if min_flux < median_flux else 0
            
            # Period estimation
            if num_dips > 1:
                dip_times = time[dip_mask]
                if len(dip_times) > 1:
                    dip_intervals = np.diff(dip_times)
                    period_estimate = np.median(dip_intervals) if len(dip_intervals) > 0 else 0
                else:
                    period_estimate = 0
            else:
                period_estimate = 0
            
            # Fourier analysis
            if len(flux) >= 32:
                fft = np.fft.fft(flux)
                freqs = np.fft.fftfreq(len(flux), d=np.median(np.diff(time)))
                power = np.abs(fft)**2
                
                if len(power) > 1:
                    dominant_freq_idx = np.argmax(power[1:len(power)//2]) + 1
                    dominant_freq = np.abs(freqs[dominant_freq_idx])
                    dominant_power = power[dominant_freq_idx]
                    total_power = np.sum(power)
                    spectral_centroid = np.sum(freqs[:len(freqs)//2] * power[:len(power)//2]) / np.sum(power[:len(power)//2])
                else:
                    dominant_freq = dominant_power = total_power = spectral_centroid = 0
            else:
                dominant_freq = dominant_power = total_power = spectral_centroid = 0
            
            # Variability measures
            rms_variability = np.sqrt(np.mean((flux - mean_flux)**2))
            mad = np.median(np.abs(flux - median_flux))
            
            # Transit shape analysis
            if num_dips > 0 and min_dip_depth < -0.001:
                deepest_dip_idx = np.argmin(flux)
                window_size = min(20, len(flux) // 10)
                start_idx = max(0, deepest_dip_idx - window_size)
                end_idx = min(len(flux), deepest_dip_idx + window_size)
                
                transit_window_flux = flux[start_idx:end_idx]
                
                if len(transit_window_flux) > 5:
                    pre_dip = transit_window_flux[:len(transit_window_flux)//2]
                    post_dip = transit_window_flux[len(transit_window_flux)//2:]
                    asymmetry = np.mean(pre_dip) - np.mean(post_dip) if len(pre_dip) > 0 and len(post_dip) > 0 else 0
                    bottom_quarter = transit_window_flux[len(transit_window_flux)//4:3*len(transit_window_flux)//4]
                    bottom_flatness = np.std(bottom_quarter) if len(bottom_quarter) > 0 else 0
                else:
                    asymmetry = bottom_flatness = 0
            else:
                asymmetry = bottom_flatness = 0
            
            # Stability check
            mid_point = len(flux) // 2
            first_half = flux[:mid_point]
            second_half = flux[mid_point:]
            
            if len(first_half) > 1 and len(second_half) > 1:
                try:
                    t_stat, p_value = stats.ttest_ind(first_half, second_half)
                    variability_p_value = p_value
                except:
                    variability_p_value = 0.5
            else:
                variability_p_value = 0.5
            
            # Compile feature vector
            features = {
                'mean_flux': mean_flux,
                'std_flux': std_flux,
                'var_flux': var_flux,
                'median_flux': median_flux,
                'min_flux': min_flux,
                'max_flux': max_flux,
                'flux_range': flux_range,
                'q25_flux': q25_flux,
                'q75_flux': q75_flux,
                'iqr_flux': iqr_flux,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'num_dips': num_dips,
                'avg_dip_duration': avg_dip_duration,
                'max_dip_duration': max_dip_duration,
                'min_dip_depth': min_dip_depth,
                'period_estimate': period_estimate,
                'asymmetry': asymmetry,
                'bottom_flatness': bottom_flatness,
                'dominant_freq': dominant_freq,
                'dominant_power': dominant_power,
                'total_power': total_power,
                'spectral_centroid': spectral_centroid,
                'rms_variability': rms_variability,
                'mad': mad,
                'variability_p_value': variability_p_value,
                'data_points': len(flux),
                'time_span': time[-1] - time[0] if len(time) > 1 else 0
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict_single(self, flux, time, model_name):
        """Make prediction using a single specified model"""
        if not self.models or model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
            
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline not loaded")
        
        try:
            # Extract features
            features = self.extract_features(flux, time)
            if features is None:
                raise ValueError("Feature extraction failed")
            
            # Convert to DataFrame and preprocess
            features_df = pd.DataFrame([features])
            features_processed = self.preprocessing_pipeline.transform(features_df)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(features_processed)[0]
            prediction_proba = model.predict_proba(features_processed)[0]
            
            return (
                int(prediction),
                float(max(prediction_proba)),
                features
            )
            
        except Exception as e:
            raise Exception(f"Single prediction failed for {model_name}: {str(e)}")
    
    def predict(self, flux, time, selected_model=None):
        """Make predictions using fast models"""
        if not self.models or self.preprocessing_pipeline is None:
            return {
                'success': False,
                'error': 'Fast models not loaded properly',
                'results': {},
                'best_prediction': None
            }
        
        try:
            # Extract features
            features = self.extract_features(flux, time)
            if features is None:
                return {'error': 'Feature extraction failed'}
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features])
            
            # Apply preprocessing pipeline
            features_processed = self.preprocessing_pipeline.transform(features_df)
            
            predictions = {}
            
            # Select models to use
            if selected_model and selected_model in self.models:
                models_to_use = {selected_model: self.models[selected_model]}
            else:
                models_to_use = self.models
            
            # Make predictions
            for model_name, model in models_to_use.items():
                try:
                    prediction = model.predict(features_processed)[0]
                    prediction_proba = model.predict_proba(features_processed)[0]
                    
                    # Convert to JSON-serializable types
                    prediction = int(prediction)
                    prediction_proba = [float(p) for p in prediction_proba]
                    
                    predictions[model_name] = {
                        'prediction': prediction,
                        'probability': prediction_proba,
                        'confidence': float(max(prediction_proba)),
                        'exoplanet_probability': float(prediction_proba[1]) if len(prediction_proba) > 1 else 0.0,
                        'result': 'Exoplanet Detected' if prediction == 1 else 'No Exoplanet',
                        'model_info': self.model_info.get(model_name, {})
                    }
                except Exception as model_error:
                    predictions[model_name] = {
                        'error': f'Model prediction failed: {str(model_error)}'
                    }
            
            # Determine best prediction (SVM Fast is the best model)
            best_model_name = selected_model if selected_model and selected_model in predictions else 'SVM Fast'
            if best_model_name in predictions and 'error' not in predictions[best_model_name]:
                best_prediction = {
                    'model': best_model_name,
                    **predictions[best_model_name]
                }
            else:
                # Fallback to first available model
                available_models = [name for name, pred in predictions.items() if 'error' not in pred]
                if available_models:
                    best_model_name = available_models[0]
                    best_prediction = {
                        'model': best_model_name,
                        **predictions[best_model_name]
                    }
                else:
                    best_prediction = None
            
            return {
                'success': True,
                'results': predictions,
                'best_prediction': best_prediction,
                'selected_model': selected_model,
                'available_models': list(self.models.keys()),
                'features_extracted': len(features),
                'selected_features': len(self.selected_features)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'results': {},
                'best_prediction': None
            }

# Initialize the detector
detector = ExoplanetDetectorFast()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_light_curve_data(data):
    """Process uploaded light curve data"""
    try:
        # Expected columns: time, flux, or similar
        columns = data.columns.tolist()
        
        # Try to identify time and flux columns
        time_col = None
        flux_col = None
        
        for col in columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['time', 'bjd', 'day', 'date']):
                time_col = col
            elif any(x in col_lower for x in ['flux', 'mag', 'brightness', 'intensity']):
                flux_col = col
        
        # If not found, use first two columns
        if time_col is None or flux_col is None:
            if len(columns) >= 2:
                time_col = columns[0]
                flux_col = columns[1]
            else:
                return None, None, "Data must have at least 2 columns (time and flux)"
        
        time = data[time_col].values
        flux = data[flux_col].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(time) & ~np.isnan(flux)
        time = time[valid_mask]
        flux = flux[valid_mask]
        
        if len(time) < 10:
            return None, None, "Not enough valid data points"
        
        # Normalize flux
        median_flux = np.median(flux)
        flux_normalized = flux / median_flux
        
        return time, flux_normalized, None
        
    except Exception as e:
        return None, None, f"Error processing data: {str(e)}"

# Routes
@app.route('/')
def index():
    """Main page"""
    stats = detector.get_model_statistics()
    return render_template('index.html', 
                         models=detector.model_info,
                         total_models=stats['total_models'],
                         average_accuracy=round(stats['average_accuracy'] * 100, 1),
                         best_model=stats['best_model'])

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models information"""
    stats = detector.get_model_statistics()
    return jsonify({
        'models': detector.model_info,
        'default': 'SVM Fast',
        'available': list(detector.models.keys()),
        'total_models': stats['total_models'],
        'average_accuracy': round(stats['average_accuracy'], 3),
        'best_model': stats['best_model'],
        'best_accuracy': round(stats['best_accuracy'], 3)
    })

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get comprehensive model information for frontend"""
    try:
        stats = detector.get_model_statistics()
        return jsonify({
            'success': True,
            'models': detector.model_info,
            'total_models': stats['total_models'],
            'average_accuracy': round(stats['average_accuracy'], 3),
            'best_model': stats['best_model'],
            'best_accuracy': round(stats['best_accuracy'], 3),
            'loaded_models': list(detector.models.keys())
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'models': {},
            'total_models': 0,
            'average_accuracy': 0.0
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make exoplanet predictions"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload CSV or TXT files.'}), 400
        
        # Get selected model
        selected_model = request.form.get('model', None)
        
        # Save uploaded file temporarily
        upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_path)
        
        try:
            # Read the data
            if file.filename.endswith('.csv'):
                data = pd.read_csv(upload_path)
            else:
                # Try different delimiters for text files
                try:
                    data = pd.read_csv(upload_path, delimiter='\t')
                except:
                    data = pd.read_csv(upload_path, delimiter=' ', skipinitialspace=True)
            
            # Process the light curve data
            time, flux, error = process_light_curve_data(data)
            
            if error:
                os.remove(upload_path)
                return jsonify({'error': error}), 400
            
            # Make predictions
            results = detector.predict(flux, time, selected_model)
            
            # Add data summary
            results['data_summary'] = {
                'data_points': len(flux),
                'time_span': float(time[-1] - time[0]) if len(time) > 1 else 0,
                'flux_mean': float(np.mean(flux)),
                'flux_std': float(np.std(flux))
            }
            
            # Clean up
            os.remove(upload_path)
            
            return jsonify(results)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(upload_path):
                os.remove(upload_path)
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/sample_data', methods=['GET'])
def get_sample_data():
    """Get sample data for testing"""
    try:
        # Load the dataset
        with open('data/photometric_dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)
        
        # Get a random sample
        import random
        idx = random.randint(0, len(dataset['light_curves']) - 1)
        
        sample_lc = dataset['light_curves'][idx]
        sample_label = dataset['labels'][idx]
        sample_star_id = dataset['star_ids'][idx]
        
        # Convert numpy arrays to lists for JSON serialization
        return jsonify({
            'success': True,
            'star_id': sample_star_id,
            'time': sample_lc['time'].tolist(),
            'flux': sample_lc['flux'].tolist(),
            'flux_err': sample_lc['flux_err'].tolist() if 'flux_err' in sample_lc else None,
            'true_label': int(sample_label),
            'data_points': len(sample_lc['time']),
            'time_span': float(sample_lc['time'][-1] - sample_lc['time'][0]),
            'description': f'Real photometric data from {sample_star_id}',
            'parameters': {
                'star_id': sample_star_id,
                'true_label': 'Exoplanet' if sample_label == 1 else 'No Exoplanet',
                'data_points': len(sample_lc['time']),
                'time_span': f"{float(sample_lc['time'][-1] - sample_lc['time'][0]):.2f} days"
            }
        })
        
    except FileNotFoundError:
        # Fallback to synthetic data if dataset not available
        return get_synthetic_sample_data()
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error loading sample data: {str(e)}'
        }), 500

def get_synthetic_sample_data():
    """Generate synthetic sample light curve data for demonstration"""
    try:
        # Generate a synthetic light curve with potential transit
        np.random.seed(42)  # For reproducible results
        
        # Time points (in days)
        time = np.linspace(0, 27.4, 1000)  # Kepler-like observation period
        
        # Base stellar flux with some variability
        base_flux = 1.0 + 0.001 * np.sin(2 * np.pi * time / 13.2)  # Stellar rotation
        noise = np.random.normal(0, 0.0005, len(time))  # Photometric noise
        
        # Add a transit signal
        transit_period = 3.52  # Transit every 3.52 days
        transit_duration = 0.13  # Transit duration in days
        transit_depth = 0.01  # 1% depth
        
        flux = base_flux.copy()
        for i in range(int(27.4 / transit_period) + 1):
            transit_start = i * transit_period
            transit_end = transit_start + transit_duration
            
            # Simple box transit model
            transit_mask = (time >= transit_start) & (time <= transit_end)
            flux[transit_mask] *= (1 - transit_depth)
        
        # Add noise
        flux += noise
        
        # Normalize
        flux = flux / np.median(flux)
        
        return jsonify({
            'success': True,
            'star_id': 'synthetic_star_001',
            'time': time.tolist(),
            'flux': flux.tolist(),
            'true_label': 1,  # This synthetic data has an exoplanet
            'data_points': len(time),
            'time_span': float(time[-1] - time[0]),
            'description': 'Synthetic Kepler-like light curve with exoplanet transit',
            'parameters': {
                'transit_period': f'{transit_period} days',
                'transit_depth': f'{transit_depth * 100}%',
                'duration': f'{transit_duration * 24:.1f} hours',
                'data_points': len(time)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error generating synthetic sample data: {str(e)}'
        }), 500

@app.route('/predict_sample', methods=['POST'])
def predict_sample():
    """Make predictions on sample data"""
    try:
        data = request.get_json()
        
        if not data or 'time' not in data or 'flux' not in data:
            return jsonify({'error': 'Invalid sample data format'}), 400
            
        time = np.array(data['time'])
        flux = np.array(data['flux'])
        
        # Make predictions using all models
        all_predictions = {}
        
        for model_name, model in detector.models.items():
            try:
                prediction, confidence, features = detector.predict_single(
                    flux, time, model_name
                )
                
                all_predictions[model_name] = {
                    'prediction': int(prediction),
                    'confidence': float(confidence),
                    'prediction_text': 'Exoplanet Detected' if prediction == 1 else 'No Exoplanet',
                    'model_info': detector.model_info.get(model_name, {})
                }
                
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                all_predictions[model_name] = {
                    'prediction': 0,
                    'confidence': 0.0,
                    'prediction_text': 'Model Error',
                    'error': str(e),
                    'model_info': detector.model_info.get(model_name, {})
                }
        
        # Get ensemble prediction (majority vote)
        predictions = [pred['prediction'] for pred in all_predictions.values() 
                      if 'error' not in pred]
        ensemble_prediction = 1 if sum(predictions) > len(predictions) / 2 else 0
        
        # Calculate average confidence
        confidences = [pred['confidence'] for pred in all_predictions.values() 
                      if 'error' not in pred]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return jsonify({
            'success': True,
            'results': all_predictions,
            'ensemble': {
                'prediction': ensemble_prediction,
                'confidence': float(avg_confidence),
                'prediction_text': 'Exoplanet Detected' if ensemble_prediction == 1 else 'No Exoplanet'
            },
            'analysis': {
                'total_models': len(all_predictions),
                'successful_models': len([p for p in all_predictions.values() if 'error' not in p]),
                'positive_predictions': sum(predictions),
                'data_points': len(time)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Sample prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(detector.models),
        'preprocessing_ready': detector.preprocessing_pipeline is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting VIRTUAL STATOSCOPE - NASA Space Apps Challenge 2025")
    print("🌟 Fast Exoplanet Detection Web Application")
    print("🤖 AI Models: SVM, LightGBM, Random Forest, Gradient Boosting, Ensemble")
    print("⚡ Training Time: ~22 seconds total")
    print("🎯 Best Model: SVM Fast (F1-Score: 0.329)")
    print("🌐 Server running on http://localhost:5000")
    
    # Create required directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
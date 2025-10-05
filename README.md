# 🌟 EXOVISION - EXOPLANET DETECTION SYSTEM

## NASA Space Apps Challenge 2025 Project

**Team:** Code Surgeons Innovator
**Challenge:** Exoplanet Detection using AI/ML  
**Dataset:** NASA Exoplanet Archive PHOTOMETRIC Data  

### 🚀 Project Overview

EXOVISION is an advanced machine learning system that automatically detects exoplanets by analyzing stellar light curves from NASA's photometric dataset. Our system uses the transit method to identify the characteristic dimming patterns that occur when an exoplanet passes in front of its host star.

### 🎯 Key Features

- **Multi-Model AI Pipeline**: 5 optimized machine learning models (SVM, Random Forest, LightGBM, Gradient Boosting, Ensemble)
- **Fast Training**: Complete model training in under 22 seconds (15x speed improvement)
- **Web Interface**: Interactive Flask web application for real-time predictions
- **Automated Data Processing**: Intelligent feature extraction from light curve data
- **NASA Dataset Integration**: Direct processing of NASA Exoplanet Archive data
- **Visualization Tools**: Comprehensive plots for data exploration and model evaluation

### 🏆 Performance Highlights

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| **SVM (Best)** ⭐ | 53.0% | **0.329** | ~4s |
| LightGBM | 61.5% | 0.238 | ~6s |
| Gradient Boosting | 64.5% | 0.145 | ~5s |
| Random Forest | 67.0% | 0.057 | ~4s |
| Ensemble | 63.5% | 0.099 | ~3s |

*SVM selected as best model due to optimal F1-score for imbalanced exoplanet detection*

### 🛠️ Technology Stack

- **Backend**: Flask, Python 3.12
- **Machine Learning**: scikit-learn, LightGBM, TensorFlow
- **Data Processing**: pandas, NumPy, SciPy
- **Visualization**: matplotlib, seaborn
- **Frontend**: HTML/CSS/JavaScript
- **Deployment**: Local Flask server

### 📁 Project Structure

```
EXOVISION/
├── app.py                          # Flask web application
├── exoplanet_detection.ipynb       # Main ML pipeline notebook
├── model_comparison.md             # Model performance analysis
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── data/
│   └── photometric_dataset.pkl     # Processed NASA dataset
├── models/                         # Trained ML models
│   ├── best_model_fast.pkl
│   ├── fast_svm_fast.pkl
│   ├── fast_lightgbm_fast.pkl
│   └── ...
├── static/                         # Web assets
│   ├── css/style.css
│   ├── js/app.js
│   └── plots/
├── templates/
│   └── index.html                  # Main web interface
└── uploads/                        # User data uploads
```

### 🚀 Quick Start

#### 1. Clone & Setup
```bash
git clone <repository-url>
cd EXOVISION
```

#### 2. Create Virtual Environment
```bash
python -m venv env
env\Scripts\activate  # Windows
# source env/bin/activate  # Linux/Mac
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the Application
```bash
python app.py
```

#### 5. Open Browser
Navigate to `http://localhost:5000` to access the web interface.

### 📊 Usage

#### Web Application
1. **Upload Data**: Upload CSV files with light curve data
2. **Select Model**: Choose from 5 trained ML models
3. **Get Predictions**: Receive instant exoplanet detection results
4. **View Analysis**: Explore detailed visualizations and confidence scores

#### Jupyter Notebook
1. Open `exoplanet_detection.ipynb`
2. Run all cells to reproduce the complete ML pipeline
3. Experiment with different models and parameters

### 🔬 Scientific Approach

#### Transit Method Detection
Our system identifies exoplanets using the transit method:
- Analyzes periodic dimming in stellar brightness
- Extracts 15 optimized features from light curves
- Applies robust preprocessing and scaling
- Uses ensemble voting for final predictions

#### Feature Engineering
Key extracted features include:
- Statistical moments (mean, variance, skewness, kurtosis)
- Periodicity detection via autocorrelation
- Transit depth and duration measurements
- Frequency domain analysis
- Outlier and anomaly detection

### 🎯 NASA Space Apps Challenge Goals

✅ **Automated Detection**: AI-powered exoplanet identification  
✅ **Real-time Processing**: Fast predictions on new data  
✅ **User-Friendly Interface**: Accessible web application  
✅ **Scientific Accuracy**: Optimized for astronomical data  
✅ **Scalable Architecture**: Handles large datasets efficiently  

### 📈 Model Optimization

Our fast training approach includes:
- **Hyperparameter Optimization**: Reduced search space for speed
- **Feature Selection**: Top 15 most informative features
- **Cross-validation**: 2-fold validation for efficiency
- **Class Balancing**: Weighted models for imbalanced data
- **Pipeline Integration**: Automated preprocessing workflows

### 🔧 API Endpoints

- `GET /`: Main web interface
- `POST /predict`: Upload and analyze light curve data
- `GET /models`: List available ML models
- `POST /batch_predict`: Batch processing for multiple files

### 🌟 Future Enhancements

- Real-time NASA data integration
- Deep learning models (CNN/LSTM)
- Multi-planet system detection
- Advanced visualization dashboards
- Cloud deployment and scaling

### 👥 Team

**Code Surgeons Innovator** - NASA Space Apps Challenge 2025 participants dedicated to advancing exoplanet discovery through innovative AI/ML solutions.

### 📄 License

This project is developed for the NASA Space Apps Challenge 2025. Please refer to NASA's data usage policies for the underlying datasets.

### 🙏 Acknowledgments

- NASA Exoplanet Archive for providing the photometric dataset
- NASA Space Apps Challenge organizers
- Open-source ML community (scikit-learn, TensorFlow, Flask)

---

### 👨‍💻 Developer

**Noaman Ayub**  
🌐 **Portfolio**: [noamanayub.netlify.app](https://noamanayub.netlify.app/)  
💼 **LinkedIn**: [linkedin.com/in/noamanayub](https://www.linkedin.com/in/noamanayub/)  
💻 **GitHub**: [github.com/noamanayub](https://github.com/noamanayub)  

---


**🌍 Discovering New Worlds Through Machine Learning** 🚀


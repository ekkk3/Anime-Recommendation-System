# 🎌 Anime Recommendation System

**Final Project - Recommendation Systems Course**

A comprehensive anime recommendation system using Content-Based Filtering and Collaborative Filtering with the Anime Recommendation Database 2020 from Kaggle.

---

## 📋 Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Details](#model-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Web Interface](#web-interface)
- [Project Requirements](#project-requirements)
- [License](#license)

---

## ✨ Features

- ✅ **2,000+ Anime Items**: ~17,000 anime with 5+ features each
- ✅ **Multiple Recommendation Approaches**:
  - Content-Based Filtering (TF-IDF + Cosine Similarity)
  - Collaborative Filtering (SVD/NMF)
  - Hybrid Model
- ✅ **Data Processing Pipeline**:
  - Missing value handling
  - Duplicate removal
  - Data normalization
  - Outlier detection
- ✅ **Rich Visualizations**:
  - Score distribution histograms
  - Genre frequency charts
  - Top anime rankings
  - Correlation heatmaps
- ✅ **Interactive Web Interface**: Built with Streamlit
- ✅ **Model Evaluation**: RMSE, MAE, Precision@K, Recall@K

---

## 📊 Dataset

**Source**: [Kaggle - Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)

### Files:
- `anime.csv`: ~17,000 anime with metadata
- `rating_complete.csv`: ~109M user ratings
- `animelist.csv`: User watch history

### Features:
- MAL_ID, Name, Score, Genres, Type, Episodes
- Studios, Producers, Source, Duration
- Rating, Rank, Popularity, Members, Favorites
- Image URL, Synopsis

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip
- Kaggle API credentials

### Step 1: Clone Repository

```bash
git clone <your-repository-url>
cd Anime-Recommender-System
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

```bash
# Setup Kaggle API (first time only)
# Place your kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d hernan4444/anime-recommendation-database-2020

# Extract to data/raw/
unzip anime-recommendation-database-2020.zip -d data/raw/
```

---

## 📁 Project Structure

```
Anime-Recommender-System/
│
├── data/
│   ├── raw/                          # Raw data from Kaggle
│   │   ├── anime.csv
│   │   ├── rating_complete.csv
│   │   └── animelist.csv
│   ├── processed/                    # Cleaned data
│   │   ├── anime_cleaned.csv
│   │   ├── ratings_sampled.csv
│   │   └── anime_features.pkl
│   └── visualizations/               # EDA plots
│
├── src/                              # Source code
│   ├── data_cleaner.py              # Data cleaning pipeline
│   ├── models/
│   │   ├── content_based.py         # Content-based model
│   │   ├── collaborative.py         # Collaborative filtering
│   │   └── hybrid_recommender.py    # Hybrid approach
│   └── utils.py
│
├── app/                              # Streamlit web app
│   └── main.py
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_content_based_model.ipynb
│   └── 04_collaborative_filtering.ipynb
│
├── models/                           # Trained models
│   ├── content_model.pkl
│   ├── collaborative_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── docs/                             # Documentation
│   └── report.pdf
│
├── train_model.py                    # Training script
├── config.py                         # Configuration
├── requirements.txt
└── README.md
```

---

## 🎯 Usage

### 1. Train Models

Train both content-based and collaborative filtering models:

```bash
python train_model.py
```

This script will:
- Clean the anime data
- Sample rating data (5M ratings from 109M)
- Train content-based model (TF-IDF + Cosine Similarity)
- Train collaborative filtering model (SVD)
- Evaluate models on test set
- Save trained models to `models/` directory

**Expected Output:**
```
Training Pipeline Completed Successfully!
Content-Based Model saved to: models/content_model.pkl
Collaborative Model saved to: models/collaborative_model.pkl
Test RMSE: 0.85
Test MAE: 0.67
```

### 2. Run Web Application

```bash
streamlit run app/main.py
```

Or:

```bash
python -m streamlit run app/main.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Explore Notebooks

```bash
jupyter notebook
```

Navigate to `notebooks/` to explore:
- Data exploration and visualization
- Model training experiments
- Evaluation metrics

---

## 🧠 Model Details

### 1. Content-Based Filtering

**Algorithm**: TF-IDF Vectorization + Cosine Similarity

**Features Used**:
- Genres (weighted 2x)
- Type (TV, Movie, OVA, etc.)
- Source (Manga, Light Novel, Original)
- Studios
- Rating Category (G, PG, PG-13, R)

**Process**:
1. Combine text features into single string
2. Apply TF-IDF vectorization (5000 features, 1-2 grams)
3. Compute cosine similarity matrix
4. Recommend top-N most similar anime

**Pros**: 
- No cold-start problem
- Explainable recommendations
- Works for new anime

**Cons**:
- Limited diversity
- Cannot discover new preferences

### 2. Collaborative Filtering

**Algorithm**: SVD (Singular Value Decomposition)

**Parameters**:
- Latent factors: 100
- Epochs: 20
- Learning rate: 0.005
- Regularization: 0.02

**Process**:
1. Create user-item rating matrix
2. Apply matrix factorization (SVD)
3. Predict ratings for unseen anime
4. Recommend top-N highest predicted ratings

**Pros**:
- Discovers hidden patterns
- Personalized recommendations
- Can suggest diverse content

**Cons**:
- Cold-start problem for new users/anime
- Requires sufficient rating data
- Computationally expensive

### 3. Hybrid Model (Optional)

**Approach**: Weighted combination

```
final_score = 0.4 * content_score + 0.6 * collaborative_score
```

Combines strengths of both approaches for better recommendations.

---

## 📈 Evaluation Metrics

### Implemented Metrics:

1. **RMSE (Root Mean Squared Error)**
   - Measures prediction accuracy
   - Target: < 1.0

2. **MAE (Mean Absolute Error)**
   - Average absolute prediction error
   - Target: < 0.8

3. **Precision@K**
   - Proportion of relevant items in top-K
   - Target: > 0.15 for K=10

4. **Recall@K**
   - Proportion of relevant items retrieved
   - Target: > 0.10 for K=10

### Expected Results:

| Model | RMSE | MAE | Precision@10 | Recall@10 |
|-------|------|-----|--------------|-----------|
| Content-Based | N/A | N/A | ~0.18 | ~0.12 |
| SVD | ~0.85 | ~0.67 | ~0.20 | ~0.15 |
| Hybrid | ~0.82 | ~0.65 | ~0.22 | ~0.17 |

---

## 🖥️ Web Interface

The Streamlit app includes:

### 🏠 Home Page
- Top rated anime
- Filters by score, type, season
- Anime cards with images

### 🔍 Search
- Search anime by name
- Instant results
- Detailed anime information

### ⭐ Recommendations
- Content-based recommendations
- Select anime you like
- Get similar suggestions
- Adjustable parameters (top-N, min score)

### 📊 Analytics
- Score distribution histogram
- Genre frequency bar chart
- Type distribution pie chart
- Timeline of anime releases
- Interactive Plotly visualizations

### ℹ️ About
- Project information
- Technical details
- Dataset statistics

---

## ✅ Project Requirements Compliance

### ✅ 1. Data Collection
- [x] Dataset ≥ 2,000 items (**17,000+ anime**)
- [x] ≥ 5 features per item (**15+ features**)

### ✅ 2. Data Cleaning (3+ tasks)
- [x] Missing values handling
- [x] Duplicate removal
- [x] Data normalization
- [x] Outlier removal
- [x] TF-IDF vectorization

### ✅ 3. Visualization (3+ tasks)
- [x] Rating distribution histogram
- [x] Genre frequency chart
- [x] Top anime rankings
- [x] Correlation heatmap
- [x] Type distribution pie chart

### ✅ 4. Recommendation Models
- [x] Content-Based Filtering (TF-IDF + Cosine)
- [x] Collaborative Filtering (SVD)
- [x] Hybrid approach (optional)

### ✅ 5. Model Evaluation
- [x] RMSE
- [x] MAE
- [x] Precision@K
- [x] Recall@K

### ✅ 6. User Interface
- [x] Streamlit web interface
- [x] Interactive components
- [x] Search functionality
- [x] Visualization dashboards

---

## 🎓 Advanced Features (Bonus Points)

### Implemented:
- ✅ TF-IDF vectorization (embeddings)
- ⬜ Real-time recommendations (can add)
- ⬜ User history tracking (can add with SQLite)
- ⬜ Context-aware recommendations (can add)
- ⬜ Cloud deployment (can deploy to Streamlit Cloud)

### To Add Real-Time Features:

```python
# Add to app/main.py
import sqlite3

def save_user_interaction(user_id, anime_id, interaction_type):
    conn = sqlite3.connect('data/user_history.db')
    # Save interaction
    conn.close()

def get_user_history(user_id):
    conn = sqlite3.connect('data/user_history.db')
    # Fetch history
    conn.close()
    return history
```

---

## 📦 Submission

Create submission package:

```bash
# Create zip file
zip -r TenSV_maSV_finalProject.zip \
  src/ \
  app/ \
  models/ \
  notebooks/ \
  data/processed/ \
  data/visualizations/ \
  docs/ \
  train_model.py \
  config.py \
  requirements.txt \
  README.md
```

**Deliverables**:
1. ✅ Complete source code
2. ✅ Report (8-12 pages) in `docs/report.pdf`
3. ⭐ Demo video (3-5 minutes) - Optional

---

## 🐛 Troubleshooting

### Issue: Kaggle API not working
```bash
# Set up Kaggle API credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: Memory error when loading data
```python
# In train_model.py, reduce sample size
RATING_SAMPLE_SIZE = 1_000_000  # Instead of 5M
```

### Issue: Streamlit app not loading models
```bash
# Ensure models are trained first
python train_model.py

# Then run app
streamlit run app/main.py
```

---

## 📝 License

MIT License - This project is for educational purposes.

---

## 👨‍💻 Author

**[Your Name]**  
Student ID: [Your ID]  
Course: Recommendation Systems  
University: [Your University]

---

## 🙏 Acknowledgments

- Dataset: MyAnimeList via Kaggle
- Libraries: Scikit-learn, Surprise, Streamlit, Plotly
- Inspiration: Netflix, Spotify recommendation systems

---

## 📚 References

1. [Kaggle Dataset](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)
2. [Surprise Documentation](https://surpriselib.com/)
3. [Streamlit Documentation](https://docs.streamlit.io/)
4. Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook.

---

**⭐ If you found this project helpful, please give it a star!**
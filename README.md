# 🎬 IMDb Sentiment Analysis Project

A comprehensive sentiment analysis application movie reviews using NLP and machine learning techniques. This project includes both the training notebook and an interactive Streamlit web application.

## 📋 Project Overview

This project analyzes the sentiment of IMDb movie reviews, classifying them as either **positive** or **negative**. It implements two machine learning models:

- **Logistic Regression** (88.47% accuracy)
- **Naive Bayes** (85.20% accuracy)

## 🎯 Key Features

### 📊 Training & Analysis
- **Data Preprocessing**: Text cleaning, tokenization, lemmatization
- **Feature Extraction**: TF-IDF vectorization with 5000 features
- **Model Training**: Logistic Regression and Naive Bayes classifiers
- **Performance Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices
- **Data Visualization**: Word frequency analysis, performance comparisons

### 🌐 Interactive Web Application
- **Project Overview**: Complete project documentation and insights
- **Training Results**: Detailed model performance analysis
- **Model Inference**: Real-time sentiment prediction interface
- **Data Analysis**: Dataset exploration and visualization

## 📁 Project Structure

```
Sentiment Analysis for IMDb Reviews/
├── Sentiment_Analysis_for_IMDb_Reviews.ipynb  # Training notebook
├── app.py                                     # Streamlit application
├── requirements.txt                           # Python dependencies
├── README.md                                 # Project documentation
├── .gitignore                                # Git ignore file
├── saved_models/                             # Trained models
│   ├── lr_model.pkl                         # Logistic Regression model
│   └── nb_model.pkl                         # Naive Bayes model
└── IMDB Dataset.csv                          # Dataset (not in git)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Sentiment Analysis for IMDb Reviews"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Place `IMDB Dataset.csv` in the project root directory
   - The dataset contains 50,000 movie reviews with sentiment labels

4. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Explore the different pages using the sidebar navigation

## 📖 Usage Guide

### 🏠 Project Overview
- Complete project documentation
- Technical approach explanation
- Dataset information
- Technology stack details

### 📊 Training Results
- Model performance comparison
- Detailed metrics and visualizations
- Confusion matrices
- Key insights and analysis

### 🔍 Model Inference
- **Enter a movie review** in the text area
- **Click "Analyze Sentiment"** to get predictions
- **View results** from both models with confidence scores
- **Try example reviews** for quick testing

### 📈 Data Analysis
- Dataset statistics and distribution
- Word frequency analysis
- Text preprocessing insights
- Model comparison details

## 🔧 Technical Details

### Data Preprocessing Pipeline
1. **Text Cleaning**: Convert to lowercase, remove special characters
2. **Tokenization**: Split text into individual words
3. **Stop Word Removal**: Remove common words (the, and, is, etc.)
4. **Lemmatization**: Convert words to base form
5. **TF-IDF Vectorization**: Convert to numerical features

### Model Architecture
- **Logistic Regression**: Linear classifier with regularization
- **Naive Bayes**: Probabilistic classifier based on Bayes theorem
- **Feature Space**: 5000 TF-IDF features
- **Evaluation**: 80-20 train-test split

### Performance Metrics
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 88.47% | 0.89 | 0.88 | 0.88 |
| Naive Bayes | 85.20% | 0.85 | 0.85 | 0.85 |

## 🛠️ Technologies Used

- **Python**: Core programming language
- **NLTK**: Natural language processing
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application framework
- **Joblib**: Model serialization

## 📊 Dataset Information

- **Source**: IMDb movie reviews dataset
- **Size**: 50,000 reviews
- **Classes**: Positive (1) and Negative (0)
- **Balance**: 50% positive, 50% negative
- **Features**: Text reviews with sentiment labels

## 🎯 Key Insights

### Model Performance
- **Logistic Regression** performs slightly better overall
- Both models handle balanced datasets well
- TF-IDF features effectively capture sentiment patterns
- Text preprocessing significantly improves performance

### Word Analysis
- **Positive words**: good, story, character, great, see
- **Negative words**: even, bad, terrible, disappointing, boring
- Some words appear in both categories with different contexts

## 🔮 Future Enhancements

- [ ] Add more ML models (BERT, LSTM, etc.)
- [ ] Implement model ensemble methods
- [ ] Add sentiment intensity scoring
- [ ] Include review rating prediction
- [ ] Add model interpretability features
- [ ] Implement batch processing for multiple reviews

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Created as part of a Natural Language Processing project for sentiment analysis.

## 🙏 Acknowledgments

- IMDb for providing the dataset
- NLTK and Scikit-learn communities
- Streamlit for the amazing web framework
- Open source community for various libraries and tools

---

**Note**: Make sure to add the `IMDB Dataset.csv` file to your `.gitignore` before pushing to GitHub, as it's not included in this repository due to size constraints. 
# 🎯 Customer Segmentation Web App

> An advanced ML web application for customer segmentation using K-Means clustering. Features StandardScaler, centroid extraction, cluster visualization, and a REST API. Built with Flask and a modern responsive UI.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Customer Segments](#customer-segments)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project is an end-to-end machine learning application that segments customers based on **Annual Income** and **Spending Score** using K-Means clustering. The pipeline uses **StandardScaler** for feature scaling, trains on scaled data, and exposes both a web UI and a REST API. Results include segment label, business description, and **cluster centroid values** (income & score). A **cluster visualization** scatter plot is generated during training and displayed on the page.

**Key Highlights:**
- 🧠 K-Means with **feature scaling** (StandardScaler)
- 📐 **Centroid extraction** saved to JSON and shown in UI/API
- 📊 **Cluster visualization** (scatter plot with centroids) in `static/cluster.png`
- 🌐 **REST API** at `POST /api/predict` for programmatic access
- 🎨 Modern UI with glassmorphism, gradient background, and responsive grid layout
- ✅ Validation: spending score 1–100, non-negative income

## ✨ Features

- **Machine Learning**
  - K-Means with 5 clusters on scaled data
  - StandardScaler for Annual Income and Spending Score
  - Model saved as `model.pkl`, scaler as `scaler.pkl`
  - Centroids inverse-transformed to original scale and saved to `centroids.json`

- **Visualization**
  - Scatter plot of data points colored by cluster
  - Centroids marked on plot
  - Saved as `static/cluster.png`, displayed on the web app

- **Web Interface**
  - Input form (income, spending score)
  - Prediction result card with color-coded segment badge
  - Business meaning and **centroid values** (income & score)
  - **K-Means Customer Segments** section showing `cluster.png`
  - Gradient background, glassmorphism cards, hover animations, responsive grid

- **API**
  - `POST /predict` – used by the UI; returns label, color, description, centroid_income, centroid_score
  - `POST /api/predict` – REST API with JSON input/output

- **Validation**
  - Spending score between 1–100
  - Non-negative income
  - Error messages shown in UI and API responses

## 🛠️ Tech Stack

### Backend
- **Flask** – Web framework
- **scikit-learn** – K-Means, StandardScaler
- **pandas** – Data loading and manipulation
- **numpy** – Numerical operations
- **matplotlib** – Cluster visualization
- **pickle** – Model and scaler serialization

### Frontend
- **HTML5** – Structure
- **CSS3** – Internal styling (glassmorphism, grid, responsive)
- **JavaScript** – Form validation and API calls

### Dataset
- **Mall_Customers.csv** – Annual Income (k$) and Spending Score (1-100)

## 📦 Installation

### Prerequisites

- Python 3.7+
- pip

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/UltronValour/customer-segmentation-app.git
   cd customer-segmentation-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```
   This generates:
   - `model.pkl` – Trained K-Means model
   - `scaler.pkl` – Fitted StandardScaler
   - `centroids.json` – Cluster centroids (original scale)
   - `static/cluster.png` – Cluster visualization

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   Navigate to **http://127.0.0.1:5000**

## 🚀 Usage

### Quick start

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open **http://127.0.0.1:5000**.

### Web UI

1. Enter **Annual Income (k$)** and **Spending Score (1–100)**.
2. Click **Predict segment**.
3. View the segment badge, business description, and **Cluster Centroid Values** (income & score).
4. Scroll to **K-Means Customer Segments** to see the cluster plot.

### Example inputs

| Annual Income (k$) | Spending Score | Expected Segment                |
|--------------------|----------------|---------------------------------|
| 15                 | 39             | Low Income – Low Spending       |
| 80                 | 90             | High Income – High Spending     |
| 20                 | 85             | Low Income – High Spending      |
| 75                 | 15             | High Income – Low Spending      |
| 50                 | 50             | Average Customers               |

## 📁 Project Structure

```
customer-segmentation-app/
│
├── app.py                    # Flask backend (loads model, scaler, centroids)
├── train_model.py            # Training: scale, K-Means, centroids, visualization
├── model.pkl                 # Trained K-Means model (generated)
├── scaler.pkl                # StandardScaler (generated)
├── centroids.json            # Cluster centroids in original scale (generated)
├── Mall_Customers.csv        # Dataset
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
│
├── static/
│   └── cluster.png           # K-Means scatter plot (generated)
│
└── templates/
    └── index.html           # Web UI (form, result, centroid display, visualization)
```

## 📡 API Documentation

### `GET /`

Returns the main HTML page.

---

### `POST /predict`

Used by the web UI. Request body can send `income` and `score` as strings or numbers.

**Request:**
```json
{
  "income": "50",
  "score": "60"
}
```

**Response (success):**
```json
{
  "success": true,
  "label": "Average Customers",
  "color": "Blue",
  "description": "Moderate Value Customers",
  "centroid_income": 55.2,
  "centroid_score": 49.8
}
```

**Response (error):**
```json
{
  "success": false,
  "error": "Spending Score must be between 1 and 100."
}
```

---

### `POST /api/predict`

REST API for programmatic access.

**Request:**
```json
{
  "income": 70,
  "score": 80
}
```

**Response:**
```json
{
  "segment": "High Income – High Spending",
  "centroid_income": 85.23,
  "centroid_score": 78.45,
  "color": "Green",
  "description": "Target Customers 💰"
}
```

**Error (e.g. validation):**  
Returns JSON with `"error"` key and appropriate HTTP status (400 or 500).

**Example (curl):**
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 70, "score": 80}'
```

## 👥 Customer Segments

| Cluster | Label                        | Color | Description              | Business meaning     |
|---------|------------------------------|-------|--------------------------|----------------------|
| 0       | Low Income – Low Spending    | Gray  | Low Priority Customers   | Least value          |
| 1       | High Income – High Spending  | Green | Target Customers 💰      | High value customers |
| 2       | Low Income – High Spending   | Orange| Impulse Buyers           | High engagement      |
| 3       | High Income – Low Spending   | Red   | Potential Customers      | Need marketing       |
| 4       | Average Customers            | Blue  | Moderate Value Customers | Regular customers    |

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit: `git commit -m 'Add some AmazingFeature'`
4. Push: `git push origin feature/AmazingFeature`
5. Open a Pull Request.

## 📝 License

This project is licensed under the MIT License – see [LICENSE](LICENSE) for details.

## 👤 Author

**Valour Moraes**
- GitHub: [@UltronValour](https://github.com/UltronValour)
- Email: valourmoraes@gmail.com

## 🙏 Acknowledgments

- Dataset: [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- scikit-learn and Flask documentation

---

⭐ If you found this project helpful, please consider giving it a star!

# FilmGuru - Movie Recommendation System

FilmGuru is an intelligent movie recommendation system built using Python, Streamlit, and Machine Learning.  
It suggests movies similar to the one you select, using a trained KNN model on movie features.

---

## How to Run This Project

### 1. Clone the Repository
```bash
git clone https://github.com/sarthak00001/FlimGuru.git
cd FlimGuru
```

### 2. Download the Required Files
This project uses some large `.pkl` files that are stored externally.

Run the script below to automatically download them from Google Drive:
```bash
python download_data.py
```

This will create the necessary folders and download the `.pkl` files into your project directory.

---

### 3. Install Dependencies
Make sure you have Python installed (>=3.8), then run:
```bash
pip install -r requirements.txt
```

---

### 4. Run the Application
Launch the web app using Streamlit:
```bash
streamlit run app.py
```

Your browser will open a local app (by default at http://localhost:8501) where you can start exploring FilmGuru.

---

## Tech Stack
- Python  
- Pandas, NumPy, Scikit-Learn  
- Streamlit for UI  
- Google Drive for large file storage  

---

## Features
- Movie-to-movie recommendation  
- User-friendly web interface  
- Lightweight and modular design  

---

## Contributing
Feel free to fork this repo, raise issues, or submit pull requests.

---

## License
This project is licensed under the MIT License – feel free to use it for learning and development.

# FlimGuru
Movie Recommendation System

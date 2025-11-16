\# Comment Categorization \& Reply Assistant



This project classifies social media comments into categories like praise, support, constructive criticism, hate, threat, emotional, spam, and questions.



---



\## Features

\- Cleans comments

\- Trains a machine learning model

\- Predicts categories for new comments

\- Has a simple UI (Streamlit)

\- Includes reply templates



---



\## Project Structure



comment-assistant/

│

├── prepared.csv

│

├── models/

│   ├── model.pkl

│   └── vectorizer.pkl

│

└── src/

&nbsp;   ├── features.py

&nbsp;   ├── train\_model.py

&nbsp;   ├── predict.py

&nbsp;   ├── app\_streamlit.py

&nbsp;   └── templates.py



---



\## Requirements



Run this to install needed packages:



pip install streamlit pandas scikit-learn joblib



---



\## How to Use



Train the model:



python src/train\_model.py



Predict labels:



python src/predict.py



Run the UI:



streamlit run src/app\_streamlit.py



---



\## Dataset

Contains 200 comments labeled into 8 categories.



---



\## Model Details

Uses TF-IDF + Logistic Regression.



---



\## Reply Templates

Stored in:



src/templates.py



---



\## Author

Harini A




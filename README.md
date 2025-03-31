# SmartCityCO2Predictor

📌 Project Overview

SmartCityCO2Predictor is a machine learning-based tool that predicts CO2 emissions based on traffic data. The project fetches real-time traffic conditions using the Google Places API and uses linear regression to estimate CO2 emissions.

📁 Project Structure

```
SmartCityCO2Predictor/
├── data/
│   ├── city_co2_emissions.csv  # Dataset containing city traffic and CO2 emission data
├── models/
│   ├── co2_predictor_model.joblib  # Trained machine learning model
├── src/
│   ├── main.py  # Main script to run the program
├── .gitignore  # Git ignore file
├── README.md  # Project documentation
├── requirements.txt  # List of dependencies
```

🚀 Installation & Setup

1️⃣ Clone the repository

```bash
git clone https://github.com/Kirankumarvel/SmartCityCO2Predictor.git
cd SmartCityCO2Predictor
```

2️⃣ Create a virtual environment (Optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

4️⃣ Set up the Google API Key

Create a `.env` file in the root directory and add:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

Alternatively, set it in your terminal:

```bash
export GOOGLE_API_KEY=your_google_api_key_here  # On Windows use: set GOOGLE_API_KEY=your_google_api_key_here
```

📊 Usage

Run the main script to predict CO2 emissions for a city:

```bash
python src/main.py
```

🛠 Dependencies

- Python 3.x
- requests
- numpy
- pandas
- scikit-learn
- joblib
- colorama
- python-dotenv

Install dependencies using:

```bash
pip install -r requirements.txt
```

🤝 Contributing

Feel free to fork this project and submit pull requests.

📜 License

This project is licensed under the MIT License.
```


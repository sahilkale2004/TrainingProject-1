# Manufacturing Efficiency Analysis System

A full-stack machine learning application that predicts the production efficiency of manufacturing processes using real-time sensor and process data. It uses a PyTorch deep learning model on the backend (FastAPI) and a Streamlit frontend for the user interface.

## Project Structure
- `training/`: Contains the Jupyter notebook (`train.ipynb`) and script (`train_local.py`) for training the model, along with the dataset (`manufacturing_dataset.csv`).
- `backend/`: FastAPI server for model inference and prediction.
- `frontend/`: Streamlit web interface for inputting manufacturing data and viewing predictions.
- `model/`: Directory where the trained model weights (`efficiency_model.pth`) and the data preprocessor (`preprocessor.pkl`) are stored.

## Technical Details

### Backend
- **Framework**: FastAPI
- **Model Framework**: PyTorch
- **Features**: Contains a PyTorch fully connected neural network (`EfficiencyNet`) with 3 linear layers (using ReLU activation) for predicting a continuous efficiency score.
- **API Endpoints**:
  - `POST /predict`: Accepts manufacturing and process data and returns the `predicted_efficiency_score`.

### Frontend
- **Framework**: Streamlit
- **Features**: Provides a user-friendly form collecting 17 operational features needed for prediction (such as Injection Temperature, Machine Age, Shift, Material Viscosity, etc.). It communicates with the backend API to retrieve the predicted efficiency score.

## Local Setup & Workflow

### 1. Training the Model (Optional/Google Colab)
If you want to train the model yourself:
1. Open `training/train.ipynb` in Google Colab or run it locally.
2. Upload `training/manufacturing_dataset.csv` if using Colab.
3. Run all cells to train the PyTorch model.
4. Download the generated `efficiency_model.pth` and `preprocessor.pkl`.

### 2. Environment Setup
1. Ensure the downloaded `efficiency_model.pth` and `preprocessor.pkl` are located in the `model/` directory.
2. Install project dependencies from the root of the project:
   - For backend: `pip install -r backend/requirements.txt`
   - For frontend: `pip install -r frontend/requirements.txt`

### 3. Starting the Project Locally

**Step A: Start the Backend (FastAPI)**
Run this command from the project root:
```bash
python backend/main.py
```
The backend API will start on `http://0.0.0.0:8001`.

**Step B: Start the Frontend (Streamlit)**
In a new terminal window, run this command from the project root:
```bash
streamlit run frontend/app.py
```
This will launch the web UI, typically hosted at `http://localhost:8501`.

## Deployment

### GitHub
To version control this project:
1. Initialize the repo: `git init`
2. Add files: `git add .`
3. Commit changes: `git commit -m "Initial commit"`
4. Add remote and push to main branch.

### Render Deployment
Connect your Github repository to Render.

**Backend Web Service**:
- **Build Command**: `pip install -r backend/requirements.txt`
- **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port 10000`

**Frontend Web Service** (If not using Streamlit Community Cloud):
- **Build Command**: `pip install -r frontend/requirements.txt`
- **Start Command**: `streamlit run frontend/app.py --server.port 10000 --server.address 0.0.0.0`

> **Note**: Update the `backend_url` in `frontend/app.py` to match your live backend URL before deploying the frontend.

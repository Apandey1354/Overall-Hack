# Complete Setup Guide - "PODIUM"
Make sure you have set this project up before: https://github.com/Apandey1354/Telemetry
Extract these datasets into the root directory: https://drive.google.com/drive/folders/1vNcNNVD3SZq4BdemxKxylKs3wP8lzFgj?usp=sharing

This comprehensive guide will help you set up and run the entire project, including the backend API, frontend dashboard, and all components.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Project Setup](#initial-project-setup)
3. [Project Structure](#project-structure)
4. [Running the Backend](#running-the-backend)
5. [Running the Dashboard](#running-the-dashboard)
6. [Code Architecture & Components](#code-architecture--components)
7. [Troubleshooting](#troubleshooting)
8. [Quick Reference](#quick-reference)

---

## Prerequisites

Before starting, ensure you have the following installed:

### Required Software

- **Python 3.9+** (Python 3.10 or 3.11 recommended)
  - Check version: `python --version`
  - Download: https://www.python.org/downloads/

- **Node.js 18+** and npm
  - Check version: `node --version` and `npm --version`
  - Download: https://nodejs.org/

- **Git** (optional, for version control)
  - Download: https://git-scm.com/downloads

### System Requirements

- **RAM**: Minimum 8GB (16GB recommended for large telemetry processing)
- **Storage**: At least 5GB free space for data and dependencies
- **Operating System**: Windows, macOS, or Linux

---

## Initial Project Setup

### Step 1: Clone or Download the Project

If you have the project in a repository:
```bash
git clone <repository-url>
cd "Overall Hack"
```

Or navigate to your project directory:
```bash
cd "C:\Users\apand\OneDrive\Desktop\Overall Hack"
```

### Step 2: Create Python Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt, indicating the virtual environment is active.

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages including:
- FastAPI & Uvicorn (API framework)
- PyTorch (Machine Learning)
- Pandas & NumPy (Data processing)
- Scikit-learn (ML models)
- OpenAI (AI recommendations)
- And many more...

**Expected installation time**: 5-15 minutes depending on your internet speed.

### Step 4: Install Frontend Dependencies

```bash
cd dashboard
npm install
cd ..
```

**Expected installation time**: 2-5 minutes.

### Step 5: Build Dashboard Cache (Optional but Recommended)

This pre-generates cached data for faster dashboard loading:

```bash
python scripts/build_dashboard_cache.py
```

This creates JSON cache files in `data/cache/`:
- `track_dna_summary.json` - Track characteristics
- `championship_state.json` - Championship standings
- `track_coach_data.json` - Coach recommendations
- `driver_embeddings.json` - Driver skill vectors

**Expected time**: 1-3 minutes.

---

## Project Structure

```
Overall Hack/
├── 📁 barber/                          # Barber Motorsports Park race data
├── 📁 COTA/                            # Circuit of the Americas race data
├── 📁 indianapolis/                   # Indianapolis Motor Speedway race data
├── 📁 virginia-international-raceway/ # VIR race data
│
├── 📁 config/                         # Configuration files
│   └── config.yaml                    # Main project configuration
│
├── 📁 data/                           # Processed data directory
│   ├── cache/                         # Pre-generated JSON caches
│   │   ├── track_dna_summary.json
│   │   ├── championship_state.json
│   │   ├── track_coach_data.json
│   │   └── driver_embeddings.json
│   ├── features/                     # Engineered features
│   ├── processed/                    # Cleaned data
│   └── raw/                          # Raw data (if organized)
│
├── 📁 dashboard/                      # Frontend React application
│   ├── src/
│   │   ├── components/               # React components
│   │   │   ├── SimulationRunner.tsx
│   │   │   ├── TrackCoachInterface.tsx
│   │   │   ├── ScenarioPlayground.tsx
│   │   │   ├── DriverEmbeddingsView.tsx
│   │   │   ├── TrackDNAView.tsx
│   │   │   ├── KarmaDisplay.tsx
│   │   │   ├── Chatbot.tsx
│   │   │   └── ...
│   │   ├── App.tsx                   # Main app component
│   │   ├── config.ts                 # Frontend configuration
│   │   └── hooks/                    # React hooks
│   ├── package.json                  # Frontend dependencies
│   └── vite.config.ts                # Vite build configuration
│
├── 📁 Karma_detector/                # Mechanical Karma system
│   ├── backend/                      # Karma backend (separate system)
│   └── frontend/                     # Karma frontend (separate system)
│
├── 📁 logs/                          # Application logs
├── 📁 models/                        # Trained ML models
│   └── transfer_model.pt            # Transfer learning model
│
├── 📁 notebooks/                     # Jupyter notebooks
│   ├── exploration/                  # EDA notebooks
│   └── experiments/                  # Model experiments
│
├── 📁 scripts/                        # Utility scripts
│   ├── build_dashboard_cache.py      # Cache generation script
│   ├── train_transfer_model.py      # Model training script
│   └── ...
│
├── 📁 src/                           # Main source code
│   ├── api/                          # FastAPI backend
│   │   ├── dashboard_api.py         # Main API endpoints
│   │   ├── ai_recommendation_service.py  # AI recommendations
│   │   ├── chatbot_service.py       # Chatbot logic
│   │   ├── eda_service.py           # EDA data service
│   │   └── cache_utils.py           # Cache utilities
│   │
│   ├── championship/                 # Championship simulation
│   │   ├── championship_simulator.py    # Main simulator
│   │   ├── championship_data_processor.py
│   │   └── butterfly_effect_analyzer.py
│   │
│   ├── data_processing/              # Data processing modules
│   │   ├── data_loader.py            # Data loading utilities
│   │   ├── track_dna_extractor.py    # Track DNA extraction
│   │   ├── driver_embedder.py        # Driver embeddings
│   │   ├── track_clustering.py       # Track clustering
│   │   ├── track_performance_analyzer.py
│   │   └── karma_stream.py           # Mechanical Karma computation
│   │
│   ├── models/                       # ML models
│   │   ├── transfer_learning_model.py    # Transfer learning model
│   │   ├── track_coach.py            # Track coach system
│   │   └── track_coaches/            # Venue-specific coaches
│   │       ├── barber_coach.py
│   │       ├── cota_coach.py
│   │       ├── indy_coach.py
│   │       └── vir_coach.py
│   │
│   └── visualization/                # Visualization utilities
│
├── 📁 tests/                         # Unit tests
│
├── 📄 requirements.txt               # Python dependencies
├── 📄 SETUP.md                      # This file
├── 📄 QUICK_START.md                # Quick start guide
└── 📄 README.md                     # Data documentation
```

---

## Running the Backend

The backend is a FastAPI application that provides REST API endpoints for the dashboard.

### Step 1: Activate Virtual Environment

Make sure your virtual environment is activated:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 2: Start the Backend Server

```bash
python -m uvicorn src.api.dashboard_api:app --reload --port 8000
```

**Alternative command:**
```bash
uvicorn src.api.dashboard_api:app --reload --port 8000
```

### Step 3: Verify Backend is Running

You should see output like:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Test the API:**
- Open your browser and visit: `http://localhost:8000/api/status`
- You should see a JSON response with status information

### Backend API Endpoints

The backend provides these main endpoints:

- `GET /api/status` - API health check
- `GET /api/tracks` - List all tracks
- `GET /api/drivers` - List all drivers
- `GET /api/track-dna/{track_id}` - Get track DNA for a track
- `GET /api/driver-embeddings/{driver_number}` - Get driver embeddings
- `GET /api/championship/state` - Get championship standings
- `POST /api/championship/simulate` - Run championship simulation
- `POST /api/scenario/analyze` - Analyze scenario changes
- `POST /api/coach/recommendations` - Get track coach recommendations
- `GET /api/karma/vehicles` - List vehicles with karma data
- `GET /api/karma/{vehicle_id}` - Get mechanical karma scores
- `POST /api/karma/upload` - Upload telemetry data
- `POST /api/ai/coach-recommendations` - AI-powered coach advice
- `POST /api/ai/scenario-recommendations` - AI scenario analysis
- `POST /api/chatbot` - Chatbot responses

**Keep the backend terminal open** - the server needs to keep running while you use the dashboard.

---

## Running the Dashboard

The dashboard is a React application built with Vite, TypeScript, and Tailwind CSS.

### Step 1: Navigate to Dashboard Directory

```bash
cd dashboard
```

### Step 2: Start the Development Server

```bash
npm run dev
```

### Step 3: Access the Dashboard

You should see output like:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

**Open your browser** and navigate to: **http://localhost:5173**

### Dashboard Features

The dashboard includes these main tabs:

1. **Project Overview** - Project introduction and features
2. **Drivers and Tracks Analysis** - Visualize driver embeddings and track DNA
3. **Simulate Championship** - Run Monte Carlo simulations
4. **Scenario Lab** - What-if scenario analysis
5. **Coaching and Realtime Telemetry Analysis** - Track coach recommendations
6. **Chatbot** - AI assistant for project questions

### Building for Production

To create a production build:
```bash
cd dashboard
npm run build
```

The built files will be in `dashboard/dist/`.

---

## Code Architecture & Components

### Backend Components (`src/api/`)

#### `dashboard_api.py` - Main API Server
- **Purpose**: FastAPI application with all REST endpoints
- **Key Features**:
  - CORS middleware for frontend communication
  - Data loading and caching
  - Championship simulation endpoints
  - Scenario analysis endpoints
  - Karma detector endpoints
  - AI recommendation endpoints
- **Dependencies**: Uses `DataLoader`, `ChampionshipSimulator`, `ButterflyEffectAnalyzer`

#### `ai_recommendation_service.py` - AI Recommendations
- **Purpose**: Generates AI-powered recommendations using OpenAI
- **Features**:
  - Coach recommendations based on driver/track analysis
  - Scenario recommendations for what-if analysis
  - Combined recommendations for comprehensive insights
- **Output**: Plain text recommendations (markdown stripped)

#### `chatbot_service.py` - AI Chatbot
- **Purpose**: Provides conversational AI assistant
- **Features**:
  - Answers questions about the project
  - Explains concepts (driver embeddings, track DNA, etc.)
  - Safety checks for irrelevant questions

#### `eda_service.py` - Exploratory Data Analysis
- **Purpose**: Generates EDA summaries and statistics
- **Features**:
  - Track statistics
  - Driver performance summaries
  - Race result analysis

#### `cache_utils.py` - Caching Utilities
- **Purpose**: Manages JSON cache generation
- **Features**:
  - Builds track DNA summaries
  - Builds championship state
  - Builds track coach data
  - Builds driver embeddings summaries

### Data Processing Components (`src/data_processing/`)

#### `data_loader.py` - Data Loading
- **Purpose**: Loads and processes race data from CSV files
- **Features**:
  - Handles multiple venue formats
  - Loads race results, lap times, telemetry
  - Handles missing data and anomalies
  - Supports multiple data formats

#### `track_dna_extractor.py` - Track DNA Extraction
- **Purpose**: Extracts track characteristics (DNA)
- **Features**:
  - Calculates track difficulty
  - Analyzes technical vs speed ratio
  - Extracts sector characteristics
  - Creates 20-dimensional track DNA vectors

#### `driver_embedder.py` - Driver Embeddings
- **Purpose**: Creates 8-dimensional skill vectors for drivers
- **Features**:
  - Technical skill component
  - Speed skill component
  - Balance/consistency component
  - Weather adaptability
  - Pressure handling
  - Track-specific performance

#### `track_clustering.py` - Track Clustering
- **Purpose**: Groups similar tracks together
- **Features**:
  - K-means clustering
  - Track similarity analysis
  - Cluster visualization

#### `track_performance_analyzer.py` - Performance Analysis
- **Purpose**: Analyzes driver performance across tracks
- **Features**:
  - Performance metrics calculation
  - Track-specific performance comparison
  - Statistical analysis

#### `karma_stream.py` - Mechanical Karma
- **Purpose**: Computes real-time component health scores
- **Features**:
  - Engine health (RPM, speed stress)
  - Gearbox health (gear usage, jolts)
  - Brakes health (pressure spikes)
  - Tires health (cornering, abrasion)
  - EMA smoothing for time series
  - Wear accumulation modeling

### Championship Components (`src/championship/`)

#### `championship_simulator.py` - Championship Simulator
- **Purpose**: Simulates full championship seasons
- **Key Classes**:
  - `PointsCalculator`: Calculates championship points
  - `RaceOutcomePredictor`: Predicts race results
  - `ChampionshipSimulator`: Main simulation engine
- **Features**:
  - Monte Carlo simulations (500+ iterations)
  - Weather variation modeling
  - Mechanical wear accumulation
  - Driver momentum factors
  - Psychological pressure modeling
  - Fatigue modeling
  - DNF probability calculation

#### `championship_data_processor.py` - Data Processing
- **Purpose**: Processes championship standings data
- **Features**:
  - Loads championship points
  - Tracks race-by-race results
  - Handles bonus points (pole, fastest lap)

#### `butterfly_effect_analyzer.py` - Butterfly Effect Analysis
- **Purpose**: Identifies critical championship moments
- **Features**:
  - Sensitivity analysis
  - Impact scoring
  - Scenario generation
  - What-if analysis

### Model Components (`src/models/`)

#### `transfer_learning_model.py` - Transfer Learning Model
- **Purpose**: Predicts driver performance on new tracks
- **Architecture**:
  - Input: 8-dim driver embedding + 5-dim performance vector + 20-dim track DNA
  - Output: Predicted lap time, speed, position
- **Features**:
  - PyTorch neural network
  - Transfer learning from known tracks to new tracks
  - Performance prediction

#### `track_coach.py` - Track Coach System
- **Purpose**: Provides venue-specific coaching advice
- **Features**:
  - Creates track-specific coaches
  - Generates actionable recommendations
  - Compares driver to optimal performance

#### `track_coaches/` - Venue-Specific Coaches
- **Files**: `barber_coach.py`, `cota_coach.py`, `indy_coach.py`, `vir_coach.py`
- **Purpose**: Each coach provides track-specific advice
- **Features**:
  - Track-specific recommendations
  - Sector-by-sector analysis
  - Weather adaptation strategies

### Frontend Components (`dashboard/src/components/`)

#### `App.tsx` - Main Application
- **Purpose**: Root component with navigation and routing
- **Features**:
  - Tab-based navigation
  - Data loading and state management
  - Error handling

#### `SimulationRunner.tsx` - Simulation Interface
- **Purpose**: UI for running championship simulations
- **Features**:
  - Simulation configuration
  - Progress tracking
  - Results visualization
  - Monte Carlo odds display

#### `TrackCoachInterface.tsx` - Track Coach UI
- **Purpose**: Interface for track coach recommendations
- **Features**:
  - Track and driver selection
  - Coach recommendations display
  - AI-powered insights

#### `ScenarioPlayground.tsx` - Scenario Analysis
- **Purpose**: What-if scenario testing interface
- **Features**:
  - Position modification
  - Impact visualization
  - Butterfly effect analysis

#### `DriverEmbeddingsView.tsx` - Driver Visualization
- **Purpose**: Visualizes driver skill embeddings
- **Features**:
  - 8-dimensional skill vector display
  - Interactive charts
  - Driver comparison

#### `TrackDNAView.tsx` - Track DNA Visualization
- **Purpose**: Visualizes track characteristics
- **Features**:
  - Track DNA vector display
  - Track comparison
  - Difficulty visualization

#### `KarmaDisplay.tsx` - Mechanical Karma Display
- **Purpose**: Shows real-time component health
- **Features**:
  - Component health cards
  - Time series charts
  - Trend indicators

#### `Chatbot.tsx` - AI Chatbot Interface
- **Purpose**: Conversational AI assistant
- **Features**:
  - Chat interface
  - Question answering
  - Project explanations

#### `ProjectOverview.tsx` - Project Introduction
- **Purpose**: Landing page with project overview
- **Features**:
  - Feature highlights
  - How it works explanations
  - Problem statements

---

## Troubleshooting

### Backend Issues

#### Port 8000 Already in Use
```bash
# Windows: Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use a different port
python -m uvicorn src.api.dashboard_api:app --reload --port 8001
```

#### Import Errors
```bash
# Make sure you're in the project root directory
cd "C:\Users\apand\OneDrive\Desktop\Overall Hack"

# Verify virtual environment is activated
# You should see (venv) in your prompt

# Reinstall dependencies
pip install -r requirements.txt
```

#### Module Not Found Errors
```bash
# Make sure Python can find the src module
# The project root should be in PYTHONPATH
# Running from project root should work:
python -m uvicorn src.api.dashboard_api:app --reload
```

### Frontend Issues

#### Port 5173 Already in Use
```bash
# Vite will automatically try the next available port
# Or specify a different port:
npm run dev -- --port 5174
```

#### Node Modules Issues
```bash
cd dashboard
rm -rf node_modules  # Windows: rmdir /s node_modules
rm package-lock.json
npm install
```

#### Build Errors
```bash
# Clear cache and rebuild
cd dashboard
rm -rf node_modules .vite dist
npm install
npm run dev
```

### Data Issues

#### Cache Not Found
```bash
# Regenerate cache
python scripts/build_dashboard_cache.py
```

#### Missing Data Files
- Ensure race data files are in the correct directories:
  - `barber/` - Barber Motorsports Park
  - `COTA/` - Circuit of the Americas
  - `indianapolis/` - Indianapolis Motor Speedway
  - `virginia-international-raceway/VIR/` - VIR

### API Connection Issues

#### Frontend Can't Connect to Backend
1. **Verify backend is running**: Visit `http://localhost:8000/api/status`
2. **Check CORS**: Backend has CORS enabled for all origins
3. **Check API URL**: Frontend uses `http://localhost:8000/api` by default
4. **Check browser console**: Look for error messages

#### CORS Errors
- The backend already has CORS middleware configured
- If you see CORS errors, check that the backend is running

### Performance Issues

#### Slow Simulation
- Reduce Monte Carlo iterations (default: 500, try 100-200)
- Use "Quick Simulation" option (100 iterations)
- Close other applications to free up RAM

#### Slow Dashboard Loading
- Build cache: `python scripts/build_dashboard_cache.py`
- Check browser console for errors
- Clear browser cache

---

## Quick Reference

### Starting Everything (Two Terminals)

**Terminal 1 - Backend:**
```bash
# Activate venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Start backend
python -m uvicorn src.api.dashboard_api:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd dashboard
npm run dev
```

Then open: **http://localhost:5173**

### Useful Commands

```bash
# Build dashboard cache
python scripts/build_dashboard_cache.py

# Run tests
pytest tests/

# Check Python version
python --version

# Check Node version
node --version

# List installed Python packages
pip list

# List installed npm packages
cd dashboard && npm list
```

### Important URLs

- **Frontend Dashboard**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Status**: http://localhost:8000/api/status
- **API Docs**: http://localhost:8000/docs (Swagger UI)

### File Locations

- **Cache Files**: `data/cache/`
- **Logs**: `logs/`
- **Models**: `models/`
- **Configuration**: `config/config.yaml`
- **Python Dependencies**: `requirements.txt`
- **Frontend Dependencies**: `dashboard/package.json`

---

## Next Steps

After setup is complete:

1. **Explore the Dashboard**: Navigate through all tabs to see features
2. **Run a Simulation**: Try the "Simulate Championship" tab
3. **Test Track Coach**: Select a track and driver in "Coaching and Realtime Telemetry Analysis"
4. **Try Scenarios**: Use "Scenario Lab" to test what-if scenarios
5. **Upload Telemetry**: Test the Mechanical Karma system with telemetry data

---

## Support

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Review error messages in terminal/console
3. Check that all dependencies are installed
4. Verify both backend and frontend are running
5. Ensure you're in the correct directory

For more information, see:
- `QUICK_START.md` - Quick start guide
- `README.md` - Data documentation
- `README_PROJECT.md` - Project overview

---

**Happy Racing! 🏁**


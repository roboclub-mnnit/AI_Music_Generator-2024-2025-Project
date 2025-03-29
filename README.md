# AI Music Generator

## Overview
AI Music Generator is an AI-powered music composition tool that enables users to generate unique melodies and drum patterns using deep learning models, specifically Long Short-Term Memory (LSTM) networks. The tool provides an interactive web interface for generating, playing, and downloading music in multiple formats.

## Features
- 🎵 **Melody Generation**: Generates unique musical melodies using an LSTM model.
- 🥁 **Drum Pattern Generation**: Creates rhythmic drum patterns with AI.
- 🎛️ **User Controls**: Adjust parameters like seed sequence, temperature, and sequence length.
- 🌐 **API Backend (FastAPI)**: Provides endpoints for AI-powered music generation.
- 🎹 **Multiple Audio Formats**: Supports MIDI, WAV, and MP3 exports.
- 📊 **Live Visualization**: Displays real-time frequency bars synchronized with playback.

## Technologies Used
- **Backend**: FastAPI, Python, TensorFlow/Keras (LSTM models)
- **Frontend**: React, Tailwind CSS
- **Audio Processing**: Fluidsynth, FFmpeg
- **Database**: JSON-based storage for history tracking

## Installation
### Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **Node.js and npm**
- **Fluidsynth** (for MIDI to WAV conversion)
- **FFmpeg** (for WAV to MP3 conversion)

### Setup Instructions
#### 1. Clone the Repository:
```sh
git clone https://github.com/roboclub-mnnit/AI_Music_Generator-2024-2025-Project
cd Music_Gen
```

#### 2. Backend Setup (FastAPI):
```sh
cd backend
pip install -r requirements.txt
```

#### 3. Frontend Setup (React):
```sh
cd music-gen-frontend
npm install
```

## Usage
### Running the Backend:
```sh
cd backend
uvicorn main:app --reload
```
Backend runs at **http://127.0.0.1:8000**.

### Running the Frontend:
```sh
cd music-gen-frontend
npm run dev
```
Frontend is available at **http://localhost:5173**.

## API Endpoints
| Method | Endpoint | Description |
|--------|-------------|-------------|
| `POST` | `/generate-melody` | Generates a melody based on input parameters |
| `POST` | `/generate-drums` | Generates a drum pattern |
| `GET` | `/download/{format}` | Downloads generated music in MIDI/WAV/MP3 format |

## Problems Faced
- **Model Training**: The model initially failed to learn the data and produced unsatisfactory results despite various architectural changes.
- **Model Generalization**: Poor generalization in early training; applied techniques like dropout and AdamW to improve performance.
- **React UI Issues**: Encountered inconsistent state management, making the frontend non-functional.
- **Frontend-Backend Integration**: Faced challenges in connecting the React frontend with the FastAPI backend.
- **File Type Conversions**: Difficulties in converting MIDI to WAV and MP3 using soundfont files on the backend.

## Future Plans
- Enhance the model’s training strategy for better music quality.
- Improve UI/UX for a more seamless user experience.
- Implement real-time music preview functionality.
- Expand model capabilities to generate multiple instruments beyond piano and drums.
- Deploy the project on AWS or Google Cloud for wider accessibility.

## Roadmap
- ✅ Implement LSTM-based melody and drum generation models.
- ✅ Develop FastAPI backend and React frontend.
- 🔄 Enhance music generation parameters.
- 🔄 Add real-time music preview.
- 🔄 Deploy on AWS/Google Cloud.

## Contributing
- Fork the repository and create a feature branch.
- Submit a pull request with detailed explanations.
- Report issues or feature requests.

## License
This project is licensed under the **MIT License**.

## Acknowledgments

- **Contributors:**
  - **Krishna Mohan**  
    [LinkedIn](https://www.linkedin.com/in/krishna-mohan-287259297) | [GitHub](https://github.com/kmohan321) 
  - **Aditya Sahani**  
    [LinkedIn](https://www.linkedin.com/in/adityasahani443/) | [GitHub](https://github.com/Aditya-en) 
  - **Pallavi Chahar**  
    [LinkedIn](https://www.linkedin.com/in/pallavichahar) | [GitHub](https://github.com/Pallavi2005-creator)

🎵 *Enjoy AI-powered music generation!* 🎵



# AquaIntel — Indian EEZ Intelligence

Web app to explore India’s EEZ with modules for Taxonomy & Morphology, eDNA, Otolith, Acoustic, Fishery Advisory, and a simple EEZ Chatbot. Built with Flask + Plotly and a clean dark UI.

## Features
- Landing with India EEZ image and facts
- Treemaps (taxonomy, eDNA), species search, EEZ map
- Acoustic map 
- Fishery advisory panel
- EEZ chatbot 

## Requirements
Create requirements.txt:
```
Flask
gunicorn
plotly
pandas
numpy
```

## Run locally
```
pip install -r requirements.txt
python app.py
# or
gunicorn -b 127.0.0.1:5000 app:app
```
Open:
- / (landing), /edna, /acoustic, /fishery, /chatbot

## Deploy on Render
- New Web Service → connect repo
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app`
- Set Environment: Python 3.x, Port 10000+ auto handled by Render

## Project structure
```
app.py
landing.html  edna.html  acoustic.html  fishery.html  chatbot.html
styles.css    attractive_ocean_map.html
taxonomy-morphology.csv  edna.csv  otolith.csv  
requirements.txt
```

Note: Replace `attractive_ocean_map.html` to update the Acoustic view.

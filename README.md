# ML1_Capstone Ideas
Project Title: "GaiaSearch: A Multi-Modal ML System for Missing Person Location Prediction"
The core idea is to create a system that ingests initial case data about a missing person and, over time, integrates new tips and data to predict the most likely areas where the person could be found.

1. Problem Definition & Core Value Proposition
Current missing person searches often rely on manual tip sorting and broad-area canvassing. This system aims to:

Prioritize Tips: Automatically analyze and geolocate tips from social media, hotlines, and public cameras.

Predict Search Zones: Use the person's profile and historical data to predict high-probability search areas on a map.

Assist Investigators: Provide a dynamic, data-driven dashboard for search and rescue teams, saving crucial time.

2. System Architecture & Key ML Components
Your system can be built as a pipeline with the following modules:

Module 1: Digital Footprint Analyzer & Tip Aggregator
Goal: Scrape and analyze public digital data related to the case.

ML Techniques:

Web Scraping & APIs: (Tools: Scrapy, BeautifulSoup, Tweepy for Twitter/X) to gather data from social media, news sites, and public forums using the missing person's name and aliases.

Named Entity Recognition (NER): (e.g., spaCy, BERT) to extract key information from text tips: locations, names, vehicle models, etc.

Sentiment Analysis & Credibility Scoring: A classifier to prioritize tips. A panicked "I just saw them at X!" would rank higher than a speculative "Maybe they went to Y." This could use a pre-trained model fine-tuned on a dataset of credible/incredible tips.

Geolocation from Text: Convert extracted location entities ("near the old train station on 5th Ave") into precise latitude and longitude coordinates using Geocoding APIs (Google Geocoding, OpenStreetMap Nomination).

Module 2: Profile-Based Location Predictor
Goal: Use the missing person's profile to generate initial likely locations.

Data Input:

Person's Profile: Home address, workplace/school, frequented locations (from family), medical conditions (e.g., dementia), hobbies, social circle.

Historical Data (Optional but powerful): Anonymized data from past missing person cases (age, condition, location, found location).

ML Techniques:

Geospatial Analysis & Clustering: Using the person's known locations as centroids. Apply a Gaussian Mixture Model (GMM) or DBSCAN to create a probability heatmap, giving higher weight to areas closer to home, work, and other key points. For individuals with dementia, research shows they are often found in "attractive nuisances" like bodies of water or dense vegetation within a certain radius.

Survival Analysis: If historical data is available, you can use survival analysis models (like Cox Proportional Hazards model) to estimate the probability of the person being in different types of terrain (urban, forest, water) based on time missing, age, and condition.

Module 3: Multi-Modal Data Fusion & Dynamic Heatmap Generator
Goal: This is the core. Combining the outputs from Module 1 (live tips) and Module 2 (static profile) into a single, dynamic probability heatmap.

ML Techniques:

Bayesian Filtering: This is a perfect fit. Treat the search area as a grid. Each cell has a prior probability (from Module 2). As new tips (evidence) come in from Module 1, update the probability of the person being in each cell using Bayes' Theorem.

Model: Implement a simplified Particle Filter. Imagine thousands of "particles" spread across the map, each representing a possible location of the person. The initial distribution is based on the profile. As a credible tip comes in from a specific area, you re-weight the particles, concentrating them around that location. The density of particles then becomes your live heatmap.

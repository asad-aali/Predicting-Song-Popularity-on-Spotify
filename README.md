# Predicting Song Popularity on Spotify

Medium Article: https://medium.com/p/e61074865251
Video Presentation: https://drive.google.com/file/d/1Rczh0jdUF3P9vRpVlBlbrW6IdRo9GveG/view

What makes you groove?

This is my final project for MIS 382N (Data Science), a graduate course at The University of Texas at Austin.

The project involves a comprehensive analysis of song attributes to identify the hits from the misses. Often we find ourselves preferring one genre over the other. And even in that, not all songs suit our taste. Some become instant hits while others fade in the huge list of the unknown. This curiosity to find out what makes a song popular or unpopular steered us in this direction. I set out with the aim to create a classification model to classify songs based on their features. To do this, I dug deep into the world of Spotify to build the dataset.

**Data Collection**
With so many platforms are available today for music, there is no dearth of data. I chose Kaggle and Spotify as reliable data sources and scraped around 30 thousand songs. From Spotify's API, along with songs, I got adjunct data breaking down the songs' attributes, for instance 'Tempo', 'Mode', 'Loudness' etc.

After acquiring the data, I shuffled and divided the dataset into train and test, keeping the ratio of the train to test as 2:1. The dependent variable for the classification is song popularity, depending on the above-identified attributes.

**Exploratory Data Analysis**
Before running predictive models on the dataset, I wanted to identify the correlation amongst the features and how they contribute to the dependent variable. I found that Popular songs have high values for attributes such as Mode , Valence and Energy. Unpopular songs have slightly higher Acousticness and Speechiness

Along with exploring songs and their attributes, I also wanted to look at songs from the perspective of time i.e what has changed between two time periods. So I explored songs before and after 2000.

I found that:
Songs made before 2000 had higher Mode, Acousticness and Valence
Songs made after 2000 had higher Energy, Danceability and Speechiness

Further when I generated feature distribution on the dataset, it confirmed the visualization results along with highlighting attributes that play a deterministic role in a song's outreach. Through this process I found that Popular songs have higher Duration and Valence range, while Unpopular songs tend to be a bit Loud and Speechy

**Classification Models**
After exploring the dataset, I applied Machine Learning models to classify the songs as either popular or unpopular. Firstly, I converted the Popularity feature (dependent variable) I scraped from Spotify into a binary variable (1 if song popularity is > 50) rather than a continuous one. Secondly, I noticed that the popularity labels were imbalanced. Hence, I had to apply resampling to the dataset to ensure that the models are able to learn features for both labels.

I ran a couple of Classification Models and chose AUC and Accuracy as the performance metrics.
XGB
Random Forest
Ensemble ( XGB + RF)

Random Forest and XGB Ensemble (XGB + RF) gave us the best result. Given that I was dealing with a real dataset with more than 50% of the data being used for testing, getting an AUC of more than 0.60 meant that the model was able to capture important features that can be linked to a song's popularity.

**Feature Importance**
It was interesting to note the importance of song features that I had used. For example, I were able to show that danceability, speechiness, energy, duration and tempo were the most important in predicting the popularity of a song.

**Analysis of Incorrect Predictions**
I also wanted to dive deeper into the model's predictions to pinpoint areas where the models performed poorly. For example, I can see that for incorrect model predictions, mode and valence seem to be the culprits. This made us realize that the models failed to use these features to make correct predictions, especially when their values were higher, due to which, I had to explore other features that could help us improve the predictive power.

**Additional Features using Natural Language Processing**
First, I scraped lyrics for the songs dataset from Genius.com in order to apply NLP models, and convert text into numerical features. Due to computational constraints, I took a smaller sample of the dataset.
**Hugging Face**: I used Hugging Face's transformers library to generate perplexity and negative log likelihood scores to add to the existing models. I made use of GPT-2 Large Model and chose a stride of 5. https://huggingface.co/docs/transformers/perplexity
**Genre**: Using Spotify API, I extracted the Genres mapped to artists and then mapped those Genres to the songs sung by those artists. The genres I extracted were converted into categorical variables using label encoding.
**Vader Sentiment Score**: I made use of Vader library in Python to generate compound sentiment scores for lyrics. The scores were between a scale of -1 to 1, and I added these to the feature set to see if the model's accuracy improves.
**Topic Modeling**: I used Latent Dirichlet Allocation (LDA) model in Python to categorize the lyrics into topics, and then I chose the most dominant topic for each lyric. I picked the top 5 topics that were dominant for each lyric, and then converted them into categorial features.

Once the features were added, I ran XGB and Random Forest again to observe whether the predictive power improved. After adding the features, I saw a greater than 2% increase in the test AUC and Accuracy scores using Random Forest. Finally, I applied weights to Random Forest and XGB model predictions (50% each) to come up with the best predictions and scores.

After running the new models, I wanted to see how the additional NLP features rank in terms of their importance. They are shown below, in descending order:
**Song Length
Vader Sentiment Scores
Perplexity Scores
Negative Log-Likelihood Scores
Topic Modeling**

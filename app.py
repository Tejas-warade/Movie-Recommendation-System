from flask import Flask, render_template, request, flash, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import pandas as pd
import requests

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'd35d3a9a'

# Load your movie data and compute similarity (use your actual file paths)
movies_data = pd.read_csv('content/movies.csv')
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

# Recommendation function
def get_recommendations(movie_name):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommendations.append(title_from_index)
        if len(recommendations) >= 10:  # Show a limited number of recommendations
            break
    
    return recommendations

# Function to fetch movie poster using the OMDB API
def get_movie_poster(movie_title):
    api_key = 'd35d3a9a'
    base_url = f'http://www.omdbapi.com/?i=tt3896198&apikey=d35d3a9a&t={movie_title}'
    response = requests.get(base_url)
    data = response.json()
    
    if response.status_code == 200 and data['Response'] == 'True':
        return data['Poster']
    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['movie_name']
        try:
            recommendations = get_recommendations(user_input)
            posters = [get_movie_poster(movie) for movie in recommendations]
            
            # Organize recommendations and posters in a list of tuples
            movie_data = list(zip(recommendations, posters))
            
            return render_template('index.html', movie_data=movie_data)
        except IndexError:
            flash('An error occurred. Please check your input or try again later.', 'error')
            return redirect(url_for('index'))
    return render_template('index.html', movie_data=[])

if __name__ == '__main__':
    app.run(debug=True)

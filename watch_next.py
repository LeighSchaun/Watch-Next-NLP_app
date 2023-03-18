# Import the necessary libraries
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained spacy model
nlp = spacy.load('en_core_web_md')

# Read the movies from the file and store them in a dictionary
with open('movies.txt', 'r') as f:
    movies = {}
    for line in f.readlines():
        title, description = line.strip().split(':')
        movies[title.strip()] = description.strip()

# Define a function to recommend a movie based on a given description
def recommend_movie(description):
    # Get the vector representation of the input description using the spacy model
    desc_vector = nlp(description).vector
    # Get the vector representation of each movie description in the dictionary
    movie_vectors = [nlp(movie).vector for movie in movies.values()]
    # Calculate the cosine similarity between the input description and each movie description
    sim_scores = cosine_similarity([desc_vector], movie_vectors)[0]
    # Get the index of the movie with the highest similarity score
    most_similar_idx = sim_scores.argmax()
    # Return the title of the most similar movie
    return list(movies.keys())[most_similar_idx]

# Example usage:
# Define the input description
description = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery and trained as a gladiator."
# Get the recommended movie based on the input description
recommended_movie = recommend_movie(description)
# Print the recommended movie
print("Recommended movie:", recommended_movie)

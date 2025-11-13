import streamlit as st
import pandas as pd
import pickle
import requests
import operator
from scipy import spatial
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# DB Management
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def main():

	st.title("RECOMMENDATION SYSTEM")

	menu = ["Login", "SignUp"]

	#Web_image
	from PIL import Image
	image = Image.open('netflix_movies_cover.webp')

	st.image(image, caption='Go ahead, make my day')
	#Go ahead, make my day from 'sudden impact' 1983

	choice = st.sidebar.selectbox("Menu",menu)

	# For Login
	if choice == "Login":
		st.subheader("")

		# Taking username, password
		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')

		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = make_hashes(password)
			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:
				st.success("Logged In as {}".format(username))

				# To fetch poster of movie
				def fetch_poster(movie_id):
					# responce = requests.get("https://api.themoviedb.org/3/movie/{""}?api_key=2d109c2cf39153f93acc291fde695357&language=en-US".format(
					# 	movie_id))
					# data = responce.json()
					# return "http://image.tmdb.org/t/p/w500/" + data['poster_path']
				
					try:
						response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=5bb21dc4c2d6e559d56d81130f46e15b&language=en-US")
						response.raise_for_status()  # Raise an exception for bad responses (e.g., 404)
						data = response.json()
						return "http://image.tmdb.org/t/p/w500/" + data['poster_path']
					except requests.exceptions.RequestException as e:
						# st.error(f"Error fetching poster for movie_id {movie_id}: {e}")
						return "https://wallpapercave.com/wp/wp9441125.jpg"


				# Recommendation based on KNN algorithm

				# function to find similarities between two movies
				def Similarity(movieId1, movieId2):
					a = movie_knn.iloc[movieId1]
					b = movie_knn.iloc[movieId2]

					genresA = a['genres_bin']
					genresB = b['genres_bin']

					genreDistance = spatial.distance.cosine(genresA, genresB)

					scoreA = a['cast_bin']
					scoreB = b['cast_bin']
					scoreDistance = spatial.distance.cosine(scoreA, scoreB)

					directA = a['director_bin']
					directB = b['director_bin']
					directDistance = spatial.distance.cosine(directA, directB)

					wordsA = a['words_bin']
					wordsB = b['words_bin']
					wordsDistance = spatial.distance.cosine(wordsA, wordsB)
					return genreDistance + directDistance + scoreDistance + wordsDistance

				# KNN based recommendatoin
				def recommend_knn(name):
					new_movie = movie_knn[movie_knn['original_title'].str.contains(name)].iloc[0].to_frame().T
					def getNeighbors(baseMovie, K):
						distances = []
						for index, movie in movie_knn.iterrows():
							if movie['new_id'] != baseMovie['new_id'].values[0]:
								dist = Similarity(baseMovie['new_id'].values[0], movie['new_id'])
								distances.append((movie['new_id'], dist))
						distances.sort(key=operator.itemgetter(1))
						neighbors = []
						for x in range(K):
							neighbors.append(distances[x])
						return neighbors
					# Recommend 10 movies
					K = 10
					neighbors = getNeighbors(new_movie, K)
					recommended_mov = []
					recommended_movies_pos = []
					for neighbor in neighbors:
						movie_id = movies_list.iloc[neighbor[0]].id
						#store movie that is recommend
						recommended_mov.append(movie_knn.iloc[neighbor[0]][0])
						#fetch poster for movies
						recommended_movies_pos.append(fetch_poster(movie_id))
					return recommended_mov, recommended_movies_pos

				movie_knn = pickle.load(open('movies_knn.pkl', 'rb'))
				movie_knn = pd.DataFrame(movie_knn)

				# Content based
				def recommend(movie):
					index = movies_list[movies_list['title'] == movie].index[0]
					distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

					recommended_movies = []
					recommended_movies_poster = []
					for i in distances[1:11]:
						movie_id = movies_list.iloc[i[0]].id
						# fetch poster
						recommended_movies.append(movies_list.iloc[i[0]].title)
						recommended_movies_poster.append(fetch_poster(movie_id))

					return recommended_movies, recommended_movies_poster

				movies_list = pickle.load(open('movies_list.pkl', 'rb'))
				movies_list = pd.DataFrame(movies_list)

				# similarities between movies
				#similarity = pickle.load(open('similarity_list.pkl', 'rb'))
				cv = CountVectorizer(max_features=5000, stop_words='english')
				vector = cv.fit_transform(movies_list['tags']).toarray()
				similarity = cosine_similarity(vector)


				# cnn

				movies = pickle.load(open('movies_cf.pkl', 'rb'))
				movies = pd.DataFrame(movies)

				# Show poster of movies
				show = movies_list['id']
				# converting movie tmdb id into array
				show = np.array(show)
				# to store poster of 24 movie
				show_poster = []
				for i in range(0,25):
					# fetch poster for 24 poster
					show_poster.append(fetch_poster(show[i]))

				show_name = movies_list['title']
				# converting movie list into array
				show_name = np.array(show_name)

				# creating columns for movies
				col1, col2, col3, col4, col5, col6 = st.columns(6)
				col7, col8, col9, col10, col11, col12 = st.columns(6)
				col13, col14, col15, col16, col17, col18 = st.columns(6)
				col19, col20, col21, col22, col23, col24 = st.columns(6)

				# showing poster and title of 24 movie
				with col1:
					st.image(show_poster[0])
					st.text(show_name[0])
				with col2:
					st.image(show_poster[1])
					st.text(show_name[1])
				with col3:
					st.image(show_poster[2])
					st.text(show_name[2])
				with col4:
					st.image(show_poster[3])
					st.text(show_name[3])
				with col5:
					st.image(show_poster[4])
					st.text(show_name[4])
				with col6:
					st.image(show_poster[5])
					st.text(show_name[5])
				with col7:
					st.image(show_poster[6])
					st.text(show_name[6])
				with col8:
					st.image(show_poster[7])
					st.text(show_name[7])
				with col9:
					st.image(show_poster[8])
					st.text(show_name[8])
				with col10:
					st.image(show_poster[9])
					st.text(show_name[9])
				with col11:
					st.image(show_poster[10])
					st.text(show_name[10])
				with col12:
					st.image(show_poster[11])
					st.text(show_name[11])
				with col13:
					st.image(show_poster[12])
					st.text(show_name[12])
				with col14:
					st.image(show_poster[13])
					st.text(show_name[13])
				with col15:
					st.image(show_poster[14])
					st.text(show_name[14])
				with col16:
					st.image(show_poster[15])
					st.text(show_name[15])
				with col17:
					st.image(show_poster[16])
					st.text(show_name[16])
				with col18:
					st.image(show_poster[17])
					st.text(show_name[17])
				with col19:
					st.image(show_poster[18])
					st.text(show_name[18])
				with col20:
					st.image(show_poster[19])
					st.text(show_name[19])
				with col21:
					st.image(show_poster[20])
					st.text(show_name[20])
				with col22:
					st.image(show_poster[21])
					st.text(show_name[21])
				with col23:
					st.image(show_poster[22])
					st.text(show_name[22])
				with col24:
					st.image(show_poster[23])
					st.text(show_name[23])

				# Heading
				st.title('MOVIE RECOMMENDATION SYSTEM')
				show_movie = movies_list['title'].values
				selected_movie = st.selectbox(
					'Enter the movie to get recommendation',
					show_movie)

				# show selected movie
				st.write('You selected:', selected_movie)

				if st.button('Recommend'):

					# Show content based Recommend
					#st.title('Content Based Recommmendation')
					st.title('Similar movies')
					names, poster = recommend(selected_movie)
					col1, col2, col3, col4, col5 = st.columns(5)
					col6, col7, col8, col9, col10 = st.columns(5)
					with col1:
						st.image(poster[0])
						st.text(names[0])
					with col2:
						st.image(poster[1])
						st.text(names[1])
					with col3:
						st.image(poster[2])
						st.text(names[2])
					with col4:
						st.image(poster[3])
						st.text(names[3])
					with col5:
						st.image(poster[4])
						st.text(names[4])
					with col6:
						st.image(poster[5])
						st.text(names[5])
					with col7:
						st.image(poster[6])
						st.text(names[6])
					with col8:
						st.image(poster[7])
						st.text(names[7])
					with col9:
						st.image(poster[8])
						st.text(names[8])
					with col10:
						st.image(poster[9])
						st.text(names[9])

					# show kNN based recommendation
					#st.title('kNN Based Recommmendation')
					st.title('Movies you might like')

					names_knn, poster_knn = recommend_knn(selected_movie)
					col1, col2, col3, col4, col5 = st.columns(5)
					col6, col7, col8, col9, col10 = st.columns(5)
					with col1:
						st.image(poster_knn[0])
						st.text(names_knn[0])
					with col2:
						st.image(poster_knn[1])
						st.text(names_knn[1])
					with col3:
						st.image(poster_knn[2])
						st.text(names_knn[2])
					with col4:
						st.image(poster_knn[3])
						st.text(names_knn[3])
					with col5:
						st.image(poster_knn[4])
						st.text(names_knn[4])
					with col6:
						st.image(poster_knn[5])
						st.text(names_knn[5])
					with col7:
						st.image(poster_knn[6])
						st.text(names_knn[6])
					with col8:
						st.image(poster_knn[7])
						st.text(names_knn[7])
					with col9:
						st.image(poster_knn[8])
						st.text(names_knn[8])
					with col10:
						st.image(poster_knn[9])
						st.text(names_knn[9])

					# if username == 'sarthak':
					# It is for user1 only.
					# show collaborative filtering result
					df = movies['title']
					# creating array of titles
					df = np.array(df)

					pos = movies['tmdbId']
					# creating array of tmdbid
					pos = np.array(pos)

					# to fetch poster
					poster_cf = []
					for i in pos:
						poster_cf.append(fetch_poster(i))

					#st.title('User-User Based Recommmendation')
					st.title('Other user also liked')

					col1, col2, col3, col4, col5 = st.columns(5)
					col6, col7, col8, col9, col10 = st.columns(5)
					with col1:
						st.image(poster_cf[0])
						st.text(df[0])
					with col2:
						st.image(poster_cf[1])
						st.text(df[1])
					with col3:
						st.image(poster_cf[2])
						st.text(df[2])
					with col4:
						st.image(poster_cf[3])
						st.text(df[3])
					with col5:
						st.image(poster_cf[4])
						st.text(df[4])
					with col6:
						st.image(poster_cf[5])
						st.text(df[5])
					with col7:
						st.image(poster_cf[6])
						st.text(df[6])
					with col8:
						st.image(poster_cf[7])
						st.text(df[7])
					with col9:
						st.image(poster_cf[8])
						st.text(df[8])
					with col10:
						st.image(poster_cf[9])
						st.text(df[9])
			else:
				#if username or password is wrong show warning
				st.warning("Incorrect Username/Password")
	# For SignUp of new user
	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")



if __name__ == '__main__':
	main()
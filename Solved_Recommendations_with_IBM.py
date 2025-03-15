# ## Table of Contents
# 
# I. [Exploratory Data Analysis]
# II. [Rank Based Recommendations]
# III. [User-User Based Collaborative Filtering]
# IV. [Content Based Recommendations]
# V. [Matrix Factorization]


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(
    'data/user-item-interactions.csv', 
    dtype={'article_id': int, 'title': str, 'email': str}
)

# Show df to get an idea of the data
df.head()


# Part I : Exploratory Data Analysis
# 
# Use the dictionary and cells below to provide some insight into the descriptive statistics of the data.
# 
# Are there any missing values? If so, provide a count of missing values. If there are missing values in `email`, assign it the same id value `"unknown_user"`.




# Some interactions do not have a user associated with it, assume the same user.
df.info()


# Count missing values for each column
missing_values_count = df.isnull().sum()
print(missing_values_count)



df[df.email.isna()]



# Fill email NaNs with "unknown_user"
df['email'] = df['email'].fillna('unknown_user')

# Verify there are no more missing values
print(df.isnull().sum())



# Check if no more NaNs 
df[df.email.isna()]


# `2.` What is the distribution of how many articles a user interacts with in the dataset?  Provide a visual and descriptive statistics to assist with giving a look at the number of times each user interacts with an article.



# What are the descriptive statistics of the number of articles a user interacts with?
# Group by email to count interactions for each user
user_interactions = df.groupby('email').size()

# Descriptive statistics for interactions
user_interactions_stats = user_interactions.describe()
print(user_interactions_stats)



# Create a plot of the number of articles read by each user
# Plot the number of articles read by each user
plt.figure(figsize=(10, 6))
plt.hist(user_interactions, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Articles')
plt.ylabel('Number of Users')
plt.title('Number of Users Reading Articles')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Create a plot of the number of times each article was read
# Count the number of users who interacted with each article
article_interactions = df.groupby('article_id').size()

# Plot the distribution of article usage
plt.figure(figsize=(10, 6))
plt.hist(article_interactions, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Users')
plt.ylabel('Number of Articles')
plt.title('Distribution of Article Usage')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Fill in the median and maximum number of user_article interactions below
# Calculate the median and maximum values
median_val = user_interactions.median()
max_views_by_user = user_interactions.max()

print(f"Median: {median_val}")
print(f"Max: {max_views_by_user}")


# `3.` Use the cells below to find:


unique_articles = df['article_id'].nunique()
print(f"Number of unique articles with interactions: {unique_articles}")
total_articles = df['article_id'].nunique() 
print(f"Total number of articles: {total_articles}")
unique_users = df['email'].nunique()
print(f"Number of unique users: {unique_users}")
user_article_interactions = len(df)
print(f"Number of user-article interactions: {user_article_interactions}")


# `4.` find the most viewed article_id, as well as how often it was viewed.  After talking to the company leaders, the `email_mapper` function was deemed a reasonable way to map users to ids.  There were a small number of null values, and it was found that all of these null values likely belonged to a single user (which is how they are stored using the function below).


# Count the number of times each article was viewed
article_views = df.groupby('article_id').size()

# Find the most viewed article ID and its view count
most_viewed_article_id = str(article_views.idxmax())
max_views = article_views.max()

print(f"The most viewed article ID: {most_viewed_article_id}")
print(f"The most views: {max_views}")



def email_mapper(df=df):
    coded_dict = {
        email: num 
        for num, email in enumerate(df['email'].unique(), start=1)
    }
    return [coded_dict[val] for val in df['email']]

df['user_id'] = email_mapper(df)
df_new = df.copy()
del df['email']

# show header
df.head()




# Rank-Based Recommendations
# 
# In this project, we don't actually have ratings for whether a user liked an article or not.  We only know that a user has interacted with an article. In these cases, the popularity of an article can really only be based on how often an article was interacted with.
# 
# `1.` Fill in the function below to return the **n** top articles ordered with most interactions as the top. Test your function using the tests below.



def get_top_articles(n, df=df):
    """
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    """
    # Count interactions per article and sort in descending order
    top_articles_df = df['title'].value_counts().head(n)
    
    # Extract the article titles
    top_articles = top_articles_df.index.tolist()
    
    return top_articles  # Return the top article titles

def get_top_article_ids(n, df=df):
    """
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article ids
    
    """
    # Count interactions per article_id and sort in descending order
    top_articles_df = df['article_id'].value_counts().head(n)
    
    # Extract the article IDs
    top_articles = top_articles_df.index.tolist()
    
    return top_articles  # Return the top article IDs



print(get_top_articles(10))
print(get_top_article_ids(10))


# Test your function by returning the top 5, 10, and 20 articles
top_5 = get_top_articles(5)
top_10 = get_top_articles(10)
top_20 = get_top_articles(20)


# Part III: User-User Based Collaborative Filtering
# 
# 
# `1.` Use the function below to reformat the **df** dataframe to be shaped with users as the rows and articles as the columns.  
# 
# * Each **user** should only appear in each **row** once.
# 
# 
# * Each **article** should only show up in one **column**.  
# 
# 
# * **If a user has interacted with an article, then place a 1 where the user-row meets for that article-column**.  It does not matter how many times a user has interacted with the article, all entries where a user has interacted with an article should be a 1.  
# 
# 
# * **If a user has not interacted with an item, then place a zero where the user-row meets for that article-column**. 



def create_user_item_matrix(df, fill_value=0):
    """
    INPUT:
    df - pandas dataframe with article_id, title, email columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    """
    # Create the user-item matrix with 1s and 0s
    user_item = df.groupby(['user_id', 'article_id']).size().unstack(fill_value=fill_value)
    
    # Convert the matrix into binary (1 for interaction, 0 for no interaction)
    user_item = user_item.applymap(lambda x: 1 if x > 0 else 0)
    
    return user_item  # Return the user-item matrix

# Create the user-item matrix
user_item = create_user_item_matrix(df)



# `2.` The function below which should take a user_id and provide an ordered list of the most similar users to that user (from most similar to least similar).  The returned result should not contain the provided user_id, as we know that each user is similar to him/herself. Because the results for each user here are binary, it (perhaps) makes sense to compute similarity as the dot product of two users. 


# Lets use the cosine_similarity function from sklearn
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_users(user_id, user_item=user_item, include_similarity=False):
    """
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    include_similarity - (bool) whether to include the similarity in the output
    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first
    
    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered list of user ids. If include_similarity is True, returns a list of lists
    where the first element is the user id and the second the similarity.
    """
    
    # Compute the cosine similarity between all users
    user_similarity = cosine_similarity(user_item)
    
    # Create a DataFrame for easy processing
    similarity_df = pd.DataFrame(user_similarity, index=user_item.index, columns=user_item.index)
    
    # Extract the similarity scores for the provided user_id
    user_similarities = similarity_df.loc[user_id].sort_values(ascending=False)
    
    # Remove the user's own ID
    user_similarities = user_similarities[user_similarities.index != user_id]
    
    # Create a list of similar user IDs and similarities
    if include_similarity:
        similar_users = list(user_similarities.items())
    else:
        similar_users = list(user_similarities.index)
    
    return similar_users


# `3.` Now that you have a function that provides the most similar users to each user, you will want to use these users to find articles you can recommend.  Complete the functions below to return the articles you would recommend to each user. 



def get_article_names(article_ids, df=df):
    """
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column in df)
    """
    # Filter the dataframe for the given article_ids and retrieve unique titles
    article_names = df[df['article_id'].isin(article_ids)]['title'].drop_duplicates().tolist()
    
    return article_names  # Return the article names associated with the list of article ids


def get_ranked_article_unique_counts(article_ids=None, user_item=user_item):
    """
    INPUT:
    article_ids - (list or None) a list of article_ids to filter for; if None, all articles are included
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise 
    
    OUTPUT:
    ranked_article_unique_counts - (list) a list of tuples with article_id and the number of 
                                   unique users that have interacted with the article,
                                   sorted by the number of unique users in descending order
    
    Description:
    Provides a list of the article_ids and the number of unique users that have
    interacted with the article using the user_item matrix, sorted by the number
    of unique users in descending order. If `article_ids` is provided, only
    those specific articles will be considered.
    """
    # Sum the interactions (1s) for each article across all users
    article_counts = user_item.sum(axis=0)  # Sum across rows (users)
    
    # Filter for specific article IDs if provided
    if article_ids is not None:
        article_counts = article_counts.loc[article_ids]
    
    # Sort the articles by interaction count in descending order
    ranked_article_unique_counts = article_counts.sort_values(ascending=False).items()
    
    # Create a list of tuples (article_id, count)
    ranked_article_unique_counts = [(article_id, int(count)) for article_id, count in ranked_article_unique_counts]
    
    return ranked_article_unique_counts




def get_user_articles(user_id, user_item=user_item):
    """
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column in df)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    """
    # Get the articles the user has interacted with
    article_ids = user_item.loc[user_id][user_item.loc[user_id] == 1].index.tolist()
    
    # Get the names of those articles
    article_names = get_article_names(article_ids, df)
    
    return article_ids, article_names  # Return the ids and names



def user_user_recs(user_id, m=10, user_item=user_item):
    """
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    recs - (list) a list of recommendations for the user
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    """
    # Get the most similar users
    similar_users = find_similar_users(user_id, user_item=user_item)
    
    # Get the articles seen by the target user
    user_articles, _ = get_user_articles(user_id, user_item)
    
    # Store recommendations
    recs = []
    
    for other_user in similar_users:
        other_user_articles, _ = get_user_articles(other_user, user_item)
        
        # Find articles not yet seen by the user
        new_recs = list(set(other_user_articles) - set(user_articles))
        
        # Add new recommendations, stopping at m
        recs.extend(new_recs)
        if len(recs) >= m:
            break
    
    return recs[:m]  # Return your recommendations for this user_id



# Check Results
get_article_names(user_user_recs(1, 10)) # Return 10 recommendations for user 1

get_ranked_article_unique_counts([1320, 232, 844])[0]



# `4.` Now we are going to improve the consistency of the **user_user_recs** function from above.  
# 
# * Instead of arbitrarily choosing when we obtain users who are all the same closeness to a given user - choose the users that have the most total article interactions before choosing those with fewer article interactions.
# * Instead of arbitrarily choosing articles from the user where the number of recommended articles starts below m and ends exceeding m, choose articles with the articles with the most total interactions before choosing those with fewer total interactions. This ranking should be  what would be obtained from the **top_articles** function you wrote earlier.


def get_top_sorted_users(user_id, user_item=user_item):
    """
    INPUT:
    user_id - (int)
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user
    
    Other Details - sort the neighbors_df by the similarity and then by the number of interactions 
                    where highest of each is higher in the dataframe, i.e., in descending order
    """
    # Calculate cosine similarity between users
    from sklearn.metrics.pairwise import cosine_similarity
    user_similarity = cosine_similarity(user_item)
    similarity_series = pd.Series(user_similarity[user_id - 1], index=user_item.index)
    
    # Calculate the number of interactions for each user
    num_interactions = user_item.sum(axis=1)
    
    # Combine into a dataframe
    neighbors_df = pd.DataFrame({
        'neighbor_id': user_item.index,
        'similarity': similarity_series,
        'num_interactions': num_interactions
    })
    
    # Remove the target user from the neighbors list
    neighbors_df = neighbors_df[neighbors_df['neighbor_id'] != user_id]
    
    # Sort by similarity and then by number of interactions (both descending)
    neighbors_df = neighbors_df.sort_values(by=['similarity', 'num_interactions'], ascending=False)
    
    return neighbors_df  # Return the sorted dataframe


def user_user_recs_part2(user_id, m=10, user_item=user_item):
    """
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
      before choosing those with fewer article interactions.

    * Choose articles with the most total interactions 
      before choosing those with fewer total interactions.
    """
    # Get sorted neighbors
    neighbors_df = get_top_sorted_users(user_id, user_item)
    
    # Get the articles viewed by the target user
    user_articles = user_item.loc[user_id][user_item.loc[user_id] == 1].index.tolist()
    
    # Initialize recommendations
    recs = []
    
    # Loop through sorted neighbors
    for neighbor_id in neighbors_df['neighbor_id']:
        # Get articles viewed by the neighbor
        neighbor_articles = user_item.loc[neighbor_id][user_item.loc[neighbor_id] == 1].index.tolist()
        
        # Find new articles to recommend
        new_recs = list(set(neighbor_articles) - set(user_articles))
        
        # Rank the new articles by total interactions (popularity)
        ranked_recs = get_ranked_article_unique_counts(user_item=user_item)
        ranked_recs = [article_id for article_id, _ in ranked_recs if article_id in new_recs]
        
        # Add to the recommendations list
        recs.extend(ranked_recs)
        
        # Stop if we have enough recommendations
        if len(recs) >= m:
            break
    
    # Limit to top m recommendations
    recs = recs[:m]
    
    # Get article names for the recommendations
    rec_names = get_article_names(recs)
    
    return recs, rec_names



# Quick spot check - don't change this code - just use it to test your functions
rec_ids, rec_names = user_user_recs_part2(20, 10)
print("The top 10 recommendations for user 20 are the following article ids:")
print(rec_ids)
print()
print("The top 10 recommendations for user 20 are the following article names:")
print(rec_names)


# `5.` Use the functions from above to correctly fill in the solutions to the dictionary below.  Then test your dictionary against the solution.  Provide the code you need to answer each following the comments below.


print(get_top_sorted_users(1, user_item=user_item).head(n=1))
print(get_top_sorted_users(2, user_item=user_item).head(n=10))
print(get_top_sorted_users(131, user_item=user_item).head(n=10))


### Tests with a dictionary of results
# Get sorted neighbors for user 1
neighbors_user1 = get_top_sorted_users(user_id=1, user_item=user_item)
user1_most_sim = neighbors_user1.iloc[0]['neighbor_id']  # Most similar user to user 1

# Get sorted neighbors for user 2
neighbors_user2 = get_top_sorted_users(user_id=2, user_item=user_item)
user2_6th_sim = neighbors_user2.iloc[5]['neighbor_id']  # 6th most similar user to user 2

# Get sorted neighbors for user 131
neighbors_user131 = get_top_sorted_users(user_id=131, user_item=user_item)
user131_10th_sim = neighbors_user131.iloc[9]['neighbor_id']  # 10th most similar user to user 131

# Create a dictionary to store the results
results = {
    "user1_most_sim": user1_most_sim,
    "user2_6th_sim": user2_6th_sim,
    "user131_10th_sim": user131_10th_sim
}

# Print the results
print(results)



# What would your recommendations be for this new user 0?  As a new user, they have no observed articles.
# Provide a list of the top 10 article ids you would give to 
# Get the ranked article IDs based on popularity
ranked_articles = get_ranked_article_unique_counts(user_item=user_item)

# Extract the top 10 article IDs
new_user_recs = [article_id for article_id, count in ranked_articles[:10]]

# Print the recommendations
print(f"Top 10 recommendations for new user: {new_user_recs}")


# Part IV: Content Based Recommendations

# `1.` Used the function bodies below to create a content based recommender function `make_content_recs`. We'll use TF-IDF to create a matrix based off article titles, and use this matrix to create clusters of related articles. You can use this function to make recommendations of new articles.

df.head()


from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD


# Extract unique articles based on article_id
df_unique_articles = df.drop_duplicates(subset='article_id')

# View the resulting DataFrame with unique articles
print(df_unique_articles.head())


# Create the vectorizer
vectorizer = TfidfVectorizer(
    max_df=0.75,
    min_df=5,
    stop_words="english",
    max_features=200
)

# Fit the vectorizer to the article titles
print("Running TF-IDF")
X_tfidf = vectorizer.fit_transform(df['title'])

print(f"n_samples: {X_tfidf.shape[0]}, n_features: {X_tfidf.shape[1]}")

# Create the LSA pipeline
lsa = make_pipeline(TruncatedSVD(n_components=50), Normalizer(copy=False))

# Fit the LSA model to the vectorized article titles
X_lsa = lsa.fit_transform(X_tfidf)

# Calculate explained variance of the SVD step
explained_variance = lsa.named_steps['truncatedsvd'].explained_variance_ratio_.sum()

print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")



# Let's map the inertia for different number of clusters to find the optimal number of clusters
# We'll plot it to see the elbow
inertia = []
clusters = 300
step = 10
max_iter = 5
n_init = 5
random_state = 42
for k in range(1, clusters, step):
    kmeans = KMeans(
        n_clusters=k,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
    ).fit(X_lsa)
    # inertia is the sum of squared distances to the closest cluster center
    inertia.append(kmeans.inertia_)
plt.plot(range(1, clusters, step), inertia)
plt.xlabel('Number of clusters')


# There appears to be an elbow about 50, so we'll use 50 clusters.

# Define the number of clusters based on the elbow method
n_clusters = 50  # Number of clusters

# Initialize the KMeans model
kmeans = KMeans(
    n_clusters=n_clusters,
    init="k-means++",  # Initialization method
    max_iter=100,      # Maximum number of iterations for a single run
    n_init=5,         # Number of times the algorithm runs with different centroids
    random_state=42    # Random seed for reproducibility
)

# Fit the model to the TF-IDF vectorized data (e.g., X_lsa)
kmeans.fit(X_lsa)

# View the cluster centers (optional)
print("Cluster centers:")
print(kmeans.cluster_centers_)



# create a new column `title_cluster` and assign it the kmeans cluster labels
# First we need to map the labels to df_unique_articles article ids and then apply those to df
# Step 1: Create a mapping of article_id to cluster label
article_cluster_map = {
    article_id: cluster_label for article_id, cluster_label in zip(df_unique_articles['article_id'], kmeans.labels_)
}

# Step 2: Map the cluster labels to the articles in the original DataFrame
df['title_cluster'] = df['article_id'].map(article_cluster_map)

# Verify the result
print(df[['article_id', 'title_cluster']].head())



# Let's check the number of articles in each cluster
np.array(np.unique(kmeans.labels_, return_counts=True)).T



def get_similar_articles(article_id, df=df):
    """
    INPUT:
    article_id - (int) an article id 
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    article_ids - (list) a list of article ids that are in the same title cluster
    
    Description:
    Returns a list of the article ids that are in the same title cluster
    """
    # Identify the title cluster of the given article_id
    title_cluster = df.loc[df['article_id'] == article_id, 'title_cluster'].values[0]
    
    # Get all articles within the same title cluster
    articles_in_cluster = df[df['title_cluster'] == title_cluster]['article_id'].tolist()
    
    # Remove the input article_id from the list
    articles_in_cluster = [id_ for id_ in articles_in_cluster if id_ != article_id]
    
    return articles_in_cluster



def make_content_recs(article_id, n, df=df):
    """
    INPUT:
    article_id - (int) an article id
    n - (int) the number of recommendations you want similar to the article id
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    n_ranked_similar_articles - (list) a list of article ids that are in the same title cluster ranked
                                by popularity
    n_ranked_article_names - (list) a list of article names associated with the list of article ids
    
    Description:
    Returns a list of the n most ranked similar articles to a given article_id based on the title
    cluster in df. Rank similar articles using the function get_ranked_article_unique_counts.
    """
    # Step 1: Identify the cluster for the given article_id
    title_cluster = df.loc[df['article_id'] == article_id, 'title_cluster'].values[0]
    
    # Step 2: Get all article IDs in the same cluster
    articles_in_cluster = df[df['title_cluster'] == title_cluster]['article_id'].tolist()
    
    # Remove the input article_id from the list
    articles_in_cluster = [id_ for id_ in articles_in_cluster if id_ != article_id]
    
    # Step 3: Rank articles in the cluster by popularity
    ranked_articles = get_ranked_article_unique_counts(user_item=user_item)
    
    # Filter ranked articles to include only those in the same cluster
    ranked_similar_articles = [article_id for article_id, _ in ranked_articles if article_id in articles_in_cluster]
    
    # Step 4: Select the top n articles
    n_ranked_similar_articles = ranked_similar_articles[:n]
    
    # Step 5: Get the article names for the top n articles
    n_ranked_article_names = get_article_names(n_ranked_similar_articles, df)
    
    return n_ranked_similar_articles, n_ranked_article_names



# Test out your content recommendations given artice_id 25
rec_article_ids, rec_article_titles = make_content_recs(25, 10)
print(rec_article_ids)
print(rec_article_titles)



# `2.` Now that we have put together your content-based recommendation system, we will use the cell below to write a summary explaining how our content based recommender works. 

# The content-based recommendation system clusters articles using semantic similarity from their titles. It uses TF-IDF vectorization and LSA for dimensionality reduction, and KMeans for clustering. Recommendations are articles from the same cluster as the given article, ranked by popularity using unique user interactions.

# Part V: Matrix Factorization
#
# quick look at the matrix
user_item.head()



from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score, recall_score, accuracy_score
# Using the full number of components which equals the number of columns
svd = TruncatedSVD(n_components=len(user_item.columns), n_iter=5, random_state=42)

u = svd.fit_transform(user_item)
v = svd.components_
s = svd.singular_values_ 
print('u', u.shape)
print('s', s.shape)
print('vt', v.shape)



num_latent_feats = np.arange(10, 700+10, 20)
metric_scores = []

for k in num_latent_feats:
    # restructure with k latent features
    u_new, vt_new = u[:, :k], v[:k, :]
    
    # take dot product
    user_item_est = abs(np.around(np.dot(u_new, vt_new))).astype(int)
    # make sure the values are between 0 and 1
    user_item_est = np.clip(user_item_est, 0, 1)
    
    # total errors and keep track of them
    acc = accuracy_score(user_item.values.flatten(), user_item_est.flatten())
    precision = precision_score(user_item.values.flatten(), user_item_est.flatten())
    recall = recall_score(user_item.values.flatten(), user_item_est.flatten())
    metric_scores.append([acc, precision, recall])
    
    
plt.plot(num_latent_feats, metric_scores, label=['Accuracy', 'Precision', 'Recall'])
plt.legend()
plt.xlabel('Number of Latent Features')
plt.title('Metrics vs. Number of Latent Features')


# `4.` From the above, we can't really be sure how many features to use, because simply having a better way to predict the 1's and 0's of the matrix doesn't exactly give us an indication of if we are able to make good recommendations. Given the plot above, what would you pick for the number of latent features and why?

# I would pick around 200 latent features, as this is where recall is relatively high but hasn't fully flattened out yet. This should provide a good trade-off between capturing useful patterns and avoiding overfitting.

# `5.` Using 200 latent features and the values of U, S, and V transpose we calculated above, create an article id recommendation function that finds similar article ids to the one provide.
# 
# Create a list of 10 recommendations that are similar to article with id 4.  The function should provide these recommendations by finding articles that have the most similar latent features as the provided article.



def get_svd_similar_article_ids(article_id, vt, user_item=user_item, include_similarity=False):
    """
    INPUT:
    article_id - (int) an article id
    vt - (numpy array) vt matrix from SVD
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    include_similarity - (bool) whether to include the similarity in the output
    
    OUTPUT:
    article_ids - (list) a list of article ids that are most similar to the input article_id
    
    Description:
    Returns a list of article ids similar using SVD factorization
    """
    # Step 1: Find the index of the given article_id
    article_idx = np.where(user_item.columns == article_id)[0][0]
    
    # Step 2: Compute cosine similarity
    # Transpose vt to get a matrix of articles x latent features
    vt_transposed = vt.T
    cos_sim = cosine_similarity(vt_transposed)
    
    # Step 3: Extract similarity scores for the given article
    article_similarities = cos_sim[article_idx]
    
    # Step 4: Sort articles by similarity (descending order)
    similar_articles = np.argsort(article_similarities)[::-1]
    
    # Step 5: Remove the input article itself from the recommendations
    similar_articles = similar_articles[similar_articles != article_idx]
    
    # Step 6: Get the top 10 similar articles
    if include_similarity:
        most_similar_items = [
            [user_item.columns[article], article_similarities[article]]
            for article in similar_articles[:10]
        ]
    else:
        most_similar_items = [
            user_item.columns[article] for article in similar_articles[:10]
        ]
    
    return most_similar_items



# Create a vt_new matrix with 200 latent features
k = 200
vt_new = v[:k, :]



# What is the article name for article_id 4?
print("Current article:", get_article_names([4], df=df)[0])


# What are the top 10 most similar articles to article_id 4?
rec_articles = get_svd_similar_article_ids(4, vt_new, user_item=user_item)[:10]



# What are the top 10 most similar articles to article_id 4?
get_article_names(rec_articles, df=df)







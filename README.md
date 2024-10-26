Recommender.py recommends you clothing brands based on your favorite book taking the following steps:
1. Gets 50 unique human emotions using OpenAI API
2. Gets 100 human clothing brands using OpenAI API. It can either read from local 'database.db' or generate one, depending on the user preference on updating brand list or not.
3. Embed user's favorite book in emotions space, generating association scores.
4. Embed each of clothing brands in emotions space, generating association scores.
5. Using cosine similarity, get the top number brands to recommend with highest similarity in scores.
#get the brands, go through 50 emotins at a time
#cosine: normalize first: l2 norm = 1
#give instructions on readme on where key goes 
#uses pydantic style


Here embeddings happen in a much smaller space of emotions (embedding dimension is emotions) as oppossed to ordinary, more common embeddings in a large space as more commonly done with openai api (read). 

Acknowledging that this is a method, #talk about options
database:
SQLite consisting of 3 tables: emotions (emotion_id, emotion), brands (brand_id, name, brand_info, scores_info, gpt), association_scores (emotion_id, brand_id, score)


# one module or package w 1 .py 
#adaptors that take in pydantic datatypes and will make into sql
#argparse

Cosine similarity:
The cosine similarity ranges from -1 to 1, where 1 indicating identical vectors (i.e., vectors point in the same direction), 0 indicating orthogonality (i.e., vectors are at a 90-degree angle to each other, no similarity) and -1 indicating opposite directions (i.e., vectors point in exactly opposite directions).
Represents similarity between feature vectors, quantifying similarity between two vectors based on their direction, irrespective of their magnitude.

I confirmed no need to l2 norm vectors for cosine similarity of sklearn.metrics.pairwise:
# Define your original vectors
A = np.array([[2, 3]])
B = np.array([[5, 4]])

# Calculate cosine similarity without normalization
cosine_sim_without_norm = cosine_similarity(A, B)

# L2 normalize the vectors
A_normalized = A / np.linalg.norm(A)
B_normalized = B / np.linalg.norm(B)

# Calculate cosine similarity with normalization
cosine_sim_with_norm = cosine_similarity(A_normalized, B_normalized)

# Print the outputs
print("Cosine Similarity without normalization:")
print(cosine_sim_without_norm[0][0])  # Output from unnormalized vectors

print("\nCosine Similarity with normalization:")
print(cosine_sim_with_norm[0][0])      # Output from normalized vectors
cosine_sim_without_norm[0][0]==cosine_sim_with_norm[0][0]




key=os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=key)
model = "gpt-4o-2024-08-06"

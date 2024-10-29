* **A clothing brand recommender**


Recommender.py recommends the user clothing brands based on their favorite book. 

Key features:
1. Here embeddings happen in a much smaller space of emotions (embedding dimension is emotions) as oppossed to ordinary, more common embeddings in a large space as more commonly done with openai api (read).

2. Recommender.py uses gpt at three different points: getting emotions, getting brands, embedding brands and/or favorite book in emotions space. For each of the three API call types, it uses Pydantic and the Structured Outputs feature to ensure adherance to JSON schema in the output (1) which makes database data modeling easier. "gpt-4o-2024-08-06" is the model of choice here, as it is among the gpt models with available structured output option. 

Workflow:
1. Gets 50 unique human emotions using the OpenAI API. Length is constant.

2. Gets 100 American clothing brands using the OpenAI API. This list can be updated depending on the user preference. Length is constant.

3. Using OpenAI API, it loops through the brands (step 2) and prompts the gpt model to assess the connection between each brand name and emotion (step 1) by assigning emotional association scores to brand-emotion pairs, at the same time. Scores reflect the association strength for each pair and fall within 0 and number of emotions (e.g. 50) range, provided as part of the prompt. The prompt also requests a brief explanation for each of the association scores. The retrieved scores along with the score explanations are stored in a local SQLite database (database.db).

4. Using OpenAI API, it prompts the gpt model to assess the connection between user's favorite book and emotion (step 1) by assigning emotional association scores to book and emotion pairs at the same time. Scores reflect the association strength for each pair and fall within 0 and number of emotions (e.g. 50) range, provided as part of the prompt. The prompt also requests a brief explanation for each of the association scores.

5. Using cosine similarity, it shows the user the top number of brands with the highest similarity to user's favorite book. Number of recommendations to be defined by user with a max of 10.


Data storage: 
* recommender.py uses the sqlite3 library to create 'database.db' with 3 tables: 
emotions (emotion_id, emotion), brands (brand_id, name, brand_info, scores_info, gpt), association_scores (emotion_id, brand_id, score). Note that scores_info is a brief explanation for each of the asscociation scores for each brand. With each run, recommender.py first ensures if all 3 tables are present. If less than 3 are present, it creates them all. If all are present, it then checks if the user wants the recommendations based on updated brands data. Given the value of update_brand_list argument, it either updates the brands table and regenerates association_scores or leaves the data as is. 

Usage: 
    $python recommender.py 'One hundred years of Solitude' 10 'no'

Arguments:
    favorite_book: user's favorite_book as string (only 1) 
    number_of_brand_recommendations: number of brand recommendations (max 10)
    update_brand_list: user's preference to update brand list or not (str: yes no)

OpenAI API key:
    To securely access your OpenAI API key from an environment variable, add it to your shell's configuration file. Access the key from the environment variable.


Notes:

* Since the emotional association scores (the input dimension values for cosine measurement) range from 0 to the number of emotions (e.g., 50), cosine similarity values are all positive.

Reference:
https://platform.openai.com/docs/guides/structured-outputs/introduction

Neda Jabbari 

## A clothing brand recommender

`recommender.py` recommends the user clothing brands based on their favorite book. 

## Usage 
`$python recommender.py 'One hundred years of Solitude' 10 'no'`

### Arguments:
`favorite_book`: user's favorite book (only 1) \
`number_of_brand_recommendations`: number of brand recommendations (max 10) \
`update_brand_list`: preference to update the brands data before getting the recommendations (str: yes no)

### API key
To securely access your key from an environment variable, add it to your shell's configuration file.

## About
### Key features

1. Here embeddings are in a space of emotions with embedding dimensions being individual emotions as oppossed to ordinary embeddings in a large space, more commonly done with OpenAI API. This makes them readily interpretable.

2. `recommender.py` uses GPT at three different points: getting emotions, getting brands, embedding brands and/or favorite book in emotions space. For each of the three API call types, it uses Pydantic and the Structured Outputs feature to ensure adherance to JSON schema in the output (reference below) which makes the database data modeling easier. "gpt-4o-2024-08-06" is the GPT model used here, as it is among the GPT models with available structured output option. 

### Workflow

1. Gets 50 unique human emotions using the OpenAI API. Length is constant.

2. Gets 100 American clothing brands using the OpenAI API. This list can be updated depending on the user preference. Length is constant.

3. Using OpenAI API, it loops through the brands (step 2) and prompts the GPT model to assess the connection between each brand name and emotion (step 1) by assigning emotional association scores to brand-emotion pairs, at the same time. Scores reflect the association strength for each pair and fall within 0 and number of emotions (e.g. 50) range, provided as part of the prompt. The prompt also requests a brief explanation for each of the scores. The retrieved scores along with the score explanations are stored in the local SQLite database, `database.db`.

4. Similar to workflow in step 3, it prompts the GPT model to assess the connection between user's favorite book and emotion (step 1) by assigning emotional association scores to book-emotion pairs at the same time. Scores fall within the same range, 0 and number of emotions (e.g. 50). The prompt also requests a brief explanation for each of the scores.

5. Using cosine similarity, it recommends the top brands with the highest similarity to user's favorite book. Number of recommendations is set by the user with a max of 10.

### Data storage

`recommender.py` uses the `sqlite3` library to create `database.db` with three tables: `emotions` (`emotion_id`, `emotion`), `brands` (`brand_id`, `name`, `brand_info`, `scores_info`, `gpt`), `association_scores` (`emotion_id`, `brand_id`, `score`). Note that `scores_info` is a brief explanation for each of the asscociation scores for each brand. With each run, `recommender.py` first checks if all three tables are present. If fewer than three are present, it creates them all. If all the tables are already present in the database, the script will check if the user has requested recommendations based on updated brand data. If `update_brand_list` is set to 'yes', it will refresh the brand dataset and re-generate association scores with the pre-existing emotions data before generating the recommendations.

##

Note that all the cosine similarity values are positive given the positive emotional association scores, the vector elements for cosine measurement.

Reference: https://platform.openai.com/docs/guides/structured-outputs/introduction

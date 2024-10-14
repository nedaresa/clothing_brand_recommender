import json
import pandas as pd
import os
import argparse

from itertools import islice
from sklearn.metrics.pairwise import cosine_similarity

from enum import Enum
from typing import List
from pydantic import BaseModel, Field

#!pip install openai -U
from openai import OpenAI


# Get human emotions
# Define the Pydantic model for the API response
class EmotionsResponse(BaseModel):
    Characteristics: List[str] = Field(None, description="List of non-redundant human emotions.")

def get_emotions(model: str, api_key: str) -> List[str]:
    """Gets a list of 50 unique and non-redundant human emotions using the specified gpt model."""
    client = OpenAI(api_key=api_key)
    # Define system and user prompts
    system_prompt = "Find 50 different, exclusive and unique human emotions. "\
    "For example, pick joy or happiness, pick Shame or Embarrassment, pick Envy or Jealousy, "\
    "pick Hate or disgust or hatered or Resentment. "\

    user_prompt = "Select 50 different and unique human emotions."

    try:
        #Call the API to get the completion
        completion = client.beta.chat.completions.parse(
            model= model,
            messages=[
                {"role": "system", "content": "Be a helpful assistant."},
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": "make sure to include either joy or happiness, not both."},
                {"role": "system", "content": "make sure to include either Shame or Embarrassment, not both"},
                {"role": "system", "content": "make sure to include either Envy or Jealousy, not both"},
                {"role": "system", "content": "make sure to include either Hate or disgust or hatered or Resentment"},
                {"role": "system", "content": "Check again to remove redundant emotions. I only want unique emotions."},
                {"role": "user", "content": user_prompt}
            ],
            response_format=EmotionsResponse
        )

        #output returns in the defined pydantic style
        output = completion.choices[0].message.parsed
        return output.json()
    
    except Exception as e:
        # Handle exceptions such as API errors, etc
        print(f"An error occurred: {e}")
        return json.dumps({})
    

#Get 100 best selling American clothing brands 

# Define the Pydantic model for the API response
class BrandsResponse(BaseModel):
    Brands: List[str] = Field(None, description="Brands as a list of strings.")

def get_brands(model: str, api_key: str) -> List[str]:
    """Get 100 best selling American clothing brands, both for adults and kids, using the specified gpt model."""

    client = OpenAI(api_key=api_key)

    try:
        #Call the API to get the completion
        completion = client.beta.chat.completions.parse(
            model= model,
            messages=[
                {"role": "system", "content": "Find 100 non-redundant best selling American clothing brands."},
                {"role": "user", "content": "Give me 100 best selling American clothing brands, both for adults and kids."}
            ],
            response_format=BrandsResponse
        )

        #output returns in the defined pydantic style
        output = completion.choices[0].message.parsed
        return output.json()
    
    except Exception as e:
        # Handle exceptions such as API errors, etc
        print(f"An error occurred: {e}")
        return json.dumps({})


# Embedding brand in emotions space: Get association scores between an input and list of emotions

def emotional_association_scores(
        thing, 
        model,
        emotions, api_key
    ):

    client = OpenAI(api_key=api_key)

    Characteristic = Enum('Characteristic', dict([(emotion, emotion) for emotion in emotions]))

    class EmotionalAssociationScore(BaseModel):
        emotion: Characteristic
        score: float

    class EmotionalAssociationScores(BaseModel):
        associations: List[EmotionalAssociationScore] = Field(description="A list of emotions and associated scores")


    prompt = f"Assign emotional association scores between {0} and {len(emotions)} for the provided thing. "\
    "Assign a score for each of the following emotions. Briefly, explain the reason behind the association score."\
    "Ensure the scores reflect the association strength for the specified thing. "\
    "Thing: "\
    f"{thing}"
            
    completion = client.beta.chat.completions.parse(
        model = model,
        messages=[
            {"role": "system", "content": "Be a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        response_format=EmotionalAssociationScores,
    )
    #output returns in the defined pydantic style
    output = completion.choices[0].message.parsed
    return thing, output.json()

#embedding thing in emotions space
def get_df(thing, model, emotions, api_key):
    gpt = emotional_association_scores(thing, model, emotions, api_key)
    data = list(json.loads(gpt[1]).values())[0]
    df = pd.DataFrame(data)
    df.rename(columns = {'score': gpt[0]}, inplace=True)
    df.set_index('emotion', inplace=True)
    return df

def get_dfs(things_ls, model, emotions, api_key):
    merged_df = pd.DataFrame()
    for thing in things_ls:
        new_df = get_df(thing, model, emotions, api_key)
        if merged_df.empty:
            merged_df = new_df
        else:
            merged_df = pd.merge(merged_df, new_df, left_index=True, right_index=True, how='outer')
    return merged_df


def get_similarity(df, dfs, number):
    similarities = dict()

    # Reshape Series to 2D array (required by cosine_similarity)
    s1 = df.values.reshape(1, -1)

    for col in list(dfs.columns):
        # Reshape
        s2= dfs[col].values.reshape(1, -1)

        cosine_sim = cosine_similarity(s1, s2)
        similarities[col]= cosine_sim[0][0]

    sorted_dict = dict(sorted(similarities.items(), key=lambda item: item[1], reverse = True))
    #print(sorted_dict)

    # Get the top number of recommendations based on similarity
    recommendations = list(dict(islice(sorted_dict.items(), number)).keys())
    return recommendations

def check_favorite_number(value):
    ivalue = int(value)
    if ivalue > 10:  #Desired maximum value here
        raise argparse.ArgumentTypeError(f"Favorite number must not exceed 10.")
    return ivalue

def main():
    parser = argparse.ArgumentParser(description="Enter your number one favorite book you've read (just one book) and the number of top matching brands.")
    parser.add_argument('favorite_book', type=str, help='Your number one favorite book')

    # Adding the favorite_number argument, ensuring it's an integer
    parser.add_argument('number_of_brand_recommendations', type=check_favorite_number, help='Number of top matching brands (max 10)')

    args = parser.parse_args()

    model = "gpt-4o-2024-08-06"
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
    
    thing = args.favorite_book
    number = args.number_of_brand_recommendations

    emotions_json = get_emotions(model= model, api_key = api_key)
    emotions = list(json.loads(emotions_json).values())[0]

    things_json = get_brands(model= model, api_key = api_key)
    things = list(json.loads(things_json).values())[0][:4]

    dfs = get_dfs(things, model, emotions, api_key = api_key)
    dfs_cleaned = dfs.dropna(axis=1)

    df = get_df(thing, model, emotions, api_key = api_key)
    df_cleaned = df.dropna(axis=1)

    result = get_similarity(df_cleaned, dfs_cleaned, number)
    
    #print(f'embedded brands{dfs_cleaned}')
    #print(f'embedded book{df}')
    

    # Print the output from the secondary script
    print(f'Given your favorite book, {thing}, the top {number} American clothing brands that match the most with your personality are {result}')

if __name__ == "__main__":
    main()




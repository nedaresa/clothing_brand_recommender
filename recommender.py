import json
import pandas as pd
import os
import sqlite3
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from enum import Enum
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI

# Get human emotions using the Pydantic model for the API response

class EmotionsResponse(BaseModel):
    #None as default if value not provided
    Emotions: List[str] = Field(None, description="List of non-redundant human emotions.") 

def get_emotions(model, key):
    """Gets a list of 50 unique and non-redundant human emotions using the specified gpt model."""
    client = OpenAI(api_key=key)

    system_prompt = "Find 50 different, exclusive and unique human emotions. "\
    "For example, pick joy or happiness, pick Shame or Embarrassment, pick Envy or Jealousy, "\
    "pick Hate or disgust or hatered or Resentment. "\

    user_prompt = "Select 50 different and unique human emotions."

    try:
        completion = client.beta.chat.completions.parse(
            model= model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=EmotionsResponse
        )

        #output in the defined pydantic style
        output = completion.choices[0].message.parsed
        return output.json()
    
    except Exception as e:
        print(f"An error occurred while trying to get emotions: {e}")
        return json.dumps({})


#Get 100 best selling American clothing brands
class BrandResponse(BaseModel):
    name: str = Field(description="Brand name as a string.")
    brand_info: str = Field(description="Brand information as a string.")
class BrandsResponse(BaseModel):
    brands: List[BrandResponse] = Field(description="A list of brand names and information.")

def get_brands(model, key):
    """Get 100 best selling American clothing brands using the specified gpt model. Provide a brief information about each brand."""
    client = OpenAI(api_key=key)
    try:
        #Call the API to get the completion
        completion = client.beta.chat.completions.parse(
            model= model,
            messages=[
                {"role": "system", "content": "Find 100 non-redundant best selling American clothing brands."},
                {"role": "user", "content": "Give me 100 best selling American clothing brands and a brief information about each brand."}
            ],
            response_format=BrandsResponse
        )
        #output in the defined pydantic style
        output = completion.choices[0].message.parsed

        return output.json()
    
    except Exception as e:
        print(f"An error occurred while trying to get brands: {e}")
        return json.dumps({})


# Embedding and getting association scores between an input and emotions
def get_scores(thing, emotions, model, key):

    Characteristic = Enum('Characteristic', dict([(emotion, emotion) for emotion in emotions]))

    class EmotionalAssociationScore(BaseModel):
        emotion: Characteristic
        score: float

    class EmotionalAssociationScores(BaseModel):
        associations: List[EmotionalAssociationScore] = Field(description="List of dictionaries with 'emotion' and 'score' as keys for each dictionary")
        explanation: str = Field(description="String explaining the association scores.")
    
    client = OpenAI(api_key=key)

    prompt = f"Assign emotional association scores, with each score reflecting the association strength between "\
    f"{thing} and each of the given {emotions}. Each score should be between 0 "\
    f"and {len(emotions)}. Briefly explain the reason behind the association scores."\
            
    completion = client.beta.chat.completions.parse(
        model = model,
        messages=[
            {"role": "system", "content": "Be a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        response_format=EmotionalAssociationScores,
    )
    #output in the defined pydantic style
    output = completion.choices[0].message.parsed
    return output.json()

def get_emotions_df(model, key):
    gpt = get_emotions(model, key)
    ls = list(json.loads(gpt).values())[0]
    df = pd.DataFrame(ls, columns = ['emotion'])
    
    #removing duplicates
    before = df.shape[0]
    value_counts = df['emotion'].value_counts() 
    duplicate_counts = dict(value_counts[value_counts >1])
    df.drop_duplicates(keep='first', inplace=True)
    after = df.shape[0]
    print(f'{before} emotions extracted, with {duplicate_counts} duplicate counts: {before-after} duplicate rows removed leaving {after} rows')

    df['emotion_id'] = df.index
    df = df[['emotion_id','emotion']]
    return df

def get_scores_one(thing, emotions_df, model, key):
    emotions = list(emotions_df['emotion'].values)
    gpt = get_scores(thing, emotions, model, key)
    info = json.loads(gpt)['explanation']

    df = pd.DataFrame(json.loads(gpt)['associations'])
    df.rename(columns = {'score': thing}, inplace=True)
    #merge to get emotion ids
    df = pd.merge(df, emotions_df, on ='emotion', how ='right')
    df.drop('emotion', axis = 1, inplace=True)
    df= df[['emotion_id',f'{thing}']]
    return ({thing: info}, df)

def get_scores_all(things, emotions_df, model, key):
    sinfo_all = []
    merged_df = pd.DataFrame()
    for thing in things:
        sinfo_one, df = get_scores_one(thing, emotions_df, model, key)
        sinfo_all.append(sinfo_one)
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='emotion_id', how='left')
    return (sinfo_all, merged_df)

def get_scores_brands(emotions_df, model, key):
    gpt = get_brands(model, key)
    brands_df = pd.DataFrame(list(json.loads(gpt).values())[0])

    #remove duplicates
    before = brands_df.shape[0]
    value_counts = brands_df['name'].value_counts() 
    duplicate_counts = dict(value_counts[value_counts >1])
    brands_df.drop_duplicates(subset='name', keep='first', inplace=True)
    after = brands_df.shape[0]
    print(f'{before} brands extracted, with {duplicate_counts} duplicate counts: {before-after} duplicate rows removed leaving {after} rows')

    brands_df.reset_index(inplace= True)
    brands_df.rename({'index':'brand_id'}, axis = 1, inplace = True)
    sinfo_all, scores_brands= get_scores_all(brands_df['name'], emotions_df, model, key)
    sinfo_all_df= pd.DataFrame([(k,v) for data in sinfo_all for k,v in data.items()], columns = ['name', 'scores_info'])
    brands_df = pd.merge(brands_df, sinfo_all_df, how = 'left', on ='name' )
    brands_df['gpt'] = model
    brands_df = brands_df[['brand_id','name','brand_info', 'scores_info','gpt']]

    scores_brands = pd.melt(scores_brands, id_vars='emotion_id', value_vars =list(scores_brands.columns))
    scores_brands.rename(columns = {'variable':'name','value':'score'}, inplace=True)
    scores_brands = pd.merge(scores_brands, brands_df[['brand_id','name']], on ='name', how ='inner')
    scores_brands.drop('name', axis= 1, inplace=True)
    scores_brands = scores_brands[['emotion_id','brand_id','score']]
    print(f"Got {scores_brands.shape[0]} emotion-brand association scores ({scores_brands['emotion_id'].nunique()} emotions and {scores_brands['brand_id'].nunique()} brands)")
    
    return (brands_df, scores_brands)

def get_similarity(brands_df, scores_thing, scores_brands, number):

    similarities = dict()
    scores_thing.sort_values(by='emotion_id',inplace=True)
    scores_thing.set_index('emotion_id', inplace=True)
    # Reshape to 2D array with 1 row and len(emotions) columns (required by cosine_similarity)
    s1 = scores_thing.values.reshape(1, -1)
    print('s1 shape',s1.shape)

    brand_ids = list(scores_brands['brand_id'].unique())
    for brand_id in brand_ids:
        scores_brand = scores_brands.loc[scores_brands['brand_id']==brand_id]
        scores_brand = scores_brand.sort_values(by='emotion_id')
        scores_brand= scores_brand.set_index('emotion_id')
        # Reshape to 2D array with 1 row and len(emotions) columns (required by cosine_similarity)
        s2= scores_brand['score'].values.reshape(1, -1)
        print('s2 shape',s1.shape)
        cosine_sim = cosine_similarity(s1, s2)
        similarities[brand_id]= cosine_sim[0][0]
    #replace brand id with name 
    name_id = dict(zip(brands_df['name'], brands_df['brand_id']))
    similarities = {k: similarities[v] for k, v in name_id.items() if v in similarities}
    sorted_s = sorted(similarities.items(), key=lambda item: item[1], reverse = True)
    recommendations = list(dict(sorted_s[:number]).keys())
    return recommendations

def check_favorite_number(value):
    ivalue = int(value)
    if ivalue > 10:  #Desired maximum value
        raise argparse.ArgumentTypeError(f"Number of top matching brands must not exceed 10.")
    return ivalue

def recommend_brands(thing, number, update_brand_list, model, key):
    with sqlite3.connect(os.path.abspath('database.db')) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name in ('emotions', 'brands', 'association_scores')")
        tables = cursor.fetchall()

        if len(tables) == 3:
            print('Reading emotions from database...')
            emotions_df = pd.read_sql_query("SELECT * FROM 'emotions'", conn) 

            if update_brand_list == 'no':
                print('Reading brands from database...')
                brands_df = pd.read_sql_query("SELECT * FROM 'brands'" , conn)
                scores_brands = pd.read_sql_query("SELECT * FROM 'association_scores'" , conn)
            else:
                print('Generating brands data...')
                brands_df, scores_brands = get_scores_brands(emotions_df, model, key)
                brands_df.to_sql('brands', conn, if_exists = 'replace', index=False)
                scores_brands.to_sql('association_scores', conn, if_exists = 'replace', index=False)
        else:
            print("The data that I need is not in 'database.db', generating...")
            emotions_df = get_emotions_df(model, key)
            brands_df, scores_brands = get_scores_brands(emotions_df, model, key)
            emotions_df.to_sql('emotions', conn, if_exists = 'replace', index=False)
            brands_df.to_sql('brands', conn, if_exists = 'replace', index=False)
            scores_brands.to_sql('association_scores', conn, if_exists = 'replace', index=False)
        
        _,scores_thing = get_scores_one(thing, emotions_df, model, key)
        result = get_similarity(brands_df, scores_thing, scores_brands, number)
    
    print(f'Given your favorite book, {thing}, here is the brand(s) I recommend: {result}')

    return result

def main():
    parser = argparse.ArgumentParser(description="Enter your number one favorite book you've read (just one book) and the number of top matching brands you'd like recommended.")
    parser.add_argument('favorite_book', type=str, help='Your number one favorite book')
    parser.add_argument('number_of_brand_recommendations', type=check_favorite_number, help='Number of top matching brands (max 10)')
    parser.add_argument('update_brand_list', choices=['yes','no'], help='Update the list of brands (yes or no)')

    args = parser.parse_args()

    thing = args.favorite_book
    number = args.number_of_brand_recommendations
    update_brand_list = args.update_brand_list
    model = "gpt-4o-2024-08-06"
    key = os.environ.get('OPENAI_API_KEY')
    if key is None:
        raise ValueError("API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
    
    recommend_brands(thing, number, update_brand_list, model, key)


if __name__ == "__main__":
    main()
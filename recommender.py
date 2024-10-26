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

def get_emotions(model: str, key: str) -> List[str]:
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

def get_brands(model: str, key: str) -> List[str]:
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
    df['emotion_id'] = df.index
    df = df[['emotion_id','emotion']]
    return df

def get_scores_thing(thing, emotions_df, model, key):
    emotions = list(emotions_df['emotion'].values)
    gpt = get_scores(thing, emotions, model, key)
    scoresinfo = json.loads(gpt)['explanation']

    df = pd.DataFrame(json.loads(gpt)['associations'])
    df.rename(columns = {'score': thing}, inplace=True)
    #merge to get emotion ids
    df = pd.merge(df, emotions_df, on ='emotion', how ='right')
    df.drop('emotion', axis = 1, inplace=True)
    df= df[['emotion_id',f'{thing}']]
    return ({thing: scoresinfo}, df)

def get_scores_things(things, emotions_df, model, key):
    scoreinfos = []
    merged_df = pd.DataFrame()
    for thing in things:
        scoreinfo, new_df = get_scores_thing(thing, emotions_df, model, key)
        scoreinfos.append(scoreinfo)
        if merged_df.empty:
            merged_df = new_df
        else:
            merged_df = pd.merge(merged_df, new_df, on='emotion_id', how='left')
    return (scoreinfos, merged_df)

def get_brands_scores(emotions_df, model, key):
    gpt = get_brands(model, key)

    brands_df = pd.DataFrame(list(json.loads(gpt).values())[0])
    brands_df.reset_index(inplace= True)
    brands_df.rename({'index':'brand_id'}, axis = 1, inplace = True)
    scoreinfos, scores_brands= get_scores_things(brands_df['name'], emotions_df, model, key)
    scoreinfos_df= pd.DataFrame([(k,v) for data in scoreinfos for k,v in data.items()], columns = ['name', 'scores_info'])
    brands_df = pd.merge(brands_df, scoreinfos_df, how = 'left', on ='name' )
    brands_df['gpt'] = model
    brands_df = brands_df[['brand_id','name','brand_info', 'scores_info','gpt']]

    scores_brands = pd.melt(scores_brands, id_vars='emotion_id', value_vars =list(scores_brands.columns))
    scores_brands.rename(columns = {'variable':'name','value':'score'}, inplace=True)
    scores_brands = pd.merge(scores_brands, brands_df[['brand_id','name']], on ='name', how ='inner')
    scores_brands.drop('name', axis= 1, inplace=True)
    scores_brands = scores_brands[['emotion_id','brand_id','score']]
    
    return (brands_df, scores_brands)

def get_similarity(brands_df, scores_thing, scores_brands, number):

    similarities = dict()
    scores_thing.sort_values(by='emotion_id',inplace=True)
    scores_thing.set_index('emotion_id', inplace=True)
    # Reshape Series to 2D array (required by cosine_similarity)
    s1 = scores_thing.values.reshape(1, -1)
    print('s1 shape',s1.shape)

    brand_ids = list(scores_brands['brand_id'].unique())
    for brand_id in brand_ids:
        scores_brand = scores_brands.loc[scores_brands['brand_id']==brand_id]
        scores_brand = scores_brand.sort_values(by='emotion_id')
        scores_brand= scores_brand.set_index('emotion_id')
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

def check_data_exists(thing, number, update_brand_list, model, key):
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
                brands_df, scores_brands = get_brands_scores(emotions_df, model, key)
                brands_df.to_sql('brands', conn, if_exists = 'replace', index=False)
                scores_brands.to_sql('association_scores', conn, if_exists = 'replace', index=False)
        else:
            print("The data that I need is not in the 'database.db' so need to get it fresh...")
            emotions_df = get_emotions_df(model, key)
            brands_df, scores_brands = get_brands_scores(emotions_df, model, key)
            emotions_df.to_sql('emotions', conn, if_exists = 'replace', index=False)
            brands_df.to_sql('brands', conn, if_exists = 'replace', index=False)
            scores_brands.to_sql('association_scores', conn, if_exists = 'replace', index=False)
        
        _,scores_thing = get_scores_thing(thing, emotions_df, model, key)
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
    
    check_data_exists(thing, number, update_brand_list, model, key)


if __name__ == "__main__":
    main()



#MVP---
# import json
# import pandas as pd
# import os
# import argparse
# from itertools import islice
# from sklearn.metrics.pairwise import cosine_similarity
# from enum import Enum
# from typing import List
# from pydantic import BaseModel, Field
# #!pip install openai -U
# from openai import OpenAI

# # Get human emotions using the Pydantic model for the API response
# class EmotionsResponse(BaseModel):
#     Characteristics: List[str] = Field(None, description="List of non-redundant human emotions.")

# def get_emotions(model: str, api_key: str) -> List[str]:
#     """Gets a list of 50 unique and non-redundant human emotions using the specified gpt model."""
#     client = OpenAI(api_key=api_key)

#     system_prompt = "Find 50 different, exclusive and unique human emotions. "\
#     "For example, pick joy or happiness, pick Shame or Embarrassment, pick Envy or Jealousy, "\
#     "pick Hate or disgust or hatered or Resentment. "\

#     user_prompt = "Select 50 different and unique human emotions."

#     try:
#         completion = client.beta.chat.completions.parse(
#             model= model,
#             messages=[
#                 {"role": "system", "content": "Be a helpful assistant."},
#                 {"role": "system", "content": system_prompt},
#                 {"role": "system", "content": "make sure to include either joy or happiness, not both."},
#                 {"role": "system", "content": "make sure to include either Shame or Embarrassment, not both"},
#                 {"role": "system", "content": "make sure to include either Envy or Jealousy, not both"},
#                 {"role": "system", "content": "make sure to include either Hate or disgust or hatered or Resentment"},
#                 {"role": "system", "content": "Check again to remove redundant emotions. I only want unique emotions."},
#                 {"role": "user", "content": user_prompt}
#             ],
#             response_format=EmotionsResponse
#         )

#         #output in the defined pydantic style
#         output = completion.choices[0].message.parsed

#         with open('emotions.json', 'w') as f:
#             json.dump(output.json(), f)
#         return output.json()
    
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return json.dumps({})

# #Get 100 best selling American clothing brands using the Pydantic model for the API response
# class BrandsResponse(BaseModel):
#     Brands: List[str] = Field(None, description="Brands as a list of strings.")

# def get_brands(model: str, api_key: str) -> List[str]:
#     """Get 100 best selling American clothing brands using the specified gpt model."""
#     client = OpenAI(api_key=api_key)
#     try:
#         #Call the API to get the completion
#         completion = client.beta.chat.completions.parse(
#             model= model,
#             messages=[
#                 {"role": "system", "content": "Find 100 non-redundant best selling American clothing brands."},
#                 {"role": "user", "content": "Give me 100 best selling American clothing brands."}
#             ],
#             response_format=BrandsResponse
#         )
#         #output in the defined pydantic style
#         output = completion.choices[0].message.parsed

#         with open('brands.json', 'w') as f:
#             json.dump(output.json(), f)

#         return output.json()
    
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return json.dumps({})

# # Embedding and getting association scores between an input and list of emotions
# def emotional_association_scores(
#         thing, 
#         model,
#         emotions, api_key
#     ):

#     client = OpenAI(api_key=api_key)

#     Characteristic = Enum('Characteristic', dict([(emotion, emotion) for emotion in emotions]))

#     class EmotionalAssociationScore(BaseModel):
#         emotion: Characteristic
#         score: float
#         explanation: str

#     class EmotionalAssociationScores(BaseModel):
#         associations: List[EmotionalAssociationScore] = Field(description="A list of emotions and associated scores")

#     prompt = f"Assign emotional association scores between {0} and {len(emotions)} for the provided thing. "\
#     "Assign a score for each of the following emotions. Briefly, explain the reason behind the association score."\
#     "Ensure the scores reflect the association strength for the specified thing. "\
#     "Thing: "\
#     f"{thing}"
            
#     completion = client.beta.chat.completions.parse(
#         model = model,
#         messages=[
#             {"role": "system", "content": "Be a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         response_format=EmotionalAssociationScores,
#     )
#     #output in the defined pydantic style
#     output = completion.choices[0].message.parsed
#     return thing, output.json()

# def get_df(thing, model, emotions, api_key):
#     gpt = emotional_association_scores(thing, model, emotions, api_key)
#     data = list(json.loads(gpt[1]).values())[0]
#     df = pd.DataFrame(data)
#     df.rename(columns = {'score': gpt[0]}, inplace=True)
#     print(df)
#     df.set_index('emotion', inplace=True)
#     return df

# def get_dfs(things, model, emotions, api_key):
#     merged_df = pd.DataFrame()
#     for thing in things:
#         new_df = get_df(thing, model, emotions, api_key)
#         if merged_df.empty:
#             merged_df = new_df
#         else:
#             merged_df = pd.merge(merged_df, new_df, left_index=True, right_index=True, how='outer')
#     return merged_df

# def get_similarity(df, dfs, number):
#     similarities = dict()
#     # Reshape Series to 2D array (required by cosine_similarity)
#     s1 = df.values.reshape(1, -1)

#     for col in list(dfs.columns):
#         s2= dfs[col].values.reshape(1, -1)
#         cosine_sim = cosine_similarity(s1, s2)
#         similarities[col]= cosine_sim[0][0]

#     sorted_dict = dict(sorted(similarities.items(), key=lambda item: item[1], reverse = True))

#     # Get the top number of recommendations based on similarity
#     recommendations = list(dict(islice(sorted_dict.items(), number)).keys())
#     return recommendations

# def check_favorite_number(value):
#     ivalue = int(value)
#     if ivalue > 10:  #Desired maximum value
#         raise argparse.ArgumentTypeError(f"Favorite number must not exceed 10.")
#     return ivalue

# def main():
#     parser = argparse.ArgumentParser(description="Enter your number one favorite book you've read (just one book) and the number of top matching brands.")
#     parser.add_argument('favorite_book', type=str, help='Your number one favorite book')
#     parser.add_argument('number_of_brand_recommendations', type=check_favorite_number, help='Number of top matching brands (max 10)')
#     parser.add_argument('update_brand_list', choices=['yes','no'], help='Update the list of brands (yes or no)')

#     args = parser.parse_args()

#     thing = args.favorite_book
#     number = args.number_of_brand_recommendations
#     update_brand_list = args.update_brand_list
#     model = "gpt-4o-2024-08-06"
#     api_key = os.environ.get('OPENAI_API_KEY')
#     if api_key is None:
#         raise ValueError("API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

#    # Retrieve emotions from datbase or through openAI API
#     if os.path.exists('emotions.json'):
#         with open('emotions.json', 'r') as f:
#             emotions_json = json.load(f)
#     else:
#         emotions_json = get_emotions(model, api_key)
#     emotions = list(json.loads(emotions_json).values())[0]
    
#     # Retrieve brand data based on user choice
#     if update_brand_list == 'yes':
#         things_json = get_brands(model, api_key)
#     else:  # update_brand_list == 'no'
#         with open('brands.json', 'r') as f:
#             things_json = json.load(f)

#     things = list(json.loads(things_json).values())[0][:4]

#     dfs = get_dfs(things, model, emotions, api_key)
#     dfs_cleaned = dfs.dropna(axis=1)

#     df = get_df(thing, model, emotions, api_key)
#     df_cleaned = df.dropna(axis=1)

#     result = get_similarity(df_cleaned, dfs_cleaned, number)

#     print(f'Given your favorite book, {thing}, the top {number} American clothing brands that match the most with your personality are {result}')

# if __name__ == "__main__":
#     main()




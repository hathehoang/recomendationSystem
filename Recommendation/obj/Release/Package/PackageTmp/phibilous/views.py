from django.http import HttpResponse, JsonResponse
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
import datetime

"define function"
def smooth_user_preference(x):
    return math.log(1+x, 2)

articles_df = {}
interactions_df = {}
interactions_from_selected_users_df = {}
interactions_full_df = {}
pivot_items_user_matrix = {}
item_sim_df = {}
all_user_predicted_ratings = {}
cf_preds_df = {}
users_items_pivot_matrix_df = {}
item_popularity_df = {}
list_new_item = {}

"Ratio between new item and recommended item"
newItemRatio = 0.2
recommendItemRatio = 0.8
defaultSimilarScore = 10

"encode event type"
def get_event_strength(x, duration = None):
    if x == '6' or x == 'ViewDuration':
        if duration >= 3000 and duration <= 10000:
            return 2 + (duration - 3000) / (10000 - 3000)  # the bonus value increase following by linear 
        if duration > 10000:
            return 3
        else:
            return "nothing"
    else:
        return event_type_strength.get(x, "nothing")
    
event_type_strength = {
   '4': 1.0, #"view"
   '1': 2.0, #"like"
   '2': 4, #"share"
   '3': -8, #"mark spam",
   '5': 5, #"contact"
   #'6': 2.0,#"view duration, if duration > 3 secs, begin set point. max is 10 secs, 10 sec = 3 points"
   'View' : 1.0,
   'Like' : 2.0,
   'Share' : 4,
   'MarkSpam' : -8,
   'Contact' : 5
   #'ViewDuration' : 2.0
}

def from_number_to_array(number):
    result = []
    for index in range(number):
        result.append(index)
    return result

def countNumberOfItemsUserRated(userId, interactionsDf):
    return interactionsDf[interactionsDf['userId'] == userId].shape[0]

def fake_created_date(contentId):
    if int(contentId % 4) == 0:
        return (datetime.datetime.now() + datetime.timedelta(days = -4) ).strftime('%Y-%m-%d %H:%M')
    if int(contentId % 4) == 1:
        return (datetime.datetime.now() + datetime.timedelta(days = -3) ).strftime('%Y-%m-%d %H:%M') 
    if int(contentId % 4) == 2:
        return (datetime.datetime.now() + datetime.timedelta(days = -2) ).strftime('%Y-%m-%d %H:%M')
    else:
        return (datetime.datetime.now() + datetime.timedelta(days = -1) ).strftime('%Y-%m-%d %H:%M')
    

"Initial dataframe"
def reTrainMatrix():
    global articles_df
    global interactions_df
    global interactions_from_selected_users_df
    global interactions_full_df
    global pivot_items_user_matrix
    global item_sim_df
    global all_user_predicted_ratings
    global cf_preds_df
    global users_items_pivot_matrix_df
    global item_popularity_df
    global list_new_item

    "import data frame of article"
    articles_df = pd.read_csv('http://phibious-uat-post.futurify.io/new_post.csv', encoding = "ISO-8859-1")  #change to uat - sit
    articles_df = articles_df[articles_df["contentId"] > 0]
    "Fake datetim"
    articles_df['createdDate'] = articles_df.apply(lambda x: fake_created_date(contentId = x['contentId']), axis = 1)
    "import data frame of user interaction"
    interactions_df = pd.read_csv('http://phibious-uat-post.futurify.io/newInteraction.csv', encoding = "ISO-8859-1") #change to uat- sit
  
    interactions_df['eventType'] = interactions_df.apply(lambda x: get_event_strength(x = x['eventType'], duration = x['durationInMilisecond']), axis=1)
    interactions_df = interactions_df[interactions_df['eventType'] != 'nothing']
    
    "get list new item( duration from created date and now is less than 2 and has less than 50 reactions on it)"
    interactions_count_on_item = interactions_df.groupby(['contentId']).size().to_frame('reactionsCount')
    interactions_count_on_item['contentId'] = interactions_count_on_item.index
    
    list_new_item = articles_df[(datetime.datetime.now() + datetime.timedelta(days = -2) ).strftime('%Y-%m-%d %H:%M') < articles_df['createdDate']].merge(interactions_count_on_item,
                                                                                                        how = 'inner',
                                                                                                        left_on = 'contentId',
                                                                                                        right_on = 'contentId')
    list_new_item = list_new_item[list_new_item['reactionsCount'] < 20].sort_values(by='createdDate',ascending=False)
    
    "take only take data of user that have with at least 5 reactions and contentId is not in list_new_item"

    interactions_df = interactions_df[~interactions_df['contentId'].isin(list_new_item['contentId'].values)]
    users_interactions_count_df = interactions_df.groupby(['userId', 'contentId']).size().groupby('userId').size()
    users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['userId']]

    interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
                   how = 'right',
                   left_on = 'userId',
                   right_on = 'userId')

    interactions_full_df = interactions_from_selected_users_df \
                        .groupby(['userId', 'contentId'])['eventType'].sum() \
                        .reset_index()

    "pivot trainning table => user - items based matrix"
    pivot_items_user_matrix = pd.pivot_table(interactions_full_df, index='contentId', columns='userId', values='eventType')

    "get mean of eventType"
    #pivot_items_user_matrix = pivot_items_user_matrix.apply(smooth_user_preference, axis=1)

    "replace nan with 0"
    pivot_items_user_matrix.fillna(0, inplace=True)

    "data frame that store calculated similar item relationship"
    item_sim_df = pd.DataFrame(cosine_similarity(pivot_items_user_matrix, pivot_items_user_matrix), index=pivot_items_user_matrix.index, columns=pivot_items_user_matrix.index)

    "process for vet"
    users_items_pivot_matrix_df = interactions_full_df.pivot(index='userId', 
                                                          columns='contentId', 
                                                          values='eventType').fillna(0)

    users_items_pivot_matrix_df.head(10)

    users_items_pivot_matrix = users_items_pivot_matrix_df.as_matrix()
    users_ids = list(users_items_pivot_matrix_df.index)

    #The number of factors to factor the user-item matrix.
    NUMBER_OF_FACTORS_MF = 8
    if users_items_pivot_matrix.shape[0] < 8:
        NUMBER_OF_FACTORS_MF = users_items_pivot_matrix.shape[0] - 1
    #Performs matrix factorization of the original user item matrix
    U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 

    #Converting the reconstructed matrix back to a Pandas dataframe
    cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()

    #Computes the most popular items
    item_popularity_df = interactions_full_df.groupby('contentId')['eventType'].sum().sort_values(ascending=False).reset_index()
    
    return None
"End of initial dataframe"
reTrainMatrix()

"Class popular based class"
"Popularity model"

class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], take = 10, skip = 0):
        if take is None:
            take = 10
        if skip is None:
            skip = 0
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)] \
                               .sort_values('eventType', ascending = False)
                               
        item_user_owned = articles_df.loc[articles_df['userId'] == user_id]['contentId'].values       
        item_user_rated = interactions_df.loc[interactions_df['userId'] == user_id]['contentId'].values                
        recommendations_df = recommendations_df[~recommendations_df['contentId'].isin(item_user_owned)]    
        #recommendations_df = recommendations_df[~recommendations_df['contentId'].isin(item_user_rated)]    
               
        recommendations_df = recommendations_df.drop(recommendations_df.index[from_number_to_array(skip)])
        recommendations_df = recommendations_df.head(take)
                
        return recommendations_df
    
popularity_model = PopularityRecommender(item_popularity_df, articles_df)
"End of popular based class"

"class matrix fatory filtering"
class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], take=10, skip=0):
        if take is None:
            take = 10
        if skip is None:
            skip = 0
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
                               .sort_values('recStrength', ascending = False)
        item_user_owned = articles_df.loc[articles_df['userId'] == user_id]['contentId'].values                   
        recommendations_df = recommendations_df[~recommendations_df['contentId'].isin(item_user_owned)]    
        #recommendations_df = recommendations_df[~recommendations_df['contentId'].isin(item_user_rated)]    
               
        recommendations_df = recommendations_df.drop(recommendations_df.index[from_number_to_array(skip)])
        recommendations_df = recommendations_df.head(take)
        
        return recommendations_df

"End of matrix factory class"
cf_recommender_model = CFRecommender(cf_preds_df, articles_df)

"User based - KNN class"
#user based model using KNN 
#nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(users_items_pivot_matrix_df.values)
#distances, indices = nbrs.kneighbors(users_items_pivot_matrix_df.values)
#a = nbrs.kneighbors([users_items_pivot_matrix_df[2559:]])

"End of user based KNN class"

"Item-based method"
def get_similar_item(item_name):
    if item_name not in pivot_items_user_matrix.index:
        return None, None
    else:
        sim_items = item_sim_df.sort_values(by=item_name, ascending=False).index[1:]
        sim_score = item_sim_df.sort_values(by=item_name, ascending=False).loc[:, item_name].tolist()[1:]

        return sim_items, sim_score

"get top n items and skip m items in list_new_items"
def get_new_item(take, skip):
    position = 0
    result = []
    for index, row in list_new_item[: take + skip].iterrows():
        if position >= skip:
            if (int(getattr(row, "contentId")) in articles_df['contentId'].values) is True:
                result.append({'itemId' : int(row['contentId']), 
                               'ownerId': int(row['userId'])})
        
        position = position + 1
    
    return result

get_new_item(take = 10, skip = 0)

"get top n items and skip m items"
def get_items(itemId, take, skip):
    items, score = get_similar_item(itemId)
    position = 0
    result = []
    if items is not None and score is not None and len(items) > 0 and len(score) > 0:
        for x, y in zip(items[: take + skip], score[:take + skip]):
            if position >= skip:
                if (int(x) in articles_df['contentId'].values) is True:
                    result.append({'itemId': int(x), 
                                   'similarScore': y, 
                                   'title': articles_df.loc[articles_df['contentId'] == x]['title'].values[0], 
                                   'ownerId': int(articles_df.loc[articles_df['contentId'] == x]['userId'].values[0])})
            position = position + 1

    return result
test = 2

"End of Item-based method"
def index(request):
    global test
    test = test + 1
    return HttpResponse(test)

def similarItem(request):
    skip = 0
    take = 10
    request.GET.get('s3_object_name')

    if request.GET.get('id') is not None:
        itemId = int(request.GET.get('id'))
        skip = int(request.GET.get('skip'))
        take = int(request.GET.get('take'))
    else:
        return "Error: No id field provided. Please specify an id."
    
    listSimilarItems = get_items(itemId = itemId, skip = skip, take = take)
    if (itemId in articles_df['contentId'].values) is True:
        return JsonResponse({ 'requestTitle': articles_df.loc[articles_df['contentId'] == itemId]['title'].values[0], 'similarItems' : listSimilarItems}, safe=False)
    else:
        return JsonResponse({'ErrorCode': 'ItemId is not exist in db'}, safe=False)

def reTrainModel(request):
    reTrainMatrix()
    return HttpResponse("Done")

def getRecomendedItemForUser(request):
    global cf_recommender_model
    global popularity_model
    userId = None
    take = None
    skip = None

    if request.GET.get('id') is not None:
        userId = int(request.GET.get('id'))
    if request.GET.get('take') is not None:
        take = int(request.GET.get('take'))
    if request.GET.get('skip') is not None:
        skip = int(request.GET.get('skip'))

    if (userId is None) or (countNumberOfItemsUserRated(userId, interactions_df) < 5):
        listSimilarItems = popularity_model.recommend_items(user_id = userId, take = take, skip = skip)
        result = []
        for index, row in listSimilarItems.iterrows():
            if (row["contentId"] in articles_df['contentId'].values) is True:
                result.append({'itemId': int(row["contentId"]), 'ownerId': int(articles_df.loc[articles_df['contentId'] == row["contentId"]]['userId'].values[0])})
    
        return JsonResponse({'listItems': result}, safe = False)
    else:
        newItemTake = int(take * newItemRatio)
        newItemSkip = int(skip * newItemRatio)
        itemTake = take - newItemTake
        itemSkip = skip - newItemSkip
        
        listSimilarItems = cf_recommender_model.recommend_items(user_id = userId, take = itemTake, skip = itemSkip)
        result = []
        
        listNewItem = get_new_item(take = newItemTake, skip = newItemSkip)
        result = listNewItem
        
        for index, row in listSimilarItems.iterrows():
            if (row["contentId"] in articles_df['contentId'].values) is True:
                result.append({'itemId': int(row["contentId"]), 'ownerId': int(articles_df.loc[articles_df['contentId'] == row["contentId"]]['userId'].values[0])})
            else:
                result.append({'itemId': int(row["contentId"]), 'ownerId': 0})
        return JsonResponse({'listItems': result}, safe = False)
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
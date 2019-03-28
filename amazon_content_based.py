import pandas as pd
import numpy as np
import logging
import random

logging.basicConfig(filename='log/app.log', filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO)



def origin_id_to_idx(fp,col):
    vocab=set()
    df=pd.read_csv(fp)
    for i in df[col].values:
        [vocab.add(c) for c in i.split('|')]

    index_lookup=dict()
    idx=0
    for i in vocab:
        index_lookup[i]=idx
        idx+=1
    return index_lookup




    
def get_order(lookup):
    order=[''] * len(lookup) 
    for k,v in lookup.items():
        order[v]=k
    return order


def get_product_features_mat(df, categories_to_idx,pid_order,mode='categories'):
    p_mat=[]
    for idx,row in df.set_index('asin').loc[pid_order].iterrows():
        v=np.zeros(len(categories_to_idx))
        for c in row['categories'].split('|'):
            idx=[categories_to_idx[c]]
            v[idx]+=1
        p_mat.append(v)
    return np.array(p_mat)


def get_user_matrix(df,pid_to_idx,product_mat):
    profile=dict()
    for idx,row in df.iterrows():
        profile[row['uid']]=np.zeros(len(pid_to_idx) )

    for idx,row in df.iterrows():
        v_idx=pid_to_idx[row['pid']]
        profile[row['uid']][v_idx]=profile[row['uid']][v_idx]+row['rating']

    for k,v in profile.items():
        # v: 1 * 10 product
        # product_mat: 10 product * 5 categories

        # update profile
        #print(v.shape)
        #print(product_mat.shape)
        profile[k]=v.dot(product_mat)
        #print(profile[k].shape)
        # profile: 1 * 5 categories

    return profile


def accuracy_at_n(user_profile,user_history,user_test,product_mat,n=3):
    from sklearn.metrics.pairwise import cosine_similarity
    training_ok=0
    testing_ok=0
    for k,v in user_profile.items():
        training_sims=[]
        testing_sims=[]
        # product mat is in the order of pid_order, idx in array is the p idx
        for idx, vec in enumerate(product_mat):
            # if prod idx is in the history
            if idx in user_history[k]:
                training_sims.append(cosine_similarity([user_profile[k]],[vec])[0][0])
                testing_sims.append(0) # we dont recommend history items in testing
            else:
                training_sims.append(cosine_similarity([user_profile[k]],[vec])[0][0])
                testing_sims.append(cosine_similarity([user_profile[k]],[vec])[0][0])

        top_n_training=np.array(training_sims).argsort()[-n:][::-1]
        top_n_testing=np.array(testing_sims).argsort()[-n:][::-1]

        if set(top_n_training) & user_history[k]:
            training_ok+=1
        if set(top_n_testing) & user_test[k]:
            testing_ok+=1
    return("most similar model training hit@%d: %f; testing hit@%d: %f"%(n,training_ok/len(user_profile),n,testing_ok/len(user_profile)))


def random_recommender_metric(user_profile,user_history,user_test,product_mat,n=3):
    training_ok=0
    testing_ok=0
    for k,v in user_profile.items():
        top_n_training=[]
        top_n_testing=[]

        for i in range(n):
            idx = random.randint(0,len(product_mat))
            top_n_training.append(n)
            top_n_testing.append(n)

        if set(top_n_training) & user_history[k]:
            training_ok+=1
        if set(top_n_testing) & user_test[k]:
            testing_ok+=1
    return("random model training hit@%d: %f; testing hit@%d: %f"%(n,training_ok/len(user_profile),n,testing_ok/len(user_profile)))


def popular_recommender_metric(train,pid_to_idx,user_history,user_test,n=3):
    top_n=get_top_n_popular_p_idx(train,n,pid_to_idx)
    training_ok=0
    testing_ok=0

    for k,v in user_profile.items():
        if set(top_n) & set(user_history[k]):
            training_ok+=1
        if set(top_n) & set(user_test[k]):
            testing_ok+=1

    return("popular model training hit@%d: %f; testing hit@%d: %f"%(n,training_ok/len(user_profile),n,testing_ok/len(user_profile)))



def get_user_to_prod_idx(df,pid_to_idx):
    d=dict()
    for idx,row in df.iterrows():
        d[row['uid']] = set()

    for idx,row in df.iterrows():
        d[row['uid']].add( pid_to_idx[row['pid']] )
    return d

def get_train_test(dfr,top_n_customer=3):
    most_active_customer=dfr[['uid','rating']].groupby('uid').count().sort_values(by='rating').tail(top_n_customer).index.values
    dfr=dfr[dfr.uid.isin(most_active_customer)].reset_index(drop=True)
    idx = dfr.groupby('uid').apply(lambda x: x.sample(1, random_state = 0)).index.get_level_values(1)
    test = dfr.iloc[idx, :].reset_index(drop = True)
    train = dfr.drop(idx).reset_index(drop = True)
    return train,test

def get_top_n_popular_p_idx(df,hit_at_n,pid_to_idx):
    pids=df.groupby('pid').count().sort_values(by='rating').tail(hit_at_n).index.values
    return [pid_to_idx[pid] for pid in pids]


if __name__=="__main__":

    top_n_customer=100
    hit_at_n=30
    vectorize_feature_mode='categories'

    logging.info('start param: hit@n: %d, n_customer to test: %d, vectorize mode: %s'%(hit_at_n,top_n_customer,vectorize_feature_mode))

    amazon_data='amazon_video_games/'
    categories_to_idx=origin_id_to_idx(amazon_data+'meta_small.csv','categories')
    pid_to_idx=origin_id_to_idx(amazon_data+'ratings_small.csv','pid')
    pid_order = get_order(pid_to_idx)

    """
    asin categories
    """
    df=pd.read_csv(amazon_data+'meta_small.csv')
    """
                  uid         pid  rating          ts
    0  A2RT9YYFP0ZHRG  B000MJB0H6       3  1362614400
    1  A1BS8YM712C0VJ  B00008KUA3       4  1088035200
    2  A3VU9EC95GZ418  B005UDTTS6       5  1363478400
    3  A2IH7X72AMFDHM  B001CM0PR8       4  1229731200
    4  A2AK59AGLZST3N  B0065NP39E       5  1368316800
    """
    dfr=pd.read_csv(amazon_data+'ratings_small.csv')

    train,test=get_train_test(dfr,top_n_customer)

    product_mat=get_product_features_mat(df,categories_to_idx,pid_order,mode=vectorize_feature_mode)

    user_profile=get_user_matrix(train,pid_to_idx,product_mat)
    
    # dict to record what user had rated
    user_history=get_user_to_prod_idx(train,pid_to_idx)
    # dict to record what user actually intereseted
    user_test=get_user_to_prod_idx(test,pid_to_idx)



    res=accuracy_at_n(user_profile,user_history,user_test,product_mat, n=hit_at_n)
    logging.info(res)

    res2=random_recommender_metric(user_profile,user_history,user_test,product_mat, n=hit_at_n)
    logging.info(res2)

    res3=popular_recommender_metric(dfr,pid_to_idx,user_history,user_test, n=hit_at_n)
    logging.info(res3)

    logging.info("done")

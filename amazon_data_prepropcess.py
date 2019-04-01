import pandas as pd
import matplotlib.pyplot as plt
import re
import ast 

import os

amazon_data='amazon_video_games/'
meta_file='meta_Video_Games.json'
ratings_file='ratings_Video_Games.csv'
plot_dir='plot/'


def filter_k_core(df):
    count_ratings={}
    K_CORE=10
    df2=df[['uid','rating']].groupby('uid').count()
    print(df2[df2.rating>K_CORE])


def plot_product_id_per_sales(df):
    df2=df[['pid','rating']].groupby('pid').count().sort_values(by='rating', ascending=False)
    #print(df2)
    # sample to plot, other wise not visible result
    df2.rating.sample(100).sort_values(ascending=False).plot.bar()
    plt.savefig(plot_dir+'product_sales_bar.png')


def plot_user_id_per_sales(df):
    df2=df[['uid','rating']].groupby('uid').count().sort_values(by='rating', ascending=False)
    # sample to plot, other wise not visible result
    df2.rating.sample(100).sort_values(ascending=False).plot.bar()
    plt.savefig(plot_dir+'user_sales_bar.png')


def try_preprocess(source_fp, dest_fp, processor, force_run=False):
    if os.path.isfile(dest_fp) and not force_run:
        print("from cache: " + dest_fp)
    else:
        print("from scratch: " + dest_fp)
        processor(source_fp, dest_fp)


def first_100_row(source, dest):
    pd.read_csv(source)[:100].to_csv(dest,index=None)


def origin_reviews_rating_processor(s,d):
    (pd.read_json(s,lines=True)
    [['reviewerID','asin','overall','unixReviewTime']]
    .rename(columns={"reviewerID": "uid", "asin": "pid", "overall":"rating","unixReviewTime":"ts"})
    .to_csv(d,index=None))


def slice_ratings(s,d):
    #pd.read_csv(s).sample(10000).to_csv(d,index=None)
    pd.read_csv(s).to_csv(d,index=None)


def flatten_categories(x):
    y=set()
    x=ast.literal_eval(x)
    for inner_list in x:
        for item in inner_list:
            y.add(item)
    return '|'.join(y)


def meta_set_from_ratings(s,d):
    ratings=pd.read_csv(s)
    mentioned_pid=set(ratings.pid.values)

    meta=pd.read_csv(amazon_data+'meta.csv').dropna(subset=['categories'])
    meta=meta[meta.asin.isin(mentioned_pid)]
    meta['categories']=meta.categories.apply(flatten_categories)
    meta.to_csv(d,index=None)


def origin_meta_processor(s,dest):
    d=[]
    with open(s,'r') as f:
        for line in f:
            line = re.sub(r'[^\x00-\x7F]+',' ', line)
            parsed=ast.literal_eval(line)
            d.append(parsed)

    df=pd.DataFrame(d)[['asin','title','categories']]
    df.to_csv(dest,index=None)


def rename_ratings_header_movie(s,d): 
    pd.read_csv(s).rename(columns= {'uid': 'userId', 'pid': 'movieId', 'ts': 'timestamp' }).to_csv(d,index=None)


def rename_meta_header_movie(s,d):
    pd.read_csv(s).rename(columns= {'asin': 'movieid', 'categories': 'genres'}).to_csv(d,index=None)


def top_download(df):
    df2=df[['pid','rating']].groupby('pid').count().sort_values(by='rating', ascending=False).iloc[:10]
    df3=pd.read_csv(amazon_data+'meta.csv')
    df2=df2.merge(df3,how='left',left_on='pid',right_on='asin')[['rating','pid','categories']]
    print(df2)


def unique_users(df):
    print(len(df.uid.unique()))
    print(len(df.pid.unique()))



if __name__=="__main__":
    # just a test
    try_preprocess(amazon_data+ratings_file,amazon_data+'toy_ratings.csv',first_100_row)

    # parse json
    try_preprocess(amazon_data+'reviews_Video_Games_5.json', amazon_data+'ratings.csv',origin_reviews_rating_processor)
    try_preprocess(amazon_data+'meta_Video_Games.json',amazon_data+'meta.csv',origin_meta_processor)


    # gen small data
    try_preprocess(amazon_data+'ratings.csv',amazon_data+'ratings_small.csv',slice_ratings,True)
    try_preprocess(amazon_data+'ratings_small.csv',amazon_data+'meta_small.csv',meta_set_from_ratings,True)

    # gen non popular 

    # try_preprocess(amazon_data+'ratings_small.csv',amazon_data+'ratings_small_movie.csv',rename_ratings_header_movie)
    # try_preprocess(amazon_data+'meta_small.csv',amazon_data+'meta_small_movie.csv',rename_meta_header_movie)


    # data exploration long tail
    #df=pd.read_csv(amazon_data+'ratings.csv',names=['uid','pid','rating'])
    df=pd.read_csv(amazon_data+'ratings.csv')

    # count
    #unique_users(df)

    #print(df)
    #plot_product_id_per_sales(df)
    #plot_user_id_per_sales(df)
    #top_download(df)

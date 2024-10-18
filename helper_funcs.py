import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

PRODUCT_HUNT_API_KEY = os.getenv("PRODUCT_HUNT_API_KEY")
PRODUCT_HUNT_API_SECRET = os.getenv('PRODUCT_HUNT_API_SECRET')
redirect_uri = os.getenv('redirect_uri')

token_url = 'https://api.producthunt.com/v2/oauth/token'
main_url = 'https://api.producthunt.com/v2/api/graphql'

data = {
  "client_id": PRODUCT_HUNT_API_KEY,
  "client_secret": PRODUCT_HUNT_API_SECRET,
  "redirect_uri": redirect_uri,
  "grant_type": "client_credentials"
}

def get_tokens():
    token_data = requests.post('https://api.producthunt.com/v2/oauth/token',data=data).json()
    token = token_data['access_token']
    token_type = token_data['token_type']
    return token,token_type

def get_headers(token,token_type):
    headers = {
        'Authorization': f'{token_type} {token}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
        }
    return headers

def fetch_top_posts(start_date, end_date,headers,limit=100):
    all_posts = []
    has_next_page = True
    after_cursor = None
    iteration_count = 0
    max_iterations = limit // 20      
    while has_next_page and iteration_count < max_iterations:
        query1 ="""
        {
            posts(first: 20, order: VOTES, postedAfter: "%s" , postedBefore: "%s" %s) {
                pageInfo {
                    endCursor
                    hasNextPage
                }
                edges {
                    node {
                        name
                        description
                        url
                        votesCount
                        createdAt
                        tagline
                        commentsCount
                        comments{
                          nodes{
                            body
                            createdAt
                            votesCount
                          }
                        }
                        topics {
                          nodes {
                            slug
                          }
                        }
                    }
                }
            }
        }
        """ % (start_date,end_date,f', after: "{after_cursor}"' if after_cursor else '')

        response = requests.post(main_url, json={'query': query1}, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # print(data)
            posts = data['data']['posts']['edges']
            all_posts.extend(posts)
            has_next_page = data['data']['posts']['pageInfo']['hasNextPage']
            after_cursor = data['data']['posts']['pageInfo']['endCursor']
            iteration_count += 1
        else:
            print(f"Failed to fetch data: {response.status_code}")
            break

    return all_posts

def fetch_last_posts(start_date, end_date,headers,limit=100):
    all_posts = []
    has_previous_page = True
    before_cursor = None
    post_count = 0
    limit = limit     
    while has_previous_page and post_count < limit:
        query1 = """
        {
          posts(last: 20, order: VOTES, postedAfter: "%s", postedBefore: "%s" %s) {
            pageInfo {
              startCursor
              hasPreviousPage
            }
            edges {
              node {
                name
                description
                url
                votesCount
                createdAt
                tagline
                commentsCount
                comments{
                  nodes{
                    body
                    createdAt
                    votesCount
                  }
                }

                topics {
                  nodes {
                    slug
                  }
                }
              }
            }
          }
        }
        """ % (start_date, end_date, f', before: "{before_cursor}"' if before_cursor else '')

        response = requests.post(main_url, json={'query': query1}, headers=headers)
        if response.status_code == 200:
            data = response.json()
            posts = data['data']['posts']['edges']
            all_posts.extend(posts)  
            post_count += len(posts)

            has_previous_page = data['data']['posts']['pageInfo']['hasPreviousPage']
            before_cursor = data['data']['posts']['pageInfo']['startCursor']
        else:
            print(f"Failed to fetch data: {response.status_code}")
            break

    return all_posts[:limit]  

def fetch_posts(start_date, end_date,headers,limit=60):
    all_posts = []
    has_next_page = True
    after_cursor = None
    iteration_count = 0
    max_iterations = limit // 20      
    while has_next_page and iteration_count < max_iterations:
        query1 ="""
        {
            posts(first: 60, postedAfter: "%s" , postedBefore: "%s" %s) {
                pageInfo {
                    endCursor
                    hasNextPage
                }
                edges {
                    node {
                        name
                        description
                        url
                        votesCount
                        createdAt
                        tagline
                        commentsCount
                        comments{
                          nodes{
                            body
                            createdAt
                            votesCount
                          }
                        }
                        topics {
                          nodes {
                            slug
                          }
                        }
                    }
                }
            }
        }
        """ % (start_date,end_date,f', after: "{after_cursor}"' if after_cursor else '')

        response = requests.post(main_url, json={'query': query1}, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(data)
            posts = data['data']['posts']['edges']
            all_posts.extend(posts)
            has_next_page = data['data']['posts']['pageInfo']['hasNextPage']
            after_cursor = data['data']['posts']['pageInfo']['endCursor']
            
        else:
            print(response.json())
            print(f"Failed to fetch data: {response.status_code}")
            break
        iteration_count += 1
    return all_posts

def build_bigram_dataset(data,stoi):
    xs = []
    ys = []

    for i in data:
        w = []
        w = ['<start>'] + list(i) + ['<end>']
        for i in range(len(w)-1):
            xs.append(stoi[w[i]])
            ys.append(stoi[w[i+1]])
    return xs,ys

def build_ngram_dataset(data,stoi,context_window = 3):
    xs = []
    ys = []
    for w in data:
        w = list(w) + ['<end>']
        context = ['<start>'] * context_window
        for i in range(len(w)):
            xs.append([stoi[token] for token in context])
            ys.append(stoi[w[i]])
            context = context[1:] + [w[i]]
    return xs,ys
import streamlit as st
import praw
import json
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer, util


# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
custom_stopwords = set(stopwords.words('english'))
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in custom_stopwords])
    return text

def fetch_posts_with_pagination(subreddit, search_queries, limit=5, page_size=10):
    reddit = praw.Reddit(client_id='sWUlJdUFimheFUch2lsGng',
                         client_secret='b7c0MbDX2xQuRdSWhM-FyJu4NMtejQ',
                         user_agent='ChangeMeClient/0.1 by YourUsername')
    posts_data = []
    
    for search_query in search_queries:
        post_count = 0
        after = None
        while post_count < limit:
            posts = reddit.subreddit(subreddit).search(search_query, sort='new', limit=page_size, params={'after': after})
            for post in posts:
                post_data = {
                    'title': post.title,
                    'text': post.selftext,
                    'comments': []
                }
                post.comments.replace_more(limit=None)
                for comment in post.comments.list():
                    if comment.body:
                        post_data['comments'].append({'body': comment.body, 'author': str(comment.author) if comment.author else 'Anonymous'})
                posts_data.append(post_data)
                post_count += 1
            after = posts._listing[-1].fullname if len(posts._listing) > 0 else None
            if not after:
                break
    return posts_data

st.set_page_config(page_title="Reddit Data Generator", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Creation", "Proposed Topics", "Sentiment Analysis"])

if page == "Dataset Creation":
    st.title("Reddit Dataset Generator")
    subreddit = st.text_input("Subreddit Name", "worldnews")
    search_queries_input = st.text_area("Enter search queries (one per line)")
    search_queries = [q.strip() for q in search_queries_input.split("\n") if q.strip()]
    filename = st.text_input("Enter filename to save JSON", "reddit_data.json")
    
    if st.button("Fetch Data"):
        if search_queries:
            posts = fetch_posts_with_pagination(subreddit, search_queries)
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(posts, f, ensure_ascii=False, indent=4)
            st.success(f"Data saved to {filename}")
        else:
            st.warning("Please enter at least one search query.")

elif page == "Proposed Topics":
    st.title("Topic Modeling with LDA")
    uploaded_file = st.file_uploader("Upload Reddit JSON File", type=["json"])
    
    if uploaded_file is not None:
        data = json.load(uploaded_file)
        texts = [preprocess_text(post["text"]) for post in data if post["text"]]
        for post in data:
            for comment in post["comments"]:
                if "body" in comment and comment["body"]:
                    texts.append(preprocess_text(comment["body"]))
        
        vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
        text_matrix = vectorizer.fit_transform(texts)
        
        lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
        lda_model.fit(text_matrix)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            topics.append(f"Topic {topic_idx+1}: {', '.join(top_words)}")
        
        st.subheader("Top 5 Topics:")
        for topic in topics:
            st.write(topic)

elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis and Topic Assignment")
    uploaded_file = st.file_uploader("Upload Reddit JSON File", type=["json"])
    topics_input = st.text_area("Enter topics and their most important words (one per line)")
    
    if uploaded_file is not None and topics_input:
        data = json.load(uploaded_file)
        topics = [line.split(":")[0].strip() for line in topics_input.split("\n") if ":" in line]
        
        post_texts = []
        for post in data:
            combined_text = f"{post['title']} {post['text']}"
            if post["comments"]:
                combined_text += f" {post['comments'][0]['body']}"
            post_texts.append(preprocess_text(combined_text))
        
        tweet_embeddings = model.encode(post_texts)
        topic_embeddings = model.encode(topics)
        
        similarity_scores = util.pytorch_cos_sim(tweet_embeddings, topic_embeddings)
        
        assigned_topics = []
        for i, post in enumerate(data):
            best_topic_index = similarity_scores[i].argmax().item()
            best_topic = topics[best_topic_index]
            assigned_topics.append([post["title"], post["text"], best_topic])
        
        df = pd.DataFrame(assigned_topics, columns=["Title", "Post Text", "Assigned Topic"])
        st.subheader("Post Topics")
        st.dataframe(df)

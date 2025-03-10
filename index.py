import streamlit as st
import praw
import json
import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer, util
import altair as alt
import os



# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
custom_stopwords = set(stopwords.words('english'))
model = SentenceTransformer('all-MiniLM-L6-v2')
# Initialize session state for topic_emotions_df
if "topic_emotions_df" not in st.session_state:
    st.session_state.topic_emotions_df = pd.DataFrame()

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in custom_stopwords])
    return text

def fetch_posts_with_pagination(subreddit, search_queries, limit=500, page_size=10):
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
page = st.sidebar.radio("Go to", ["Dataset Creation", "Proposed Topics", "Sentiment Analysis", "Visualization"])

if page == "Dataset Creation":
    st.title("Reddit Dataset Generator")
    subreddit = st.text_input("Subreddit Name", "worldnews")
    search_queries_input = st.text_area("Enter search queries (one per line)")
    search_queries = [q.strip() for q in search_queries_input.split("\n") if q.strip()]
    filename = "data\\" + st.text_input("Enter filename to save JSON", "reddit_data.json")
    
    if st.button("Fetch Data"):
        if search_queries:
            posts = fetch_posts_with_pagination(subreddit, search_queries)
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(posts, f, ensure_ascii=False, indent=4)
            st.success(f"Data saved to {filename}")
        else:
            st.warning("Please enter at least one search query.")

elif page == "Proposed Topics":
    import pyLDAvis
    import pyLDAvis.lda_model

    st.title("Topic Modeling with LDA")
    uploaded_file = st.file_uploader("Upload Reddit JSON File", type=["json"])

    # User input for number of topics
    num_topics = st.number_input("Enter the number of topics:", min_value=2, max_value=20, value=5, step=1)

    if uploaded_file is not None:
        data = json.load(uploaded_file)
        texts = [preprocess_text(post["text"]) for post in data if post["text"]]
        for post in data:
            for comment in post["comments"]:
                if "body" in comment and comment["body"]:
                    texts.append(preprocess_text(comment["body"]))

        vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
        text_matrix = vectorizer.fit_transform(texts)

        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(text_matrix)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            topics.append(f"Topic {topic_idx+1}: {', '.join(top_words)}")

        st.subheader(f"Top {num_topics} Topics:")
        for topic in topics:
            st.write(topic)

        # LDA Visualization
        st.subheader("LDA Topic Visualization")
        with st.spinner("Generating visualization..."):
            panel = pyLDAvis.lda_model.prepare(lda_model, text_matrix, vectorizer)
            html_string = pyLDAvis.prepared_data_to_html(panel)
            st.components.v1.html(html_string, width=1700, height=1200)



elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis and Topic Assignment")
    uploaded_file = st.file_uploader("Upload Reddit JSON File", type=["json"])
    topics_input = st.text_area("Enter topics and their most important words (one per line)")
    emotion = "Good (Positive): trust respectful strictly \n Fear (Negative): outbreak death spreading \n Happy (Positive): joy hope celebration \n Disgust (Negative): lies hypocrisy pollution \n Sadness (Negative): depressed loss suffering \n Surprise (Neutral): shock unexpected"

    
    if uploaded_file is not None and topics_input:
        data = json.load(uploaded_file)
        topics = [line.split(":")[0].strip() for line in topics_input.split("\n") if ":" in line]
        emotions =  [line.split(":")[0].strip() for line in emotion.split("\n") if ":" in line]

        post_texts = []
        for post in data:
            combined_text = f"{post['title']} {post['text']}"
            if post["comments"]:
                combined_text += f" {post['comments'][0]['body']}"
            post_texts.append(preprocess_text(combined_text))

        tweet_embeddings = model.encode(post_texts)
        topic_embeddings = model.encode(topics)
        emotion_embeddings = model.encode(emotions) 

        similarity_scores = util.pytorch_cos_sim(tweet_embeddings, topic_embeddings)
        similarity_scores_emotions = util.pytorch_cos_sim(tweet_embeddings, emotion_embeddings )
        
        assigned_topics = []
        for i, post in enumerate(data):
            best_topic_index = similarity_scores[i].argmax().item()
            best_emotion_index = similarity_scores_emotions[i].argmax().item()
            best_emotion = emotions[best_emotion_index]
            best_topic = topics[best_topic_index]
            assigned_topics.append([post["title"] if post["title"] and post["title"].strip() else post["text"], post["text"], best_topic, best_emotion])


        df = pd.DataFrame(assigned_topics, columns=["Title","Post", "Assigned Topic", "Assigned Emotion"])
        st.session_state.topic_emotions_df = df  
        # Add a "Select" column for row selection
        df.insert(0, "Select", False)

        # Pagination setup (Show 10 rows per page)
        rows_per_page = 10
        total_pages = (len(df) // rows_per_page) + (1 if len(df) % rows_per_page else 0)

        # User selects the page
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

        # Get the subset of the DataFrame for the current page
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        df_paginated = df.iloc[start_idx:end_idx]
        st.subheader("Post Topics")

        # Show the table with selection inside
        edited_df = st.data_editor(
            df_paginated[["Select", "Title", "Assigned Topic", "Assigned Emotion"]],  # Only show relevant columns
            hide_index=True, 
            column_config={
                "Select": st.column_config.CheckboxColumn("Select a row"),
                "Title": st.column_config.TextColumn("Title"),
            }
        )

        # Filter selected row(s)
        selected_rows = df[df["Title"].isin(edited_df[edited_df["Select"]]["Title"])]

        # Show details below if any row is selected
        if not selected_rows.empty:
            st.markdown("---")  # Separator line
            for _, row in selected_rows.iterrows():
                st.write(f"### {row['Title']}")
                st.write(f"**Post Content:** {row['Post']}")
                st.write(f"**Assigned Topic:** {row['Assigned Topic']}")
                st.write(f"**Assigned Emotion:** {row['Assigned Emotion']}")
                st.markdown("---")  # Separator for multiple selections

elif page == "Visualization":
    if not st.session_state.topic_emotions_df.empty:  # ✅ Check stored DataFrame
        st.subheader("Visualization of Topics and Emotions")
        output_json_path = "data/dfTweets.json"
        df_json = st.session_state.topic_emotions_df.to_json(orient="records", force_ascii=False, indent=4)

        # Write JSON to file
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.write(df_json)
        import seaborn as sns
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        import pandas as pd
        from matplotlib.colors import to_hex  

        # 🎨 Define Custom Colors for Emotions
        emotion_color_map = {
            "Good (Positive)": "#2ECC71",      # Green
            "Fear (Negative)": "#E67E22",      # Orange
            "Happy (Positive)": "#58D68D",     # Light Green
            "Disgust (Negative)": "#C0392B",   # Red
            "Sadness (Negative)": "#5DADE2",   # Blue
            "Surprise (Neutral)": "#F4D03F"    # Yellow
        }

        # 🎯 1️⃣ Distribution of Topics (Bar Chart)
        topic_counts = st.session_state.topic_emotions_df["Assigned Topic"].value_counts().reset_index()
        topic_counts.columns = ["Topic", "Count"]
        num_topics = len(topic_counts)
        color_palette = sns.color_palette("husl", num_topics)  # Generates a diverse color scheme
        topic_counts["Color"] = [to_hex(color_palette[i]) for i in range(num_topics)]


        bar_chart = (
            alt.Chart(topic_counts)
            .mark_bar()
            .encode(
                x=alt.X("Topic:N", title="Topics"),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color("Color:N", scale=alt.Scale(domain=topic_counts["Color"].tolist(), range=topic_counts["Color"].tolist()), legend=None),
                tooltip=["Topic", "Count"]
            )
        ).properties(title="Distribution of Topics", width=350, height=300)

        # 🎭 2️⃣ Overall Emotion Distribution (Pie Chart)
        overall_emotion_counts = (
            st.session_state.topic_emotions_df["Assigned Emotion"]
            .value_counts()
            .reset_index()
        )
        overall_emotion_counts.columns = ["Emotion", "Count"]

        pie_chart = (
            alt.Chart(overall_emotion_counts)
            .mark_arc()
            .encode(
                theta=alt.Theta("Count:Q", stack=True),
                color=alt.Color("Emotion:N", scale=alt.Scale(domain=list(emotion_color_map.keys()), range=list(emotion_color_map.values()))),
                tooltip=["Emotion", "Count"]
            )
        ).properties(title="Overall Emotion Distribution", width=350, height=300)

        # ✅ Display Bar Chart & Pie Chart Side by Side
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(bar_chart, use_container_width=True)
        with col2:
            st.altair_chart(pie_chart, use_container_width=True)

        # 🎯 3️⃣ Small Radar Chart for Emotion-Topic Distribution
        st.subheader("Radar Chart & Word Cloud")

        # Prepare Data for Radar Chart
        radar_data = (
            st.session_state.topic_emotions_df
            .groupby(["Assigned Topic", "Assigned Emotion"])
            .size()
            .unstack(fill_value=0)
        )

        categories = list(radar_data.columns)
        topics = list(radar_data.index)
        num_vars = len(categories)

       # Compute angles for radar chart
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        # Create Smaller Radar Chart
        fig_radar, ax_radar = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        topic_color_map = dict(zip(topic_counts["Topic"], topic_counts["Color"]))


        topic_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for topic in topics:
            values = radar_data.loc[topic].values.tolist()
            values += values[:1]  # Close the shape
            color = topic_color_map.get(topic, "#333333")  # Default to gray if no color found

            ax_radar.plot(angles, values, color=color, linewidth=1.5, linestyle='solid', label=topic)
            ax_radar.fill(angles, values, color=color, alpha=0.2)

        # Adjust the font size of x-axis labels (categories)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories, fontsize=7, fontweight='bold', color="black", ha='center')

        # Remove y-tick labels for clarity
        ax_radar.set_yticklabels([])

        # Adjust title
        ax_radar.set_title("Emotion by Topic", fontsize=10, fontweight='bold', pad=10)

        # Move legend BELOW the chart, reduce font size, and arrange in 2 columns
        ax_radar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7, labelspacing=0.3, ncol=2, frameon=False)

        plt.tight_layout()


        # 🎯 4️⃣ Topic Word Cloud
        topic_text = " ".join(st.session_state.topic_emotions_df["Assigned Topic"])

        fig_wc, ax_wc = plt.subplots(figsize=(4, 4), dpi=150)
        wordcloud_topic = WordCloud(
            background_color="white",
            colormap="plasma",
            width=600, height=300,
            max_words=100,
            prefer_horizontal=1.0
        ).generate(topic_text)

        ax_wc.imshow(wordcloud_topic, interpolation="bilinear")
        ax_wc.axis("off")
        ax_wc.set_title("Most Frequent Words in Topics", fontsize=10)

        # ✅ Display Radar Chart and Word Cloud Side by Side
        col3, col4 = st.columns(2)
        with col3:
            st.pyplot(fig_radar)
        with col4:
            st.pyplot(fig_wc)
        # 🎯 4️⃣ Heatmap for Topic-Emotion Relationships (Smaller)
        st.subheader("Heatmap of Emotion-Topic Correlation")

        fig, ax = plt.subplots(figsize=(5, 3))  # Smaller heatmap
        emotion_counts = (
            st.session_state.topic_emotions_df
            .groupby("Assigned Topic")["Assigned Emotion"]
            .value_counts()
            .reset_index(name="Count")
        )
        sns.heatmap(
            emotion_counts.pivot(index="Assigned Topic", columns="Assigned Emotion", values="Count").fillna(0),
            cmap="coolwarm",
            annot=True,
            fmt=".0f",
            linewidths=0.5,
            ax=ax
        )
        ax.set_title("Emotion-Topic Heatmap")
        st.pyplot(fig)

    else:
        output_json_path = "data/dfTweets.json"
        if os.path.exists(output_json_path):
            # Load the JSON file into a DataFrame
            with open(output_json_path, "r", encoding="utf-8") as f:
                df_json = json.load(f)
            st.session_state.topic_emotions_df = pd.DataFrame(df_json)
            st.success("Loaded data from saved JSON file.")
        else:
            st.warning("⚠ Please perform sentiment analysis first before viewing visualizations!")

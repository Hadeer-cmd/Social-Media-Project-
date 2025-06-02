import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("Social Media Engagement Cleaned Dataset.csv", parse_dates=["timestamp"])
    df['total_engagement'] = df['likes_count'] + df['shares_count'] + df['comments_count']
    df['text_length'] = df['text_content'].apply(lambda x: len(str(x).split()))
    df['num_hashtags'] = df['hashtags'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) and x != '' else 0)
    return df

df = load_data()

# ---------------- MAIN TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(["📂 Data & Filters", "📈 Univariate", "📊 Bivariate", "🔀 Multivariate"])

# ---------------- TAB 1: DATA OVERVIEW ----------------
with tab1:
    st.title("📂 Data Overview & Filters")

    st.write("Simulated social media engagement dataset with detailed user, platform, and sentiment insights.")

    with st.expander("🧾 Key Fields"):
        st.markdown("""
        - **post_id, timestamp, platform**
        - **user_id, location, language**
        - **text_content, hashtags, mentions, keywords**
        - **sentiment_label, emotion_type, topic_category**
        - **likes_count, shares_count, comments_count, impressions**
        - **engagement_rate, buzz_change_rate, etc.**
        """)

    # Filters in sidebar
    st.sidebar.header("🔎 Filter Data")
    date_range = st.sidebar.date_input("Date Range", [df["timestamp"].min(), df["timestamp"].max()])
    location = st.sidebar.multiselect("Location", df["location"].dropna().unique())
    language = st.sidebar.multiselect("Language", df["language"].dropna().unique())

    filtered_df = df[
        (df["timestamp"].dt.date >= date_range[0]) & (df["timestamp"].dt.date <= date_range[1])
    ]
    if location:
        filtered_df = filtered_df[filtered_df["location"].isin(location)]
    if language:
        filtered_df = filtered_df[filtered_df["language"].isin(language)]

    st.dataframe(filtered_df.head(50))

# ---------------- TAB 2: UNIVARIATE ----------------
with tab2:
    st.title("📈 Univariate Analysis")

    st.subheader("1. What is the distribution of posts across different platforms?")
    st.plotly_chart(px.pie(filtered_df, names='platform', title='Posts per Platform'), use_container_width=True)

    st.subheader("2. How are sentiments distributed in the posts?")
    sentiment_counts = filtered_df['sentiment_label'].value_counts()
    fig, ax = plt.subplots(figsize=(5,5))
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader("3. What is the distribution of shares among posts?")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(filtered_df['shares_count'], bins=30, color='skyblue', edgecolor='black')
    avg = filtered_df['shares_count'].mean()
    ax.axvline(avg, color='red', linestyle='dashed', label=f'Avg: {avg:.2f}')
    ax.legend()
    ax.set_title('Distribution of Shares per Post')
    ax.set_xlabel('Number of Shares')
    ax.set_ylabel('Number of Posts')
    st.pyplot(fig)

    st.subheader("4. How does average engagement vary by day of the week?")
    avg_eng = filtered_df.groupby('day_of_week')['total_engagement'].mean()
    fig, ax = plt.subplots(figsize=(5,5))
    ax.pie(avg_eng, labels=avg_eng.index, autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader("5. What are the distributions and engagement patterns across topic categories?")
    topic_counts = filtered_df['topic_category'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(5,5))
    ax1.pie(topic_counts, labels=topic_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.boxplot(data=filtered_df, x='topic_category', y='total_engagement', ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# ---------------- TAB 3: BIVARIATE ----------------
with tab3:
    st.title("📊 Bivariate Analysis")

    st.subheader("1. How does average engagement vary by platform?")
    avg_platform = filtered_df.groupby('platform')['total_engagement'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=avg_platform, x='platform', y='total_engagement', palette='viridis', ax=ax)
    st.pyplot(fig)

    st.subheader("2. How do sentiment and emotion types vary by day of the week?")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    sentiment_counts = filtered_df.groupby(['day_of_week', 'sentiment_label']).size().unstack().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    sentiment_counts.plot(kind='bar', stacked=True, ax=axs[0])
    axs[0].set_title('Sentiment by Day')
    axs[0].set_xlabel('Day of Week')
    axs[0].set_ylabel('Number of Posts')

    emotion_counts = filtered_df.groupby(['day_of_week', 'emotion_type']).size().unstack().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    emotion_counts.plot(kind='bar', stacked=True, ax=axs[1])
    axs[1].set_title('Emotion by Day')
    axs[1].set_xlabel('Day of Week')
    axs[1].set_ylabel('Number of Posts')

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("3. What is the relationship between sentiment and average engagement?")
    avg_sent = filtered_df.groupby('sentiment_label')['total_engagement'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5,4))
    sns.barplot(data=avg_sent, x='sentiment_label', y='total_engagement', palette='pastel', ax=ax)
    st.pyplot(fig)

    st.subheader("4. How does post length relate to engagement?")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.regplot(data=filtered_df, x='text_length', y='total_engagement', scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax)
    st.pyplot(fig)

    st.subheader("5. What is the relationship between the number of hashtags and engagement rate?")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.regplot(data=filtered_df, x='num_hashtags', y='engagement_rate', scatter_kws={'alpha':0.3}, line_kws={'color':'blue'}, ax=ax)
    st.pyplot(fig)

# ---------------- TAB 4: MULTIVARIATE ----------------
with tab4:
    st.title("🔀 Multivariate Analysis")

    st.subheader("1. How do platform and sentiment together affect average engagement?")
    grouped = filtered_df.groupby(['platform', 'sentiment_label'])['total_engagement'].mean().unstack()
    st.bar_chart(grouped)

    st.subheader("2. How does emotion type and day of week interact in terms of engagement?")
    heatmap_data = filtered_df.pivot_table(values='total_engagement', index='emotion_type', columns='day_of_week', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    st.subheader("3. How do topic categories and platforms relate in terms of engagement?")
    cross = filtered_df.pivot_table(values='total_engagement', index='topic_category', columns='platform', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(cross, annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("4. How does campaign phase combined with platform impact engagement rate?")
    pivot = filtered_df.pivot_table(values='engagement_rate', index='campaign_phase', columns='platform', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(pivot, annot=True, cmap='Blues', fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("5. What is the relationship between buzz change rate, past sentiment average, and emotion type?")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=filtered_df, x='user_past_sentiment_avg', y='buzz_change_rate', hue='emotion_type', alpha=0.6, ax=ax)
    st.pyplot(fig)

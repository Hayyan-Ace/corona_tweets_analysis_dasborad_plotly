import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64  # Import base64 module for encoding image data

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("C:/Users\Muhammad Hayyan\Downloads\archive\Corona_NLP_train.csv", encoding='latin')

# Preprocess the data
stop_words = set(stopwords.words('english'))
df['cleaned_tweet'] = df['OriginalTweet'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

# Create the Dash app
app = dash.Dash(_name_)

# Define the app layout
app.layout = html.Div([
    html.H1('COVID-19 Tweet Analysis'),
    dcc.Tabs([
        dcc.Tab(label='Sentiment Analysis', children=[
            dcc.Graph(id='sentiment-chart')
        ]),
        dcc.Tab(label='Word Cloud', children=[
            html.Img(id='word-cloud-image')
        ]),
        dcc.Tab(label='Choropleth Map', children=[
            dcc.Graph(id='choropleth-map')
        ]),
        dcc.Tab(label='Sentiment Pie Chart', children=[
            dcc.Graph(id='sentiment-pie-chart')
        ])
    ])
])

# Define the callback functions
@app.callback(
    Output('sentiment-chart', 'figure'),
    [Input('sentiment-chart', 'hoverData')])
def update_sentiment_chart(hoverData):
    # Utilize the function to generate sentiment line chart
    fig = generate_sentiment_line_chart()
    return fig

@app.callback(
    Output('word-cloud-image', 'src'),
    [Input('word-cloud-image', 'id')])
def update_word_cloud(id):
    text = ' '.join(df['cleaned_tweet'])
    wordcloud = WordCloud().generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('word_cloud.png', bbox_inches='tight')
    with open('word_cloud.png', 'rb') as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode()
    return 'data:image/png;base64,{}'.format(encoded_image)

@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('choropleth-map', 'id')])
def update_choropleth_map(id):
    # Utilize the function to generate choropleth map
    fig = generate_sentiment_choropleth()
    return fig

@app.callback(
    Output('sentiment-pie-chart', 'figure'),
    [Input('sentiment-pie-chart', 'id')])
def update_sentiment_pie_chart(id):
    # Utilize the function to generate sentiment pie chart
    fig = generate_sentiment_pie_chart()
    return fig

# Define the sentiment line chart function
def generate_sentiment_line_chart():
    # Read the CSV file into a DataFrame
    df = pd.read_csv("Corona_NLP_train.csv", encoding='latin1')

    # Convert the 'TweetAt' column to datetime
    df['TweetAt'] = pd.to_datetime(df['TweetAt'], format="%d-%m-%Y")

    # Group the data by date and sentiment, and count the occurrences
    sentiment_counts = df.groupby(['TweetAt', 'Sentiment']).size().reset_index(name='Count')

    # Combine negative and extremely negative sentiments
    sentiment_counts.loc[sentiment_counts['Sentiment'].isin(['Negative', 'Extremely Negative']), 'Sentiment'] = 'Negative'

    # Combine positive and extremely positive sentiments
    sentiment_counts.loc[sentiment_counts['Sentiment'].isin(['Positive', 'Extremely Positive']), 'Sentiment'] = 'Positive'

    # Create a line chart using Plotly
    fig = px.line(sentiment_counts, x='TweetAt', y='Count', color='Sentiment',
                  labels={'Count': 'Number of Tweets', 'TweetAt': 'Date'},
                  title='Number of Tweets Over Time by Sentiment')

    # Customize the chart layout
    fig.update_layout(
        legend_title='Sentiment',
        xaxis_tickformat='%b %Y'  # Format the x-axis tick labels as month and year
    )

    return fig

# Define the choropleth map function
def generate_sentiment_choropleth():
    # Load the dataset
    df = pd.read_csv("Corona_NLP_train.csv", encoding='latin1')

    # Define color mapping
    color_map = {'Positive': 'lightgreen', 'Extremely Positive': 'darkgreen',
                 'Negative': 'orange', 'Extremely Negative': 'red', 'Neutral': 'grey'}

    # Create a choropleth map using Plotly Express
    fig = px.choropleth(df,
                         locations="Location",
                         locationmode="country names",
                         color="Sentiment",
                         color_discrete_map=color_map,
                         projection="natural earth",
                         title="Sentiment Choropleth Map",
                         hover_name="Location",
                         labels={"Sentiment": "Sentiment"}
                        )

    return fig

# Define the sentiment pie chart function
def generate_sentiment_pie_chart():
    # Read the CSV file into a DataFrame
    df = pd.read_csv("Corona_NLP_train.csv", encoding='latin1')

    # Filter out rows with missing location data
    df = df.dropna(subset=['Location'])

    # Group the data by sentiment and count the occurrences
    sentiment_counts = df['Sentiment'].value_counts().reset_index()

    # Rename the columns
    sentiment_counts.columns = ['Sentiment', 'Count']

    # Define a color mapping for each sentiment category
    color_mapping = {
        'Positive': 'lightgreen',
        'Extremely Positive': 'darkgreen',
        'Negative': 'orange',
        'Extremely Negative': 'darkred',
        'Neutral': 'lightgrey'
    }

    # Map each sentiment category to a color
    sentiment_counts['Color'] = sentiment_counts['Sentiment'].map(color_mapping)

    # Create a pie chart using Plotly
    fig_pie = px.pie(sentiment_counts,
                     values='Count',
                     names='Sentiment',
                     color='Sentiment',
                     color_discrete_map=color_mapping,
                     title='Sentiment Distribution')

    return fig_pie

if _name_ == '_main_':
    app.run_server(debug=True)
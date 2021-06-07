import os
import re
import nltk
import string


import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')


from os.path import join
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# location where the file is stored
file_path = "C-SCM-DATA-Candidates_Evaluation_Anonymized_SS21.xlsx"


# downloads that are needed for the text processing (downloaded once)
#nltk.download('vader_lexicon')

stopwords = nltk.corpus.stopwords.words("english")


def evaluator_count(data=None):
    if data is None:
        data = pd.read_excel(file_path, engine='openpyxl')
    eval_count = data.groupby('Evaluated by Code').count()[['Person ID', 'Agreed By Code']]

    eval_count.columns = ['evaluated', 'agreed']
    return eval_count

eval_count = evaluator_count()

# gives the evaluator (index), number of evaluations and count of evaluations which have been agreed by other persons
eval_count.head(10)



def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("\n \n CREATED DIRECTORY: {}".format(dir_path))


# checks whether number is included
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

# filters numbers (ids) from the statements
def filter_numbers(word_list):
    return [word for word in word_list if hasNumbers(word) is False]

def filter_stop_words(word_list):
    return [word for word in word_list if word.lower() not in stopwords]

def prepare_word_list():
    data = pd.read_excel(file_path, engine='openpyxl')
    statements = data['Evaluation Statement'].tolist()

    word_list = []
    for statement in statements:
        #statement = statements[0]
        if type(statement) == str:
            # remove punctuation marks
            statement = statement.translate(str.maketrans({a: None for a in string.punctuation}))
            # split statement into single words
            statement_words = nltk.word_tokenize(statement)
            # remove numeric values
            statement_words = filter_numbers(statement_words)
            # remove stopwords
            statement_words = filter_stop_words(statement_words)

            word_list += statement_words

    return word_list


def worde_freq_analysis(num_top=None):
    word_list = prepare_word_list()

    if num_top is None:
        num_top = len(np.unique(word_list))

    fd = nltk.FreqDist(word_list)

    most_common_words = pd.DataFrame(fd.most_common(num_top), columns=['word', 'count'])

    return most_common_words


word_ranking = worde_freq_analysis()

# top 20 words and the count of words
word_ranking.head(20)



def statement_sentiment(statement):
    vader = SentimentIntensityAnalyzer()
    polarity_dict = vader.polarity_scores(statement)
    return polarity_dict

def sentiment_data():
    data = pd.read_excel(file_path, engine='openpyxl')
    data.dropna(axis=0, inplace=True)
    data['Person ID'] = data['Person ID'].astype(int)
    data.set_index('Person ID', inplace=True)

    sentiments = ['neg', 'neu', 'pos', 'compound']

    for sentiment in sentiments:
        data[sentiment] = 0

    sentiment_index_dict = {}
    for index in data.index:
        statement = data.loc[index]['Evaluation Statement']
        try:
            sentiment_index_dict[index] = statement_sentiment(statement)

        except:
            print(statement)

        for sentiment in sentiments:
            data.loc[index, sentiment] = np.copy(sentiment_index_dict[index][sentiment])

    return data

data = sentiment_data()

data['sentiment'] = data['pos'] - data['neg']

# replace dummy line
data = data[data['Current Potential']!='tbd']


def plot_pie_chart(data, col, save_dir=True):
    x_data = data[col].value_counts().tolist()
    labels = data[col].unique().tolist()

    colors = ['#191970', '#001CF0', '#0038E2', '#0055D4', '#0071C6', '#008DB8', '#00AAAA',
              '#00C69C', '#00E28E', '#00FF80', ]
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"), )
    plt.legend(labels=labels, loc="best")


    explode = [0.1] * len(labels)

    def func(pct, allvals):
        absolute = int(round(pct / 100. * np.sum(allvals)))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(x_data,
                                      autopct=lambda pct: func(pct, x_data),
                                      textprops=dict(color="w"),
                                      explode=explode,
                                      shadow=True
                                      )

    ax.legend(wedges, labels,
              title=col,
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    autotexts = [autotext for autotext in autotexts if int(autotext._text.split("\n")[-1][1:-1]) > sum(x_data)*0.05 ]
    #plt.setp(autotexts, size=8, weight="bold")

    autotexts_small = [autotext for autotext in autotexts if int(autotext._text.split("\n")[-1][1:-1]) < sum(x_data)*0.05 ]
    #plt.setp(autotexts_small, size=0.1, weight="bold")


    if save_dir:
        save_dir = os.getcwd()
        for directory in ['plot', "pie_chart"]:
            save_dir = join(save_dir, directory)
            create_dir_if_not_exists(save_dir)

        file_name = "pie_chart_{column}.jpg".format(column=col)
        save_path = join(save_dir, file_name)
        plt.savefig(save_path)

    plt.show()
    plt.close()



#######################################
#        CREATE PIE PLOTS
#######################################

for category in ['Current Potential', 'Gender']:
    plot_pie_chart(data, col=category)

def plot_density(data, col, hue, save_dir=True):
    sns.displot(data, x=col, kind='kde', hue=hue, fill=True)

    if save_dir:
        save_dir = os.getcwd()
        for directory in ['plot', "density"]:
            save_dir = join(save_dir, directory)
            create_dir_if_not_exists(save_dir)

        file_name = "density_plot_{column}_{hue}.jpg".format(column=col, hue=hue)
        save_path = join(save_dir, file_name)
        plt.savefig(save_path)

    plt.show()
    plt.close()

sentiment_columns = ['neg', 'neu', 'pos', 'compound']
categorization = ['Current Potential', 'Gender']




def create_bar_chart(data, col, save_dir=True):
    frequencies = data.groupby(col).count()
    x_labels = frequencies.index.tolist()
    freq_series = pd.Series(frequencies['Level'].tolist())


    my_cmap = plt.get_cmap("viridis")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))


    # Plot the figure.
    plt.figure(figsize=(12, 8))
    ax = freq_series.plot(kind='bar', color=my_cmap(rescale(freq_series)))
    ax.set_title('Count: {}'.format(col))
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(x_labels)


    def add_value_labels(ax, spacing=5):
        """Add labels to the end of each bar in a bar chart.

        Arguments:
            ax (matplotlib.axes.Axes): The matplotlib object containing the axes
                of the plot to annotate.
            spacing (int): The distance between the labels and the bars.
        """

        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.1f}".format(y_value)

            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va)                      # Vertically align label differently for
                                            # positive and negative values.


    # Call the function above. All the magic happens there.
    add_value_labels(ax)



    if save_dir:
        save_dir = os.getcwd()
        for directory in ['plot', "bar_charts"]:
            save_dir = join(save_dir, directory)
            create_dir_if_not_exists(save_dir)

        file_name = "bar_chart_{col}.jpg".format(col=col)
        save_path = join(save_dir, file_name)
        plt.savefig(save_path)


    plt.show()
    plt.close()


for category in ['Current Potential', 'Gender']:
    create_bar_chart(data, col=category)





#######################################
#        CREATE DENSITY PLOTS
#######################################


for category in categorization:
    for sentiment in sentiment_columns:
        plot_density(data, col=sentiment, hue=category)


plot_density(data, col="pos", hue='Gender')

senti = data[sentiment_columns].head(10)

text = prepare_word_list()



#############################################
#       EXAMPLE BIAS
#############################################

code_count = data.groupby("Code").count().sort_values('Level', ascending=False)

top_code_count = code_count[code_count['Level']> 5].index.tolist()

top_data = data[data['Code'].isin(top_code_count)].groupby('Code').mean()

top_data['sentiment'] = top_data['pos'] - top_data['neg']

top_data = top_data.sort_values('sentiment', ascending=False)

min_sentimen = top_data['sentiment'].idxmin()
max_sentimen = top_data['sentiment'].idxmax()

sns.kdeplot(data[data['Code'] == min_sentimen]['sentiment'].values, label=min_sentimen)
sns.kdeplot(data[data['Code'] == max_sentimen]['sentiment'].values, label=max_sentimen)
plt.legend()


file_path = join(os.getcwd(), 'plot', "EXAMPLE.jpg")
plt.savefig(file_path)


plt.close()

def creat_word_cloud(text, save_dir=True):
    if type(text) == list:
        text = " ".join(text)

    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 100) ** 2 > 230 ** 2
    mask = 255 * mask.astype(int)

    wc = WordCloud(background_color="white", repeat=True, mask=mask)
    wc.generate(text)

    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    if save_dir:
        save_dir = os.getcwd()
        for directory in ['plot', "word_clouds"]:
            save_dir = join(save_dir, directory)
            create_dir_if_not_exists(save_dir)

        file_name = "word_clouds.jpg"
        save_path = join(save_dir, file_name)
        plt.savefig(save_path)

    plt.show()
    plt.close()


creat_word_cloud(text)


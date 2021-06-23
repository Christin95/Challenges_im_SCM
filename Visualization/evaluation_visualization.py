import os
import numpy as np
import pandas as pd
from math import pi

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from os.path import join
from wordcloud import WordCloud
from whatlies.language import SpacyLanguage

from Data.data_loader import load_processed_data
from Data.web_scraper import get_leaership_trait_list

#from Model.evaluation_model import get_nlp_lg, remove_punctuation, filter_adjectives
from utils import create_dir_if_not_exists
from Model.model_utils import get_eval_text, filter_numbers, remove_punctuation, filter_adjectives, get_nlp_lg


core_values_list = ['open', 'responsible', 'creative', 'entrepreneurial']
# list of color names available in matplotlib
color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values())

def visualize_word_embedding():
    nlp_lg = get_nlp_lg()
    language_model = SpacyLanguage(nlp_lg)

    text = get_eval_text(num_statements=2000)
    text = remove_punctuation(text)

    unique_words = list(set(text.split())) + core_values_list
    unique_words = unique_words[:-25]
    text = " ".join(unique_words)

    doc = nlp_lg(text.lower())

    adjectives = filter_adjectives(doc)

    embeddings = language_model[adjectives]
    embeddings.plot(kind="scatter", show_ops=True, x_axis='open', y_axis='creative', color="red")
    #embeddings.plot(kind="scatter", color="blue")

    plt.show()


#visualize_word_embedding()

def make_radar(data, num_radars=4, title="Radar Plot", index_col='Person ID', columns=core_values_list):
    if type(index_col) == str:
        group = index_col
        index_col = [index_col]
    else:
        group = index_col[0]

    #df = avg_values[index_col + trait_columns]
    df = data[index_col + columns].head(num_radars)
    num_y_ticks = 4

    y_tick_list = [np.round((i/num_y_ticks), 2) for i in range(1, num_y_ticks + 1)]
    categories = list(df)[1:]
    N = len(categories)

    # determination number of subplots
    num_plots_x = int(np.ceil(np.sqrt(len(df))))
    num_plots_y = int(np.ceil(len(df) / num_plots_x))

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(num_plots_y, num_plots_x, subplot_kw={'projection': 'polar'})
    fig.suptitle(title, size=11, color='grey', y=1.1)
    fig.tight_layout()

    row_index = 0
    for i in range(num_plots_y):
        for j in range(num_plots_x):
            color = color_list[row_index]
            plt.sca(ax[i, j])
            plt.xticks(angles[:-1], categories, color='grey', size=8)
            values = df.loc[row_index].drop(group).values.flatten().tolist()
            values += values[:1]
            ax[i, j].set_rlabel_position(0)

            ax[i, j].plot(angles, values, color=color, linewidth=0.5, linestyle='solid')
            ax[i, j].fill(angles, values, color=color, alpha=0.2)

            ax[i, j].set_yticks(y_tick_list)
            ax[i, j].set_yticklabels([str(y_tick) for y_tick in y_tick_list], fontsize=6, alpha=0.5)
            row_index += 1


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

    autotexts = [autotext for autotext in autotexts if int(autotext._text.split("\n")[-1][1:-1]) > sum(x_data)*0.05]
    #plt.setp(autotexts, size=8, weight="bold")

    autotexts_small = [autotext for autotext in autotexts if int(autotext._text.split("\n")[-1][1:-1]) < sum(x_data)*0.05]
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




if __name__ == "__main__":

    # example for the creation of radar plot
    data = load_processed_data()
    make_radar(data, num_radars=4)
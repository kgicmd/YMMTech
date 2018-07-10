import os
import warnings

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from plotnine import *

warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_matches_year(data: pd.DataFrame):
    """Plot year-match_count stats.

     Data structure:
    ['date', 'team1', 'team1Text', 'team2', 'team2Text', 'resText',
       'statText', 'venue', 'IdCupSeason', 'CupName', 'team1Score',
       'team2Score', 'team1PenScore', 'team2PenScore']
    """
    g = ggplot(mapping=aes('date.dt.year'), data=data) + \
        geom_bar(mapping=aes(fill="CupName"), width=1, color="black") + \
        theme(legend_position='bottom', legend_direction='vertical') + \
        ggtitle("Matches played by Year")

    print(g)


def plot_mean_goals_year(data: pd.DataFrame):
    """Plot year-mean_goals stats.

     Data structure:
    ['date', 'team1', 'team1Text', 'team2', 'team2Text', 'resText',
       'statText', 'venue', 'IdCupSeason', 'CupName', 'team1Score',
       'team2Score', 'team1PenScore', 'team2PenScore']
    """
    year = sorted(data.date.dt.year.drop_duplicates())
    goals_team_sum = data['team1Score'] + data['team2Score']
    goals_per_game = [goals_team_sum[data.date.dt.year == y].sum() / sum(data.date.dt.year == y) for y in year]
    new_data = pd.DataFrame({'year': year, 'goals_per_game': goals_per_game})

    g = ggplot(mapping=aes(x='year', y='goals_per_game'), data=new_data) + \
        geom_point() + \
        geom_smooth(method="loses") + ggtitle("Goals scored per game, over time")

    print(g)


def plot_missing_map(data: pd.DataFrame):
    """Plot missing map of the data.

     Data structure:
    ['date', 'team1', 'team1Text', 'team2', 'team2Text', 'resText',
       'statText', 'venue', 'IdCupSeason', 'CupName', 'team1Score',
       'team2Score', 'team1PenScore', 'team2PenScore']
    """
    msno.matrix(data)
    plt.show()


def plot_win_percent(team_perf: pd.DataFrame, min_games: int = 100):
    """Plot win percentage diagram.

    Data structure:
    ['match_id': index, 'date', 'name', 'opponentName', 'homeVenue',
    'neutralVenue', 'gs', 'ga', 'gd', 'win', 'loss', 'draw',
    'friendly', 'qualifier', 'finaltourn']
    """

    def win_percent(wins, draws, total):
        return (wins + 0.5 * draws) / total

    # Group by team name.
    groups = team_perf.groupby('name')

    names, win_percentage, total_games = [], [], []
    for name, group in groups:
        count = len(group)
        if count >= min_games:
            names.append(name)
            win_percentage.append(
                win_percent(sum(group.win), sum(group.draw), count))
            total_games.append(count)

    new_data = pd.DataFrame({'name': names, 'win_percentage': win_percentage,
                             'total_games': total_games})

    g = ggplot(mapping=aes(x='win_percentage', y='total_games'), data=new_data) + \
        geom_point(size=1.5) + \
        geom_text(aes(label='name'), ha='left', va='bottom') + \
        geom_vline(xintercept=.5, linetype='solid', color="red") + \
        ggtitle("Winning Percentage vs Games Played") + \
        expand_limits(x=(0, 1))

    print(g)


def plot_goal_sum_per_match(data: pd.DataFrame):
    """Plot goals scored per match distribution.

    Data structure.
    ['gameSum','count','freq']
    """
    data = data[data.freq >= 0.01]
    g = ggplot(mapping=aes(x='goalScore', y='freq'), data=data) + \
        geom_bar(stat="identity") + \
        geom_text(aes(label='round(freq, 2)'), va='top', nudge_y=0.01) + \
        ggtitle('Goals scored per match distribution')

    print(g)


def plot_goal_diff_per_match(data: pd.DataFrame):
    """Plot goals scored per match distribution.

    Data structure.
    ['goalDiff','count','freq']
    """
    data = data[abs(data.goalDiff) <= 4]
    g = ggplot(mapping=aes(x='goalDiff', y='freq'), data=data) + \
        geom_bar(stat="identity") + \
        geom_text(aes(label='round(freq, 2)'), va='top', nudge_y=0.01) + \
        ggtitle('Goal differential distribution')
    print(g)


def plot_features_corr(data: pd.DataFrame):
    """Plot data correlation.

    Data structure.
    ['user_id', 'id', 'label', 'cargo_type', 'handling_type', 'truck_count',
    'highway', 'lcl_cargo', 'truck_len_match', 'regular_subscribe_match',
    'line_match', 'line30_match', 'line60_match', 'scan_cargo_level_3',
    'scan_cargo_level_7', 'scan_cargo_level_14', 'scan_cargo_level_30',
    'click_cargo_level_3', 'click_cargo_level_7', 'click_cargo_level_14',
    'click_cargo_level_30', 'call_cargo_level_3', 'call_cargo_level_7',
    'call_cargo_level_14', 'call_cargo_level_30]
    """
    data = data.drop(columns=['user_id', 'id'])
    # Compute the correlation matrix
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()


def plot_scatters(data: list, x_title, y_title, title=None):
    data = pd.DataFrame({x_title: data.index, y_title: data})
    g = ggplot(mapping=aes(x=x_title, y=y_title), data=data) + \
        geom_point(shape='o', size=2, fill='', color='black') + \
        ggtitle(title)
    print(g)


def plot_wins_per_team(sim_results: pd.DataFrame):
    new_data = sim_results.groupby(by='winner').size().reset_index(name='wins')
    new_data = new_data.sort_values(by='wins', ascending=False)
    new_data = new_data.reset_index(drop=True)

    new_data['winner'] = pd.Categorical(new_data['winner'], categories=reversed(new_data.winner))

    g = ggplot(mapping=aes(x='winner', y='wins'), data=new_data) + \
        geom_bar(stat="identity", ) + \
        coord_flip() + \
        geom_text(aes(label="wins/100", va=0.3, ha=0.5, size=3)) + \
        ggtitle("Tournament simulation winners (10,000 iterations)")
    print(g)


if __name__ == '__main__':
    preprocess_dir = os.path.join('data', 'preprocess')

    train_features = pd.read_csv(os.path.join(preprocess_dir, 'train_features.csv'))
    plot_features_corr(train_features)

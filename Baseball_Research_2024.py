import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from sklearn.preprocessing import LabelEncoder
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import r2_score
import seaborn as sns
import sys

# Definition: Expected Weighted On-base Average (xwOBA) is formulated using exit velocity, launch angle and,
# on certain types of batted balls, Sprint Speed. In the same way that each batted ball is assigned an expected
# batting average, every batted ball is given a single, double, triple and home run probability based on the results
# of comparable batted balls since StatCast was implemented Major League wide in 2015. For the majority of batted
# balls, this is achieved using only exit velocity and launch angle. As of 2019, "topped" or "weakly hit" balls also
# incorporate a batter's seasonal Sprint Speed. All hit types are valued in the same fashion for xwOBA as they are in
# the formula for standard wOBA: (unintentional BB factor x unintentional BB + HBP factor x HBP + 1B factor x 1B + 2B
# factor x 2B + 3B factor x 3B + HR factor x HR)/(AB + unintentional BB + SF + HBP), where "factor" indicates the
# adjusted run expectancy of a batting event in the context of the season as a whole. Knowing the expected outcomes
# of each individual batted ball from a particular player over the course of a season -- with a player's real-world
# data used for factors such as walks, strikeouts and times hit by a pitch -- allows for the formation of said
# player's xwOBA based on the quality of contact, instead of the actual outcomes. Likewise, this exercise can be done
# for pitchers to get their expected xwOBA against.

# Why it's useful xwOBA is more indicative of a player's skill than regular wOBA, as xwOBA removes defense from the
# equation. Hitters, and likewise pitchers, are able to influence exit velocity and launch angle but have no control
# over what happens to a batted ball once it is put into play.

# Below 0.300 is considered poor
# 0.320 is considered average
# 0.350 is considered good
# 0.400 is considered excellent

# All data utilized is owned and maintained by Baseball Savant, I do not own this data. Only utilizing it for analysis.
# Assuming you have data for each player and season in a CSV file named 'player_stats.csv'
data2023 = pd.read_csv(r'C:\Users\jsric\OneDrive\Documents\expected_stats_2023_new.csv', encoding='utf-8')
data2022 = pd.read_csv(r'C:\Users\jsric\OneDrive\Documents\expected_stats_2022_new.csv', encoding='utf-8')
data2021 = pd.read_csv(r'C:\Users\jsric\OneDrive\Documents\expected_stats_2021_new.csv', encoding='utf-8')
exitdata2023 = pd.read_csv(r'C:\Users\jsric\OneDrive\Documents\exit_velocity_2023_new.csv', encoding='utf-8')
exitdata2022 = pd.read_csv(r'C:\Users\jsric\OneDrive\Documents\exit_velocity_2022_new.csv', encoding='utf-8')
exitdata2021 = pd.read_csv(r'C:\Users\jsric\OneDrive\Documents\exit_velocity_2021_new.csv', encoding='utf-8')

# Merge the datasets based on 'player_id' for each year
merged_data2023 = pd.merge(data2023, exitdata2023, on='player_id')
merged_data2022 = pd.merge(data2022, exitdata2022, on='player_id')
merged_data2021 = pd.merge(data2021, exitdata2021, on='player_id')

# Combine the merged datasets into a single dataframe
all_data = pd.concat([merged_data2023, merged_data2022, merged_data2021])

# Resetting the index to avoid any issues
all_data.reset_index(drop=True, inplace=True)

# Drop the original columns if needed
all_data.drop(['year_y'], axis=1, inplace=True)
all_data.drop(['player_name_y'], axis=1, inplace=True)
all_data.rename(columns={'player_name_x': 'player_name'}, inplace=True)
all_data.rename(columns={'year_x': 'year'}, inplace=True)

#######################################################################################################################
# First Regression Test on xWOBA.
label_encoder = LabelEncoder()
all_data['player_label'] = label_encoder.fit_transform(all_data['player_name'])

# Prepare the data
X = all_data[['year', 'player_label', 'est_ba', 'brl_pa', 'est_slg']]
y = all_data['est_woba']

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Add a new column for predicted est_woba using features of xBatting Average, barrels per plate appearance, and xSLG
all_data['predicted_est_woba(advanced)'] = model.predict(X)

# Evaluate the model (you can replace this with your own evaluation metrics)
mse = mean_squared_error(y, all_data['predicted_est_woba(advanced)'])
print(f'Mean Squared Error: {mse}')

top_players = all_data.nlargest(10, 'predicted_est_woba(advanced)')
bottom_players = all_data.nsmallest(5, 'predicted_est_woba(advanced)')

plt.figure(figsize=(10, 6))
sns.regplot(x='est_woba', y='predicted_est_woba(advanced)', data=all_data, scatter_kws={'s': 20}, line_kws={'color': 'red'})

# Calculate and display R-squared
r_squared = r2_score(y, all_data['predicted_est_woba(advanced)'])
plt.title(f'Regression Test for xWOBA Across the entire MLB (min 100 PA) with R-squared: {r_squared:.2f}')
plt.xlabel('Actual xWOBA')
plt.ylabel('Predicted xWOBA')

# Annotate the top players
for i, player in top_players.iterrows():
    plt.annotate(player['player_name'], (player['est_woba'], player['predicted_est_woba(advanced)']), textcoords="offset points", xytext=(0, 30), ha='center', va='center', fontsize=8, color='black', bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.8), rotation=60)

# Annotate the bottom players
for i, player in bottom_players.iterrows():
    plt.annotate(player['player_name'], (player['est_woba'], player['predicted_est_woba(advanced)']), textcoords="offset points", xytext=(0, -30), ha='center', va='center', fontsize=8, color='black', bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.8), rotation=60)
plt.show()
#############################################################################################################3

# ---Bar chart of the top predicted xWOBA for 2024--------------------------------------------------------------------

# Closer look at the top 10 leaders in expected 2024 xWOBA
# Sort the DataFrame by 'predicted_woba_2024' in descending order
top_players = all_data.sort_values(by='predicted_est_woba(advanced)', ascending=False).drop_duplicates('player_name').head(30)

cmap = cm.get_cmap('Blues_r')

# Reverse the normalization to have dark colors for higher values
norm = colors.Normalize(vmin=top_players['predicted_est_woba(advanced)'].min(), vmax=top_players['predicted_est_woba(advanced)'].max())

# Create a ScalarMappable with the colormap and reversed normalization
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

# Create a bar chart
plt.figure(figsize=(14, 8))
bars = plt.bar(top_players['player_name'], top_players['predicted_est_woba(advanced)'], color=cmap(norm(top_players['predicted_est_woba(advanced)'])))
plt.xlabel('Player Name')
plt.ylabel('Predicted xWOBA for 2024')
plt.title('Top 30 Players with Highest Predicted xWOBA for 2024')
plt.xticks(rotation=45, ha='right')
plt.yticks(np.arange(0, 1.0, 0.1))

# Add color bar for better interpretation
cbar = plt.colorbar(sm, label='Color Intensity')

plt.tight_layout()
plt.show()
##################################################################################################################
# Regression Test on xBA.
label_encoder = LabelEncoder()
all_data['player_label'] = label_encoder.fit_transform(all_data['player_name'])

# Prepare the data
W = all_data[['year', 'player_label', 'ba', 'brl_pa', 'est_slg']]
z = all_data['est_ba']

# Train a linear regression model
model = LinearRegression()
model.fit(W, z)

# Add a new column for predicted est_woba using features of xBatting Average, barrels per plate appearance, and xSLG
all_data['predicted_est_ba'] = model.predict(W)

# Evaluate the model (you can replace this with your own evaluation metrics)
mse = mean_squared_error(z, all_data['predicted_est_ba'])
print(f'Mean Squared Error for xBA: {mse}')

top_players = all_data.nlargest(10, 'predicted_est_ba')
bottom_players = all_data.nsmallest(5, 'predicted_est_ba')

plt.figure(figsize=(10, 6))
sns.regplot(x='est_ba', y='predicted_est_ba', data=all_data, scatter_kws={'s': 20}, line_kws={'color': 'red'})

# Calculate and display R-squared
r_squared = r2_score(z, all_data['predicted_est_ba'])
plt.title(f'Regression Test for xBA Across the entire MLB (min 100 PA) with R-squared: {r_squared:.2f}')
plt.xlabel('Actual xBatting Average')
plt.ylabel('Predicted xBatting Average')

# Annotate the top players
for i, player in top_players.iterrows():
    plt.annotate(player['player_name'], (player['est_ba'], player['predicted_est_ba']), textcoords="offset points", xytext=(0, 30), ha='center', va='center', fontsize=8, color='black', bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.8), rotation=60)

# Annotate the bottom players
for i, player in bottom_players.iterrows():
    plt.annotate(player['player_name'], (player['est_ba'], player['predicted_est_ba']), textcoords="offset points", xytext=(0, -30), ha='center', va='center', fontsize=8, color='black', bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.8), rotation=60)
plt.show()
##################################################################################################################
# Regression Test on xSLG.
label_encoder = LabelEncoder()
all_data['player_label'] = label_encoder.fit_transform(all_data['player_name'])

# Prepare the data
A = all_data[['year', 'player_label', 'avg_hit_angle', 'avg_hit_speed', 'brl_pa']]
b = all_data['est_slg']

# Train a linear regression model
model = LinearRegression()
model.fit(A, b)

# Add a new column for predicted est_woba using features of xBatting Average, barrels per plate appearance, and xSLG
all_data['predicted_est_slg'] = model.predict(A)

# Evaluate the model (you can replace this with your own evaluation metrics)
mse = mean_squared_error(b, all_data['predicted_est_slg'])
print(f'Mean Squared Error for xSLG: {mse}')

top_players = all_data.nlargest(10, 'predicted_est_slg')
bottom_players = all_data.nsmallest(5, 'predicted_est_slg')

plt.figure(figsize=(10, 6))
sns.regplot(x='est_slg', y='predicted_est_slg', data=all_data, scatter_kws={'s': 20}, line_kws={'color': 'red'})

# Calculate and display R-squared
r_squared = r2_score(b, all_data['predicted_est_slg'])
plt.title(f'Regression Test for xSLG Across the entire MLB (min 100 PA) with R-squared: {r_squared:.2f}')
plt.xlabel('Actual xSLG')
plt.ylabel('Predicted xSLG')

# Annotate the top players
for i, player in top_players.iterrows():
    plt.annotate(player['player_name'], (player['est_slg'], player['predicted_est_slg']), textcoords="offset points", xytext=(0, 30), ha='center', va='center', fontsize=8, color='black', bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.8), rotation=60)

# Annotate the bottom players
for i, player in bottom_players.iterrows():
    plt.annotate(player['player_name'], (player['est_slg'], player['predicted_est_slg']), textcoords="offset points", xytext=(0, -30), ha='center', va='center', fontsize=8, color='black', bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.8), rotation=60)
plt.show()

#######################################################################################################################
# Regression Test on Barrel per Plate Appearance.
label_encoder = LabelEncoder()
all_data['player_label'] = label_encoder.fit_transform(all_data['player_name'])

# Prepare the data
C = all_data[['year', 'player_label', 'avg_hit_angle', 'avg_hit_speed', 'avg_distance']]
d = all_data['brl_pa']

# Train a linear regression model
model = LinearRegression()
model.fit(C, d)

# Add a new column for predicted est_woba using features of xBatting Average, barrels per plate appearance, and xSLG
all_data['predicted_brl_pa'] = model.predict(C)

# Evaluate the model (you can replace this with your own evaluation metrics)
mse = mean_squared_error(d, all_data['predicted_brl_pa'])
print(f'Mean Squared Error for Barrels per Plate Appearance: {mse}')

top_players = all_data.nlargest(10, 'predicted_brl_pa')
bottom_players = all_data.nsmallest(5, 'predicted_brl_pa')

plt.figure(figsize=(10, 6))
sns.regplot(x='brl_pa', y='predicted_brl_pa', data=all_data, scatter_kws={'s': 20}, line_kws={'color': 'red'})

# Calculate and display R-squared
r_squared = r2_score(d, all_data['predicted_brl_pa'])
plt.title(f'Regression Test for Barrels per Plate Appearance Across the entire MLB (min 100 PA) with R-squared: {r_squared:.2f}')
plt.xlabel('Actual Barrels per Plate Appearance')
plt.ylabel('Predicted Barrels per Plate Appearance')

# Annotate the top players
for i, player in top_players.iterrows():
    plt.annotate(player['player_name'], (player['brl_pa'], player['predicted_brl_pa']), textcoords="offset points", xytext=(0, 30), ha='center', va='center', fontsize=8, color='black', bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.8), rotation=60)

# Annotate the bottom players
for i, player in bottom_players.iterrows():
    plt.annotate(player['player_name'], (player['brl_pa'], player['predicted_brl_pa']), textcoords="offset points", xytext=(0, -30), ha='center', va='center', fontsize=8, color='black', bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white', alpha=0.8), rotation=60)
plt.show()
#######################################################################################################################
# print(all_data.to_string())

fielding_data = pd.read_csv(r'C:\Users\jsric\Downloads\fielding_run_value_new.csv', encoding='utf-8')
data_plus_fielding = pd.merge(all_data, fielding_data, on='player_id')
data_plus_fielding.drop(['player_name_y'], axis=1, inplace=True)
data_plus_fielding.drop(['year_y'], axis=1, inplace=True)
data_plus_fielding = data_plus_fielding.rename(columns={'player_name_x': 'player_name'})
data_plus_fielding = data_plus_fielding.rename(columns={'year_x': 'year'})
data_plus_fielding = data_plus_fielding[['player_name', 'team'] + [col for col in data_plus_fielding.columns if col not in ['player_name', 'team']]]

extra_data = pd.read_csv(r'C:\Users\jsric\OneDrive\Documents\2023extrastats.csv', encoding='utf-8')

# print(data_plus_fielding.to_string())

#######################################################################################################################
# Anomaly Detection on run value to observe some of the outliers of the field
data_plus_fielding['z_score'] = zscore(data_plus_fielding['run_value'])
# Define a threshold for anomaly detection (e.g., 3 standard deviations)
threshold = 3
# Identify anomalies
anomalies = data_plus_fielding[abs(data_plus_fielding['z_score']) > threshold]
# Print or analyze the anomalies
anomalies_unique_run_value = anomalies[['player_name', 'team', 'run_value', 'outs', 'z_score']].drop_duplicates(subset=['player_name'])
# Print the unique anomalies
print("Anomaly Players in terms of Run Value that either dramatically hurt their teams chances or helped them:")
print(anomalies_unique_run_value)
print("\n" + "="*40 + "\n")  # Add a separator line

######################################################################################################################

# Anomaly Detection on xWOBA to observe some of the outliers of the field
data_plus_fielding['z_score'] = zscore(data_plus_fielding['predicted_est_woba(advanced)'])
# Define a threshold for anomaly detection (e.g., 3 standard deviations)
threshold = 3
# Identify anomalies
anomalies = data_plus_fielding[abs(data_plus_fielding['z_score']) > threshold]
# Print or analyze the anomalies
anomalies_unique_xWoba = anomalies[['player_name', 'team', 'predicted_est_woba(advanced)', 'ba', 'brl_percent']].drop_duplicates(subset=['player_name'])
# Print the unique anomalies
print("Anomaly Players in terms of xWOBA for 2024 exceed the norm: ")
print(anomalies_unique_xWoba.to_string())
print("\n" + "="*40 + "\n")  # Add a separator line

######################################################################################################################

def generate_team_plots(team_name):
    # Filter data for the specified team
    filtered_data = data_plus_fielding[(data_plus_fielding['team'] == team_name) & (data_plus_fielding['year'] == 2023)]

    # Merge with extra_data
    team_df = pd.merge(filtered_data, extra_data, on=['player_id', 'year'], how='inner')

    # Rename columns
    team_df.rename(columns={'player_name_x': 'player_name', 'team_x': 'team'}, inplace=True)

    # Drop unnecessary columns
    team_df.drop(['player_name_y', 'team_y'], axis=1, inplace=True)

    # Display data frame
    print(f"{team_name} Data Frame:")
    print(team_df.to_string())
    print("\n" + "=" * 40 + "\n")  # Add a separator line
######################################################################################################################
    # Get the top 5 players in terms of run_value
    top5_run_value = team_df.sort_values(by='run_value', ascending=False).head(5).reset_index(drop=True)
    print(f"Top 5 {team_name} Players in terms of Run Value:")
    print(top5_run_value[['player_name', 'run_value']])
    print("\n" + "=" * 40 + "\n")  # Add a separator line

    # Get the top 5 players in terms of outs
    top5_outs = team_df.sort_values(by='outs', ascending=False).head(5).reset_index(drop=True)
    print(f"Top 5 {team_name} Players in terms of Outs:")
    print(top5_outs[['player_name', 'outs']])
    print("\n" + "=" * 40 + "\n")  # Add a separator line
######################################################################################################################
    # Get the top 5 players in terms of x batting average
    top5_xbatting_average = team_df.sort_values(by='est_ba', ascending=False).head(5).reset_index(drop=True)
    print(f"Top 5 {team_name}Players in terms of Batting Average:")
    print(top5_xbatting_average[['player_name', 'ba']])
    print("\n" + "=" * 40 + "\n")  # Add a separator line

    # Get the top 5 players in terms of x batting average for 2024
    top5_x_batting_average_2024 = team_df.sort_values(by='predicted_est_ba', ascending=False).head(5).reset_index(drop=True)
    print(f"Top 5 {team_name} Players in terms of x Batting Average:")
    print(top5_x_batting_average_2024[['player_name', 'est_ba']])
    print("\n" + "=" * 40 + "\n")  # Add a separator line
######################################################################################################################
    # Get the top 5 players in terms of predicted_est_woba(advanced)
    top5_est_woba_2024 = team_df.sort_values(by='predicted_est_woba(advanced)', ascending=False).head(5).reset_index(drop=True)
    print(f"Top 5 {team_name} Players in terms of Predicted Estimated xWOBA for 2024:")
    print(top5_est_woba_2024[['player_name', 'predicted_est_woba(advanced)']])
    print("\n" + "=" * 40 + "\n")  # Add a separator line

    # Get the top 5 players in terms of xWOBA
    top5_est_woba = team_df.sort_values(by='est_woba', ascending=False).head(5).reset_index(drop=True)
    print(f"Top 5 {team_name} Players in terms of xWOBA:")
    print(top5_est_woba[['player_name', 'est_woba']])
    print("\n" + "=" * 40 + "\n")  # Add a separator line
######################################################################################################################
    # Get the top 5 players in terms of OPS
    top5_est_ops = team_df.sort_values(by='on_base_plus_slg', ascending=False).head(5).reset_index(drop=True)
    print(f"Top 5 {team_name} Players in terms of OPS:")
    print(top5_est_ops[['player_name', 'on_base_plus_slg']])
    print("\n" + "=" * 40 + "\n")  # Add a separator line

    # Get the top 5 players in terms of xISO
    top5_est_xiso = team_df.sort_values(by='xiso', ascending=False).head(5).reset_index(drop=True)
    print(f"Top 5 {team_name} Players in terms of xISO:")
    print(top5_est_xiso[['player_name', 'xiso']])
    print("\n" + "=" * 40 + "\n")  # Add a separator line
#######################################################################################################################
    # Get the top 5 players in terms of xSLG for 2024
    top5_xslg_2024 = team_df.sort_values(by='predicted_est_slg', ascending=False).head(5).reset_index(drop=True)
    print(f"Top 5 {team_name} Players in terms of 2024 Predicted xSLG:")
    print(top5_xslg_2024[['player_name', 'predicted_est_slg']])
    print("\n" + "=" * 40 + "\n")  # Add a separator line

    # Get the top 5 players in terms of xSLG
    top5_xslg = team_df.sort_values(by='est_slg', ascending=False).head(5).reset_index(drop=True)
    print(f"Top 5 {team_name} Players in terms of xSLG:")
    print(top5_xslg[['player_name', 'est_slg']])
    print("\n" + "=" * 40 + "\n")  # Add a separator line
######################################################################################################################
    # Get the top 5 players in terms of Barrels per PA for 2024
    top5_brl_pa_2024 = team_df.sort_values(by='predicted_brl_pa', ascending=False).head(5).reset_index(drop=True)
    print(f"Top 5 {team_name} Players in terms of 2024 Predicted Barrels per Plate Appearance:")
    print(top5_brl_pa_2024[['player_name', 'predicted_brl_pa']])
    print("\n" + "=" * 40 + "\n")  # Add a separator line

    # Get the top 5 players in terms of Barrels per PA
    top5_brl_pa = team_df.sort_values(by='brl_pa', ascending=False).head(5).reset_index(drop=True)
    print(f"Top 5 {team_name} Players in terms of Barrels per Plate Appearance:")
    print(top5_brl_pa[['player_name', 'brl_pa']])
    print("\n" + "=" * 40 + "\n")  # Add a separator line
#####################################################################################################################
    # xWoba Bar Charts
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Set the same color for both bar charts
    bar_color = 'slategrey'

    # Bar chart for top 5 players in terms of predicted_est_woba(advanced) for 2024
    axs[0].bar(top5_est_woba_2024['player_name'], top5_est_woba_2024['predicted_est_woba(advanced)'], color=bar_color)
    axs[0].set_title(f'Top 5 {team_name} Players - 2024 Predicted Estimated xWOBA')
    axs[0].set_ylabel('2024 Predicted Estimated xWOBA')
    axs[0].tick_params(axis='x', rotation=45, labelsize=8)
    axs[0].set_xlabel('Player Name')
    axs[0].grid(axis='y')

    # Add value annotations at the top of each bar for the first subplot
    for i, v in enumerate(top5_est_woba_2024['predicted_est_woba(advanced)']):
        axs[0].text(i, v + 0.01, str(round(v, 3)), ha='center', va='bottom', fontsize=8)

    # Bar chart for top 5 players in terms of xWOBA
    axs[1].bar(top5_est_woba['player_name'], top5_est_woba['est_woba'], color=bar_color)
    axs[1].set_title(f'Top 5 {team_name} Players - xWOBA')
    axs[1].set_ylabel('xWOBA')
    axs[1].tick_params(axis='x', rotation=45, labelsize=8)
    axs[1].set_xlabel('Player Name')
    axs[1].grid(axis='y')

    # Add value annotations at the top of each bar for the second subplot
    for i, v in enumerate(top5_est_woba['est_woba']):
        axs[1].text(i, v + 0.01, str(round(v, 3)), ha='center', va='bottom', fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
######################################################################################################################
    # Batting Average Bar Charts
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Set the color for both bar charts
    bar_color = 'slategrey'

    # Bar chart for top 5 players in terms of x batting average for 2024
    axs[0].bar(top5_x_batting_average_2024['player_name'], top5_x_batting_average_2024['predicted_est_ba'], color=bar_color)
    axs[0].set_title(f'Top 5 {team_name} Players - 2024 Predicted xBatting Average')
    axs[0].set_ylabel('2024 xBatting Average')
    axs[0].tick_params(axis='x', rotation=45, labelsize=8)
    axs[0].set_xlabel('Player Name')
    axs[0].grid(axis='y')

    # Add value annotations at the top of each bar for the first subplot
    for i, v in enumerate(top5_x_batting_average_2024['predicted_est_ba']):
        axs[0].text(i, v + 0.001, str(round(v, 3)), ha='center', va='bottom', fontsize=8)

    # Bar chart for top 5 players in terms of x batting average
    axs[1].bar(top5_xbatting_average['player_name'], top5_xbatting_average['est_ba'], color=bar_color)
    axs[1].set_title(f'Top 5 {team_name} Players - xBatting Average')
    axs[1].set_ylabel('xBatting Average')
    axs[1].tick_params(axis='x', rotation=45, labelsize=8)
    axs[1].set_xlabel('Player Name')
    axs[1].grid(axis='y')

    # Add value annotations at the top of each bar for the second subplot
    for i, v in enumerate(top5_x_batting_average_2024['est_ba']):
        axs[1].text(i, v + 0.001, str(round(v, 3)), ha='center', va='bottom', fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
######################################################################################################################
    # Bar charts for xSLG
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Set the color for both bar charts
    bar_color = 'slategrey'

    # Bar chart for top 5 players in terms of xSLG for 2024
    axs[0].bar(top5_xslg_2024['player_name'], top5_xslg_2024['predicted_est_slg'],
               color=bar_color)
    axs[0].set_title(f'Top 5 {team_name} Players - 2024 Predicted xSLG')
    axs[0].set_ylabel('2024 xSLG')
    axs[0].tick_params(axis='x', rotation=45, labelsize=8)
    axs[0].set_xlabel('Player Name')
    axs[0].grid(axis='y')

    # Add value annotations at the top of each bar for the first subplot
    for i, v in enumerate(top5_xslg_2024['predicted_est_slg']):
        axs[0].text(i, v + 0.001, str(round(v, 3)), ha='center', va='bottom', fontsize=8)

    # Bar chart for top 5 players in terms of xSLG
    axs[1].bar(top5_xslg['player_name'], top5_xslg['est_slg'], color=bar_color)
    axs[1].set_title(f'Top 5 {team_name} Players - xSLG')
    axs[1].set_ylabel('xSLG')
    axs[1].tick_params(axis='x', rotation=45, labelsize=8)
    axs[1].set_xlabel('Player Name')
    axs[1].grid(axis='y')

    # Add value annotations at the top of each bar for the second subplot
    for i, v in enumerate(top5_xslg['est_slg']):
        axs[1].text(i, v + 0.001, str(round(v, 3)), ha='center', va='bottom', fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
######################################################################################################################
    # Bar Charts for Barrels per PA
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Set the color for both bar charts
    bar_color = 'slategrey'

    # Bar chart for top 5 players in terms of Brl per PA for 2024
    axs[0].bar(top5_brl_pa_2024['player_name'], top5_brl_pa_2024['predicted_brl_pa'],
               color=bar_color)
    axs[0].set_title(f'Top 5 {team_name} Players - 2024 Predicted Barrels per Plate Appearance')
    axs[0].set_ylabel('2024 Barrels per Appearance')
    axs[0].tick_params(axis='x', rotation=45, labelsize=8)
    axs[0].set_xlabel('Player Name')
    axs[0].grid(axis='y')

    # Add value annotations at the top of each bar for the first subplot
    for i, v in enumerate(top5_brl_pa_2024['predicted_brl_pa']):
        axs[0].text(i, v + 0.001, str(round(v, 3)), ha='center', va='bottom', fontsize=8)

    # Bar chart for top 5 players in terms of Brl per PA
    axs[1].bar(top5_brl_pa['player_name'], top5_brl_pa['brl_pa'], color=bar_color)
    axs[1].set_title(f'Top 5 {team_name} Players - Barrels per Plate Appearance')
    axs[1].set_ylabel('Barrels per Plate Appearance')
    axs[1].tick_params(axis='x', rotation=45, labelsize=8)
    axs[1].set_xlabel('Player Name')
    axs[1].grid(axis='y')

    # Add value annotations at the top of each bar for the second subplot
    for i, v in enumerate(top5_brl_pa['brl_pa']):
        axs[1].text(i, v + 0.001, str(round(v, 3)), ha='center', va='bottom', fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
######################################################################################################################
    # Run Value and Outs Bar Charts
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Set the color for both bar charts
    bar_color = 'slategrey'

    # Bar chart for top 5 players in terms of run value
    axs[0].bar(top5_run_value['player_name'], top5_run_value['run_value'], color=bar_color)
    axs[0].set_title(f'Top 5 {team_name} Players - Run Value')
    axs[0].set_ylabel('Run Value')
    axs[0].tick_params(axis='x', rotation=45, labelsize=8)
    axs[0].set_xlabel('Player Name')
    axs[0].grid(axis='y')

    # Add value annotations at the top of each bar for the first subplot
    for i, v in enumerate(top5_run_value['run_value']):
        axs[0].text(i, v + 0.1, str(round(v, 3)), ha='center', va='bottom', fontsize=8)

    # Bar chart for top 5 players in terms of outs
    axs[1].bar(top5_outs['player_name'], top5_outs['outs'], color=bar_color)
    axs[1].set_title(f'Top 5 {team_name} Players - Outs')
    axs[1].set_ylabel('Outs')
    axs[1].tick_params(axis='x', rotation=45, labelsize=8)
    axs[1].set_xlabel('Player Name')
    axs[1].grid(axis='y')

    # Add value annotations at the top of each bar for the second subplot
    for i, v in enumerate(top5_outs['outs']):
        axs[1].text(i, v + 0.1, str(round(v, 3)), ha='center', va='bottom', fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
######################################################################################################################
    # Scatterplot for Barrel per PA and avg hit speed with xSLUG as value
    # Set the size of the points based on est_slg

    league_average_slg = 0.413
    point_size = 100  # Set a default size for all points

    # Create a scatter plot with color representing est_slg values
    scatter = plt.scatter(team_df['avg_hit_speed'], team_df['brl_pa'], s=point_size, alpha=0.5, c=team_df['est_slg'],
        cmap='Blues', vmin=team_df['est_slg'].min(), vmax=team_df['est_slg'].max())

    # Set labels and title
    plt.xlabel('Average Hit Speed')
    plt.ylabel('Barrel per Plate Appearance')
    plt.title(f'Scatter Plot of Average Hit Speed vs. Barrel per Plate Appearance for the {team_name}')

    # Add a colorbar to show est_slg values
    cbar = plt.colorbar(scatter)
    cbar.set_label('est_slg')

    # Add player_name and est_slg annotations
    for i, row in team_df.iterrows():
        plt.annotate(f"{row['player_name']}\n{row['est_slg']:.3f}", (row['avg_hit_speed'], row['brl_pa']), fontsize=8,
                    ha='center', va='bottom')

    # Show the plot
    plt.show()
######################################################################################################################
    # Scatter PLot for Damage
    point_size = 100  # Set a default size for all points
    # Create a scatter plot with color representing est_slg values
    scatter = plt.scatter(team_df['on_base_plus_slg'], team_df['xiso'], s=point_size, alpha=0.5, c=team_df['player_age'],
                        cmap='Blues', vmin=team_df['player_age'].min(), vmax=team_df['player_age'].max())

    # Set labels and title
    plt.xlabel('On Base Plus Slug')
    plt.ylabel('x Isolated Power')
    plt.title(f'Scatter Plot of OPS vs. xISO with Players Age for the {team_name}')

    cbar = plt.colorbar(scatter)
    cbar.set_label('player_age')

    for i, row in team_df.iterrows():
        age_label = f"$\\bf{{{row['player_name']}}}$\n$\\bf{{{int(row['player_age'])}}}$"  # Bold text using LaTeX-like syntax
        plt.annotate(age_label, (row['on_base_plus_slg'], row['xiso']), fontsize=8, ha='center', va='bottom')

    plt.show()
######################################################################################################################
    # Points Function
    player_points = {}

    def assign_points(top5_df):
        for idx, row in top5_df.iterrows():
            player_name = row['player_name']
            points = len(top5_df) - idx  # Ensure points are non-negative
            if player_name in player_points:
                player_points[player_name] += points
            else:
                player_points[player_name] = points

    # Assign points based on ranking
    assign_points(top5_run_value)
    assign_points(top5_outs)
    assign_points(top5_x_batting_average_2024)
    assign_points(top5_xslg_2024)
    assign_points(top5_est_woba_2024)
    assign_points(top5_brl_pa_2024)
    assign_points(top5_est_ops)
    assign_points(top5_est_xiso)

    # Display the total points for each player
    for player, points in sorted(player_points.items(), key=lambda x: x[1], reverse=True):
        print(f"{player}: {points} points")
    if player_points:
        max_player, max_points = max(player_points.items(), key=lambda x: x[1])
        print(f"\nPlayer with the most points: {max_player} with {max_points} points")
    else:
        print("\nNo players to evaluate.")
######################################################################################################################
    print(
        f'For our analysis of the {team_name} we observe a few key results. {max_player} is the our predicted key contributors for the {team_name} the 2024\n'
        f'MLB season. Specifically {max_player} contributes a large value to the team. He leads the team in xWOBA, predicted xWOBA for 2024, exceeds the team in xSLG, and \n'
        'exceeds in fielding based upon his fielding run value. This means a variety of things. Kepler causes damage due to his xSLG being 0.086 higher\n'
        'than league average. He barrels up the ball on average far more than the rest of his team and his average ball hit is far harder than his team as well. \n'
        'He saves runs on the field where hes in the top 80 of the whole league in fielding run value. Based upon three years of historical data I found Keplers \n'
        f'xWOBA for 2024 exceeds the league average xWOBA for their predicted xWOBA by 0.086 which is significant. \033[1m{max_player} is my predicted\n'
        'key top contributor for the Twins\033[0m')
    sys.exit("Exiting the function")
######################################################################################################################
team_name = input("Enter a team to analyze: ")
generate_team_plots(team_name)

# load data
import pandas as pd
import config 

def load_data():
    xt              = pd.read_csv(config.xt)
    xt_test         = pd.read_csv(config.xt_test)
    vaep            = pd.read_csv(config.vaep)
    vaep_test       = pd.read_csv(config.vaep_test)

    games           = pd.read_csv(config.games)
    games_test      = pd.read_csv(config.games_test)
    players         = pd.read_csv(config.players)
    players_test    = pd.read_csv(config.players_test)
    target_players  = pd.read_csv(config.target_players)

    return xt, xt_test, vaep, vaep_test, games, games_test, players, players_test, target_players

# def load_data():
#     xt              = pd.read_csv('data/xt.csv')
#     xt_test         = pd.read_csv('data/xt_test.csv')
#     vaep            = pd.read_csv('data/vaep.csv')
#     vaep_test       = pd.read_csv('data/vaep_test.csv')

#     games           = pd.read_csv('data/games.csv')
#     games_test      = pd.read_csv('data/games_test.csv')
#     players         = pd.read_csv('data/players.csv')
#     players_test    = pd.read_csv('data/players_test.csv')
#     target_players  = pd.read_csv('data/target_players.csv')

#     return xt, xt_test, vaep, vaep_test, games, games_test, players, players_test, target_players
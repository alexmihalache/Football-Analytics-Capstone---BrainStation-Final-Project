def create_team_data(team_col: str, team_id: int, train_df, test_df, target: str):

    """
    Inputs: 
    - team_col:    the column to filter for team_id in
    - team_id:     the team to filter for
    - player_col:  the column to filter for player_id in - DISABLED
    - player_id:   the player to filter for - DISABLED
    - train_df:    the train dataset to filter 
    - test_df:     the test dataset to filter
    - target:      the value we want to predict, can be used for classification and regression models
    
    Returns  X_train, y_train, X_test, y_test
    """

    # get the team and player data
    team_train_set = train_df[train_df[team_col]==team_id].dropna()
    team_test_set = test_df[test_df[team_col]==team_id].dropna()
    # team_player_test_set = test_df[(test_df[team_col]==team_id)&(test_df[player_col]==player_id)].dropna()

    # get the train data
    X_train = team_train_set.drop(columns=[target])
    y_train = team_train_set[target]

    # get the test data
    X_test = team_test_set.drop(columns=[target])
    y_test = team_test_set[target]

    return X_train, y_train, X_test, y_test


def create_player_data(mode, train_df, test_df, player_id, **kwargs):
    """
    Mode: 
    "classification": 

        Returns
        X_train, y_train_action, y_train_end_zone, X_test, y_test_action, y_test_end_zone

    "regression": 

        Use the **kwargs argument to add in the regression target, either VAEP score or xT score

        Returns X_train, y_train, X_test, y_test
    """
    if mode == 'classification': 

        # create the train and test sets - remove rows where the same player performed an action as those would leak into the model

        player_train_set = train_df[
            (train_df['player_id']==player_id)
            & (train_df['n-5_same_player']!=True)
            & (train_df['n-4_same_player']!=True)
            & (train_df['n-3_same_player']!=True)
            & (train_df['n-2_same_player']!=True)
            & (train_df['n-1_same_player']!=True)
            ].dropna()


        player_test_set = test_df[
            (test_df['player_id']==player_id)
            & (test_df['n-5_same_player']!=True)
            & (test_df['n-4_same_player']!=True)
            & (test_df['n-3_same_player']!=True)
            & (test_df['n-2_same_player']!=True)
            & (test_df['n-1_same_player']!=True)
            ].dropna()

        X_train = player_train_set.drop(columns=['type_name_encoded', 'end_pitch_zone'])
        y_train_action = player_train_set['type_name_encoded']
        y_train_end_zone = player_train_set['end_pitch_zone']

        X_test = player_test_set.drop(columns=['type_name_encoded', 'end_pitch_zone'])
        y_test_action = player_test_set['type_name_encoded']
        y_test_end_zone = player_test_set['end_pitch_zone']
            
        return X_train, y_train_action, y_train_end_zone, X_test, y_test_action, y_test_end_zone
    
    if mode == 'regression': 

        reg_target = kwargs.get('reg_target', None)

        team_train_set = train_df[train_df['player_id']==player_id].dropna()
        team_test_set = test_df[test_df['player_id']==player_id].dropna()
        
        X_train = team_train_set.drop(columns=[reg_target])
        y_train = team_train_set[reg_target]

        X_test = team_test_set.drop(columns=[reg_target])
        y_test = team_test_set[reg_target]

        return X_train, y_train, X_test, y_test

def set_ct_mode(mode):
    """ 
    Modes:

    Classification:

    - player-action : for models built on a single player's data, for classification models on the next-action

    - player-end: for models built on a single player's data, for classification models on the end-zone for an action

    - team-action: for models built on a single team's data, for classification models on the next-action

    - team-end: for models built on a single team's data, for classification models on the end-zone for an action

 
    Regression:

    - player-vaep: for models built on a single player's data, for regression models on their expected vaep in a new team

    - player-xt: for models built on a single player's data, for regression models on their expected xt in a new team

    - team-vaep: for models built on a single team's data, for regression models on expected vaep

    - team-xt: for models built on a single team's data, for regression models on expected xt

    """

    if (mode == 'player-action') | (mode == 'team-action'):
        numeric_features = [
            'start_x',
            'start_y',
            'time_seconds',
            'n-1_x_distance',
            'n-1_y_distance',
            'n-1_start_x',
            'n-1_start_y',
            'n-1_end_x',
            'n-1_end_y',
            'n-1_offensive_value',
            'n-1_defensive_value',
            'n-1_vaep_value',
            'n-2_x_distance',
            'n-2_y_distance',
            'n-2_start_x',
            'n-2_start_y',
            'n-2_end_x',
            'n-2_end_y',
            'n-2_offensive_value',
            'n-2_defensive_value',
            'n-2_vaep_value',
            'n-3_x_distance',
            'n-3_y_distance',
            'n-3_start_x',
            'n-3_start_y',
            'n-3_end_x',
            'n-3_end_y',
            'n-3_offensive_value',
            'n-3_defensive_value',
            'n-3_vaep_value',
            'n-4_x_distance',
            'n-4_y_distance',
            'n-4_start_x',
            'n-4_start_y',
            'n-4_end_x',
            'n-4_end_y',
            'n-4_offensive_value',
            'n-4_defensive_value',
            'n-4_vaep_value',
            'n-5_x_distance',
            'n-5_y_distance',
            'n-5_start_x',
            'n-5_start_y',
            'n-5_end_x',
            'n-5_end_y',
            'n-5_offensive_value',
            'n-5_defensive_value',
            'n-5_vaep_value',
            ]

        categorical_features = [
            'period_id',
            'start_pitch_zone',
            'opponent_id',
            'home',
            'n-1_same_team',
            'n-1_x_fwd_direction',
            'n-1_y_lft_right_direction',
            'n-1_start_pitch_zone',
            'n-1_end_pitch_zone',
            'n-1_type_name_encoded',
            'n-1_result_name',
            'n-1_bodypart_name',
            'n-2_same_team',
            'n-2_x_fwd_direction',
            'n-2_y_lft_right_direction',
            'n-2_start_pitch_zone',
            'n-2_end_pitch_zone',
            'n-2_type_name_encoded',
            'n-2_result_name',
            'n-2_bodypart_name',
            'n-3_same_team',
            'n-3_x_fwd_direction',
            'n-3_y_lft_right_direction',
            'n-3_start_pitch_zone',
            'n-3_end_pitch_zone',
            'n-3_type_name_encoded',
            'n-3_result_name',
            'n-3_bodypart_name',
            'n-4_same_team',
            'n-4_x_fwd_direction',
            'n-4_y_lft_right_direction',
            'n-4_start_pitch_zone',
            'n-4_end_pitch_zone',
            'n-4_type_name_encoded',
            'n-4_result_name',
            'n-4_bodypart_name',
            'n-5_same_team',
            'n-5_x_fwd_direction',
            'n-5_y_lft_right_direction',
            'n-5_start_pitch_zone',
            'n-5_end_pitch_zone',
            'n-5_type_name_encoded',
            'n-5_result_name',
            'n-5_bodypart_name',
            ]
        

        # some of the features below will need dropping before training the model - but are required here for filtering dataset
        # passthrough_features = [
        #     'game_id',
        #     'player_id',
        #     ]

        drop_features = [
            'original_event_id',
            'game_id',
            'player_id',
            'team_id',
            'end_x',
            'end_y',
            'type_id',
            'result_id',
            'bodypart_id',
            'action_id',
            'type_name',
            'result_name',
            'bodypart_name',
            'offensive_value',
            'defensive_value',
            'vaep_value',
            'x_dif',
            'y_dif',
            'n-1_same_player',
            'n-2_same_player',
            'n-3_same_player',
            'n-4_same_player',
            'n-5_same_player',
            ]

        return numeric_features, categorical_features, drop_features
    
    if (mode == 'player-vaep') | (mode == 'team-vaep'):
        
        numeric_features = [
            'start_x',
            'start_y',
            'end_x',
            'end_y',
            'x_dif',
            'y_dif',
            'time_seconds',
            'n-1_x_distance',
            'n-1_y_distance',
            'n-1_start_x',
            'n-1_start_y',
            'n-1_end_x',
            'n-1_end_y',
            'n-2_x_distance',
            'n-2_y_distance',
            'n-2_start_x',
            'n-2_start_y',
            'n-2_end_x',
            'n-2_end_y',
            'n-3_x_distance',
            'n-3_y_distance',
            'n-3_start_x',
            'n-3_start_y',
            'n-3_end_x',
            'n-3_end_y',
            'n-4_x_distance',
            'n-4_y_distance',
            'n-4_start_x',
            'n-4_start_y',
            'n-4_end_x',
            'n-4_end_y',
            'n-5_x_distance',
            'n-5_y_distance',
            'n-5_start_x',
            'n-5_start_y',
            'n-5_end_x',
            'n-5_end_y',
            ]

        categorical_features = [
            'period_id',
            'start_pitch_zone',
            'end_pitch_zone',
            'opponent_id',
            'home',
            'type_name',
            'result_name',
            'bodypart_name',
            'n-1_same_team',
            'n-1_x_fwd_direction',
            'n-1_y_lft_right_direction',
            'n-1_start_pitch_zone',
            'n-1_end_pitch_zone',
            'n-1_result_name',
            'n-1_bodypart_name',
            'n-2_same_team',
            'n-2_x_fwd_direction',
            'n-2_y_lft_right_direction',
            'n-2_start_pitch_zone',
            'n-2_end_pitch_zone',
            'n-2_result_name',
            'n-2_bodypart_name',
            'n-3_same_team',
            'n-3_x_fwd_direction',
            'n-3_y_lft_right_direction',
            'n-3_start_pitch_zone',
            'n-3_end_pitch_zone',
            'n-3_result_name',
            'n-3_bodypart_name',
            'n-4_same_team',
            'n-4_x_fwd_direction',
            'n-4_y_lft_right_direction',
            'n-4_start_pitch_zone',
            'n-4_end_pitch_zone',
            'n-4_result_name',
            'n-4_bodypart_name',
            'n-5_same_team',
            'n-5_x_fwd_direction',
            'n-5_y_lft_right_direction',
            'n-5_start_pitch_zone',
            'n-5_end_pitch_zone',
            'n-5_result_name',
            'n-5_bodypart_name',
            'n-1_same_player',
            'n-2_same_player',
            'n-3_same_player',
            'n-4_same_player',
            'n-5_same_player',
            ]

        # some of the features below will need dropping before training the model - but are required here for filtering dataset
        # passthrough_features = [
        #     'vaep_value'
        #     ]

        drop_features = [
            'original_event_id',
            'game_id',
            'player_id',
            'team_id',
            'type_id',
            'result_id',
            'bodypart_id',
            'action_id',
            'type_name_encoded',
            'n-5_type_name_encoded',
            'n-4_type_name_encoded',
            'n-3_type_name_encoded',
            'n-2_type_name_encoded',
            'n-1_type_name_encoded',
            'offensive_value',
            'defensive_value',
            'n-5_offensive_value',
            'n-5_defensive_value',
            'n-5_vaep_value',
            'n-4_offensive_value',
            'n-4_defensive_value',
            'n-4_vaep_value',
            'n-3_offensive_value',
            'n-3_defensive_value',
            'n-3_vaep_value',
            'n-2_offensive_value',
            'n-2_defensive_value',
            'n-2_vaep_value',
            'n-1_offensive_value',
            'n-1_defensive_value',
            'n-1_vaep_value'
            ]
        return numeric_features, categorical_features, drop_features
    

    if (mode == 'player-end') | (mode == 'team-end'):
        numeric_features = [
            'start_x',
            'start_y',
            'time_seconds',
            'n-1_start_x',
            'n-1_start_y',
            'n-2_start_x',
            'n-2_start_y',
            'n-3_start_x',
            'n-3_start_y',
            'n-4_start_x',
            'n-4_start_y',
            'n-5_start_x',
            'n-5_start_y',
            'vaep_value',
            'offensive_value',
            'defensive_value',
            'n-5_offensive_value',
            'n-5_defensive_value',
            'n-5_vaep_value',
            'n-4_offensive_value',
            'n-4_defensive_value',
            'n-4_vaep_value',
            'n-3_offensive_value',
            'n-3_defensive_value',
            'n-3_vaep_value',
            'n-2_offensive_value',
            'n-2_defensive_value',
            'n-2_vaep_value',
            'n-1_offensive_value',
            'n-1_defensive_value',
            'n-1_vaep_value'
            ]

        categorical_features = [
            'period_id',
            'start_pitch_zone',
            'opponent_id',
            'home',
            'type_name',
            'result_name',
            'bodypart_name',
            'n-1_same_team',
            'n-1_start_pitch_zone',
            'n-1_result_name',
            'n-1_bodypart_name',
            'n-2_same_team',
            'n-2_start_pitch_zone',
            'n-2_result_name',
            'n-2_bodypart_name',
            'n-3_same_team',
            'n-3_start_pitch_zone',
            'n-3_result_name',
            'n-3_bodypart_name',
            'n-4_same_team',
            'n-4_start_pitch_zone',
            'n-4_result_name',
            'n-4_bodypart_name',
            'n-5_same_team',
            'n-5_start_pitch_zone',
            'n-5_result_name',
            'n-5_bodypart_name',
            'n-1_same_player',
            'n-2_same_player',
            'n-3_same_player',
            'n-4_same_player',
            'n-5_same_player'
            ]

        # some of the features below will need dropping before training the model - but are required here for filtering dataset
        # passthrough_features = [
        #     'vaep_value'
        #     ]

        drop_features = [
            'original_event_id',
            'game_id',
            'player_id',
            'team_id',
            'type_id',
            'result_id',
            'bodypart_id',
            'action_id',
            'n-5_type_name_encoded',
            'n-4_type_name_encoded',
            'n-3_type_name_encoded',
            'n-2_type_name_encoded',
            'n-1_type_name_encoded',
            'end_x',
            'end_y',
            'x_dif',
            'y_dif',
            'n-1_x_distance',
            'n-1_y_distance',
            'n-1_end_x',
            'n-1_end_y',
            'n-2_x_distance',
            'n-2_y_distance',
            'n-2_end_x',
            'n-2_end_y',
            'n-3_x_distance',
            'n-3_y_distance',
            'n-3_end_x',
            'n-3_end_y',
            'n-4_x_distance',
            'n-4_y_distance',
            'n-4_end_x',
            'n-4_end_y',
            'n-5_x_distance',
            'n-5_y_distance',
            'n-5_end_x',
            'n-5_end_y',
            'n-1_x_fwd_direction',
            'n-1_y_lft_right_direction',
            'n-1_end_pitch_zone',
            'n-2_x_fwd_direction',
            'n-2_y_lft_right_direction',
            'n-2_end_pitch_zone',
            'n-3_x_fwd_direction',
            'n-3_y_lft_right_direction',
            'n-3_end_pitch_zone',
            'n-4_x_fwd_direction',
            'n-4_y_lft_right_direction',
            'n-4_end_pitch_zone',
            'n-5_x_fwd_direction',
            'n-5_y_lft_right_direction',
            'n-5_end_pitch_zone'
            ]

        return numeric_features, categorical_features, drop_features
    
    if (mode == 'player-xt') | (mode == 'team-xt'):

        numeric_features = [
            'start_x',
            'start_y',
            'end_x',
            'end_y',
            'x_dif',
            'y_dif',
            'time_seconds',
            'n-1_x_distance',
            'n-1_y_distance',
            'n-1_start_x',
            'n-1_start_y',
            'n-1_end_x',
            'n-1_end_y',
            'n-2_x_distance',
            'n-2_y_distance',
            'n-2_start_x',
            'n-2_start_y',
            'n-2_end_x',
            'n-2_end_y',
            'n-3_x_distance',
            'n-3_y_distance',
            'n-3_start_x',
            'n-3_start_y',
            'n-3_end_x',
            'n-3_end_y',
            'n-4_x_distance',
            'n-4_y_distance',
            'n-4_start_x',
            'n-4_start_y',
            'n-4_end_x',
            'n-4_end_y',
            'n-5_x_distance',
            'n-5_y_distance',
            'n-5_start_x',
            'n-5_start_y',
            'n-5_end_x',
            'n-5_end_y',
            ]

        categorical_features = [
            'period_id',
            'start_pitch_zone',
            'end_pitch_zone',
            'opponent_id',
            'home',
            'type_name',
            'result_name',
            'bodypart_name',
            'n-1_same_team',
            'n-1_x_fwd_direction',
            'n-1_y_lft_right_direction',
            'n-1_start_pitch_zone',
            'n-1_end_pitch_zone',
            'n-1_result_name',
            'n-1_bodypart_name',
            'n-2_same_team',
            'n-2_x_fwd_direction',
            'n-2_y_lft_right_direction',
            'n-2_start_pitch_zone',
            'n-2_end_pitch_zone',
            'n-2_result_name',
            'n-2_bodypart_name',
            'n-3_same_team',
            'n-3_x_fwd_direction',
            'n-3_y_lft_right_direction',
            'n-3_start_pitch_zone',
            'n-3_end_pitch_zone',
            'n-3_result_name',
            'n-3_bodypart_name',
            'n-4_same_team',
            'n-4_x_fwd_direction',
            'n-4_y_lft_right_direction',
            'n-4_start_pitch_zone',
            'n-4_end_pitch_zone',
            'n-4_result_name',
            'n-4_bodypart_name',
            'n-5_same_team',
            'n-5_x_fwd_direction',
            'n-5_y_lft_right_direction',
            'n-5_start_pitch_zone',
            'n-5_end_pitch_zone',
            'n-5_result_name',
            'n-5_bodypart_name',
            'n-1_same_player',
            'n-2_same_player',
            'n-3_same_player',
            'n-4_same_player',
            'n-5_same_player',
            ]

        # some of the features below will need dropping before training the model - but are required here for filtering dataset
        # passthrough_features = [
        #     'vaep_value'
        #     ]

        drop_features = [
            'original_event_id',
            'game_id',
            'player_id',
            'team_id',
            'type_id',
            'result_id',
            'bodypart_id',
            'action_id',
            'type_name_encoded',
            'n-5_type_name_encoded',
            'n-4_type_name_encoded',
            'n-3_type_name_encoded',
            'n-2_type_name_encoded',
            'n-1_type_name_encoded',
            'n-1_xT_value',
            'n-2_xT_value',
            'n-3_xT_value',
            'n-4_xT_value',
            'n-5_xT_value'
            ]
        return numeric_features, categorical_features, drop_features

import os

# FOLDER
CONF_FOLDER = "conf"
CONF_TRAINING_FOLDER = os.path.join(CONF_FOLDER, "conf_training")
CONF_TRAINING_MODEL_FOLDER = os.path.join(CONF_TRAINING_FOLDER, "model")
CONF_TESTING_FOLDER = os.path.join(CONF_FOLDER, "conf_testing")
CONF_EVENTS_FOLDER = os.path.join(CONF_FOLDER, "conf_events")
CONF_COE_FOLDER = os.path.join(CONF_FOLDER, "conf_coe")
DATA_FOLDER = "data"
ORDERBOOK_CHANGES_FOLDER = os.path.join(DATA_FOLDER, "orderbook_changes")
TRAINED_PARAMS_FOLDER = os.path.join(DATA_FOLDER, "trained_params")
SIMULATIONS_FOLDER = os.path.join(DATA_FOLDER, "simulations")
COE_DATAFRAMES_FOLDER = os.path.join(DATA_FOLDER, "coe_dataframes")
COE_TRAINING_DATAFRAMES_FOLDER = os.path.join(COE_DATAFRAMES_FOLDER, "training_dataframes")
COE_SIMULATION_DATAFRAMES_FOLDER = os.path.join(COE_DATAFRAMES_FOLDER, "simulation_dataframes")

# Files
TRAINING_CONF_FILE= "training_conf.yml"
TESTING_CONF_FILE = "testing_conf.yml"
COE_CONF_FILE = "coe_conf.yml"
BEST_DENSITIES_FILE = 'best_densities_full.csv'
ORDERBOOK_CHANGES_FILE_PREFIX = 'orderbook_changes_'
ORDER_OF_EVENT_TYPES_FILE = 'order_of_event_type.txt'
LIKELIHOODS_FILE = 'likelihoods.txt'

# ALGORITHMS
MULTIVARIATE_HAWKES = 'multivariate_hawkes'
UNIVARIATE_HAWKES = 'univariate_hawkes'
POISSON = 'poisson'
NAIVE = 'naive'
MOVING_AVERAGE = 'moving_average'

#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],  # ASK_MARKET_ORDER_CHANGER
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],  # BID_MARKET_ORDER_CHANGER
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],  # ASK_MARKET_ORDER_NOT_CHANGER
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],  # BID_MARKET_ORDER_NOT_CHANGER
#         [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # ASK_LIMIT_ORDER_CHANGER
#         [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],  # BID_LIMIT_ORDER_CHANGER
#         [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],  # ASK_LIMIT_ORDER_NOT_CHANGER
#         [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1],  # BID_LIMIT_ORDER_NOT_CHANGER
#         [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],  # ASK_CANCELLED_ORDER_NOT_CHANGER
#         [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1],  # BID_CANCELLED_ORDER_NOT_CHANGER
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # MID_PRICE_CHANGE
#     ]
# )
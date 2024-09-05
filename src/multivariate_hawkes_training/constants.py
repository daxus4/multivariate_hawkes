import numpy as np

# FOLDER
CONF_FOLDER = "conf"
MULTIVARIATE_HAWKES_TRAINING_CONF_FILE = "multivariate_hawkes_training_conf.yml"
DATA_FOLDER = "data"
HAWKES_PARAMETERS_FOLDER = "multivariate_hawkes_trained"

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
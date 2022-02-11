import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

from load_data import load_train_data, load_test_data

logger = getLogger(__name__)

LOG_DIR = '../log/'
SAMPLE_SUBMIT_FILE = '../data/input/sample_submission.csv'
OUTPUT_DIR = '../data/output/'

if __name__ == '__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(LOG_DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df = load_train_data()

    X_train = df.drop('target', axis=1)
    y_train = df['target'].values

    use_cols = X_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))
    logger.info('train data loaded {}'.format(X_train.shape))
    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    logger.info('training finished')

    df = load_test_data()

    X_test = df[use_cols].sort_values('id')

    logger.info('test data loaded {}'.format(X_test.shape))
    pred_test = clf.predict_proba(X_test)

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['target'] = pred_test

    df_submit.to_csv(OUTPUT_DIR + 'submit.csv', index=False)
    logger.info('end')

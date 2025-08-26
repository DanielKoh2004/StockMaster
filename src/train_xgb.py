import argparse
from src.data import download_ticker
from src.labeling import create_class_labels
from src.features import add_basic_features
from src.model import train_xgb_classifier

def main(ticker, start, end, save):
    df = download_ticker(ticker, start=start, end=end)
    if df.empty:
        print('No data')
        return
    df = create_class_labels(df)
    df = add_basic_features(df)
    non_features = {'Date','next_close','ret_next','label'}
    features = [c for c in df.columns if c not in non_features and df[c].dtype != 'O']
    print('Training with features:', features)
    train_xgb_classifier(df, features, save_name=save)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', default='AAPL')
    p.add_argument('--start', default='2018-01-01')
    p.add_argument('--end', default=None)
    p.add_argument('--save', default='xgb_class_AAPL.joblib')
    args = p.parse_args()
    main(args.ticker, args.start, args.end, args.save)

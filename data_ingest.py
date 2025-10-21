import argparse, os
import pandas as pd
import yfinance as yf
from pathlib import Path


def load_universe(config):
    cfg_u = config.get('universe', {})
    if 'tickers' in cfg_u and cfg_u['tickers']:
        return cfg_u['tickers']
    csv = cfg_u.get('tickers_csv')
    if csv and os.path.exists(csv):
        return pd.read_csv(csv)['ticker'].dropna().unique().tolist()
    # fallback few
    return ['AAPL','MSFT','AMZN','GOOGL','META']


def download_prices(tickers, start, end, interval='1d'):
    data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True, threads=True)
    # yf returns multiindex columns (field, ticker) when multiple tickers
    # if multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        data = data.stack(level=1, future_stack=True) \
                .rename_axis(['date','ticker']).reset_index()
    else:
        data['ticker'] = tickers[0]
        data = data.reset_index().rename(columns={'index':'date'})

    data = data.rename(columns={
        'Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'
    })
    return data


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--interval', type=str, default='1d')
    args = parser.parse_args()

    with open('config/main.yaml', 'r') as f:
        config = yaml.safe_load(f)

    tickers = load_universe(config)
    start = args.start or config['prices']['start']
    end = args.end or config['prices']['end']
    interval = args.interval or config['prices'].get('interval','1d')

    df = download_prices(tickers, start, end, interval)

    raw_dir = Path(config['paths']['raw'])
    raw_dir.mkdir(parents=True, exist_ok=True)
    # save one parquet per ticker
    for t, g in df.groupby('ticker'):
        g.sort_values('date').to_parquet(raw_dir / f'{t}.parquet', index=False)
    print(f'Saved {len(df.ticker.unique())} tickers to {raw_dir}')

if __name__ == '__main__':
    main()
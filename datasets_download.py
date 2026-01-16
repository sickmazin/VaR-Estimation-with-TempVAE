import numpy as np
import pandas as pd
import yfinance as yf

# 0. DOWNLOAD DEI DATI
# ==========================================
# 0.1 Definizione degli assets
# SPY, GLD, ^VIX: Coprono l'intero ventennio (start 2005).
# BTC-USD: Entra nel dataset solo nel 2014.

data= {
    'tickers':[
        # --- EQUITY (AZIONARIO) ---
        'SPY',  # S&P 500 ETF: Il proxy del mercato globale. Essenziale.
        'QQQ',  # Nasdaq 100 ETF: Tech/Growth. Alta volatilità, ottimo per stress test.
        'IWM',  # Russell 2000 ETF: Small Caps. Più rischioso delle Large Caps.
        'EEM',  # Emerging Markets ETF: Esposizione a rischi geopolitici diversi.
        # --- SECTOR SPECIFIC (Per catturare shock specifici) ---
        'XLF',  # Financials ETF: Sensibile ai tassi e crisi sistemiche (2008).
        'XLE',  # Energy ETF: Cruciale per il COVID (quando il petrolio andò negativo).
        # --- SAFE HAVENS & COMMODITIES ---
        'TLT',  # 20+ Year Treasury Bond: Storicamente decorrelato (fino al 2022).
        'GLD',  # Gold ETF: Bene rifugio per eccellenza.
        # --- VOLATILITY ---
        '^VIX',  # CBOE Volatility Index: Indispensabile per modelli VaR.
    ],
    'start_date':'2004-01-01',  # Inizio ben prima della crisi 2008
    'end_date':'2025-12-31',  # Fine recente
    'train_split':0.70,  # 70% Train / 30% Test
}

paper = {
    'tickers': [
        # Selezione RIDOTTA e ROBUSTA di titoli DAX (Blue Chips sicure)
        'ALV.DE', 'BAS.DE', 'BMW.DE', 'DBK.DE', 'DTE.DE',
        'EOAN.DE', 'IFX.DE', 'MUV2.DE', 'RWE.DE', 'SAP.DE', 'SIE.DE', 'SPY', '^VIX', 'GLD',
    ],
    'start_date': '2018-10-01', # Periodo più recente e stabile
    'end_date': '2021-07-01',
}

assets = ['SPY', '^VIX', 'GLD', 'BTC-USD']
# Range di 20 anni
START = "2018-10-01"
END = "2021-07-01"

def download_data(assets, START, END, title):
    data = yf.download(assets, start=START, end=END, auto_adjust=False)['Adj Close']

    # Forward fill per gestire i giorni di chiusura mercato (es. weekend)
    data = data.ffill()

    # Analisi delle Inception Dates effettive nel range richiesto
    print("\n--- Analisi disponibilità dati (Start: 2005) ---")
    for col in data.columns:
        first_valid = data[col].first_valid_index()
        if first_valid:
            print(f"{col}: Disponibile dal {first_valid.date()}")
        else:
            print(f"{col}: Nessun dato trovato in questo intervallo.")

    # Salvataggio dei prezzi grezzi
    data.to_csv('dataset/market_data_'+title+'.csv')

    # Calcolo dei log-returns
    log_returns = (np.log(data) - np.log(data.shift(1)))

    log_returns.to_csv('dataset/log_returns_'+title+'.csv')

#download_data(assets,START,END,title='')

#download_data(data['tickers'],data['start_date'],data['end_date'],title='completo')
download_data(paper['tickers'],paper['start_date'],paper['end_date'],title='paper')


df = pd.read_csv('dataset/log_returns.csv',index_col=0,parse_dates=True)
df = df.iloc[1:]
df = df.dropna()

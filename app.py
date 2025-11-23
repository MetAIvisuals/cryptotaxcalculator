import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
import copy
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Crypto Tax Report", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADS / MONETIZATION PLACEHOLDER ---
def show_ad_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.caption("Support this Tool:")
    
    # Example: Ledger Wallet Affiliate
    # You get this link from the Ledger Affiliate portal
    ledger_html = """
    <a href="YOUR_AFFILIATE_LINK_HERE" target="_blank">
        <img src="https://www.ledger.com/wp-content/uploads/2021/11/Banner-300x250.jpg" 
             style="width:100%; border-radius:10px; border:2px solid #ddd;">
    </a>
    """
    st.sidebar.markdown(ledger_html, unsafe_allow_html=True)


# --- TRANSLATIONS ---
TRANSLATIONS = {
    "EN": {
        "title": "📊 Global Crypto Tax Calculator",
        "upload_label": "Upload Kraken Ledger CSV",
        "success_load": "Successfully processed {} transactions.",
        "err_no_trades": "No valid transactions found in this file.",
        "total_profit": "Realized Profit",
        "taxable_base": "Taxable Base",
        "tax_liab": "Est. Tax Liability",
        "tax_credit": "Tax Credit",
        "port_header_past": "Portfolio Snapshot ({})",
        "port_header_now": "Current Portfolio ({})",
        "port_cost": "Cost Basis (Invested)",
        "port_market": "Market Value (Est.)",
        "btn_live": "🔴 Update Market Prices",
        "live_success": "Market data updated.",
        "tab_years": "Tax Reports",
        "ignored_fiat": "ℹ️ Note: {} Fiat transfers (EUR/USD) detected and excluded (Tax neutral).",
        "checkbox_debug": "Show Detailed Verification Mode (Buys & Deposits)",
        "col_type": "Transaction", "col_gain": "Realized P/L",
        "chart_cost": "Allocation (By Invested Amount)",
        "chart_market": "Allocation (By Current Value)",
        "settings_header": "Tax Configuration"
    },
    "DE": {
        "title": "📊 Krypto Steuerrechner",
        "upload_label": "Kraken CSV Datei hier ablegen",
        "success_load": "{} Transaktionen erfolgreich verarbeitet.",
        "err_no_trades": "Keine gültigen Transaktionen in der Datei gefunden.",
        "total_profit": "Realisierter Gewinn",
        "taxable_base": "Steuerpflichtige Basis",
        "tax_liab": "Geschätzte Steuer",
        "tax_credit": "Steuer-Gutschrift",
        "port_header_past": "Portfolio Bestand ({})",
        "port_header_now": "Aktuelles Portfolio ({})",
        "port_cost": "Einstandswert (Investiert)",
        "port_market": "Marktwert (Geschätzt)",
        "btn_live": "🔴 Markt-Daten aktualisieren",
        "live_success": "Daten aktualisiert.",
        "tab_years": "Steuerberichte",
        "ignored_fiat": "ℹ️ Info: {} Fiat-Transfers (EUR/USD) erkannt und ausgeschlossen (Steuerneutral).",
        "checkbox_debug": "Detail-Ansicht (Käufe & Einzahlungen einblenden)",
        "col_type": "Transaktion", "col_gain": "Gewinn/Verlust",
        "chart_cost": "Verteilung (Nach Investition)",
        "chart_market": "Verteilung (Nach Marktwert)",
        "settings_header": "Steuer-Einstellungen"
    }
}

# --- SESSION STATE ---
if 'lang_index' not in st.session_state: st.session_state.lang_index = 0
if 'live_prices' not in st.session_state: st.session_state.live_prices = {}

# --- HELPER ---
def normalize_symbol(sym):
    s = str(sym).upper().strip()
    if s.endswith('.S') or s.endswith('.M'): return s[:-2] 
    if s in ['XBT', 'XXBT']: return 'BTC'
    if s in ['XETH', 'XXETH']: return 'ETH'
    if s in ['XXRP']: return 'XRP'
    if s in ['XLTC']: return 'LTC'
    return s

COIN_MAPPING = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'ADA': 'cardano', 
    'DOT': 'polkadot', 'XRP': 'ripple', 'LTC': 'litecoin', 'USDT': 'tether', 
    'USDC': 'usd-coin', 'LINK': 'chainlink', 'DOGE': 'dogecoin', 'MATIC': 'matic-network',
    'ALGO': 'algorand', 'ATOM': 'cosmos', 'MANA': 'decentraland', 'KILT': 'kilt-protocol',
    'OCEAN': 'ocean-protocol', 'NANO': 'nano', 'SHIB': 'shiba-inu', 'KEEP': 'keep-network',
    'CRV': 'curve-dao-token', 'CTSI': 'cartesi', 'KAVA': 'kava', 'SUSHI': 'sushi',
    'GRT': 'the-graph', 'MOVR': 'moonriver', 'ETHW': 'ethereum-pow', 'EIGEN': 'eigenlayer',
    'GNO': 'gnosis', 'EOS': 'eos', 'XLM': 'stellar', 'XMR': 'monero'
}

def fetch_current_prices(symbol_list):
    ids = [COIN_MAPPING.get(s) for s in symbol_list if COIN_MAPPING.get(s)]
    ids = list(set(ids)) 
    if not ids: return
    ids_str = ",".join(ids)
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/simple/price", 
                            params={'ids': ids_str, 'vs_currencies': 'eur'}, timeout=5)
        data = resp.json()
        for sym in symbol_list:
            cid = COIN_MAPPING.get(sym)
            if cid and cid in data:
                st.session_state.live_prices[sym] = data[cid]['eur']
    except:
        st.warning("Market Data API busy. Please wait 30 seconds.")

# --- PARSER ---
@st.cache_data(show_spinner=False)
def parse_csv_robust(file):
    file.seek(0)
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower().str.strip()
    
    df = df.loc[:, ~df.columns.duplicated()]
    if 'refid' not in df.columns and 'txid' in df.columns:
        df.rename(columns={'txid': 'refid'}, inplace=True)
    if 'refid' in df.columns and 'txid' in df.columns:
        df['refid'] = df['refid'].fillna(df['txid'])
    
    required = ['refid', 'amount', 'asset'] 
    if not all(c in df.columns for c in required): return pd.DataFrame(), 0
    
    def clean_num(x):
        if isinstance(x, str): x = x.replace(',', '')
        return float(x)

    try:
        df['amount'] = df['amount'].apply(clean_num).fillna(0)
        df['fee'] = df.get('fee', 0)
        if df['fee'].dtype == object: df['fee'] = df['fee'].apply(clean_num).fillna(0)
    except:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    
    df['time'] = pd.to_datetime(df['time'])
    df['asset'] = df['asset'].astype(str).str.strip().apply(normalize_symbol)
    
    clean_trades = []
    grouped = df.groupby('refid')
    FIAT = {'EUR', 'USD', 'ZEUR', 'ZUSD', 'KFEE'}
    ignored_fiat = 0
    
    for refid, group in grouped:
        try:
            agg = group.groupby('asset').agg({'amount': 'sum', 'fee': 'sum', 'time': 'first'}).reset_index()
            incoming = agg[agg['amount'] > 0]
            outgoing = agg[agg['amount'] < 0]
            date_val = agg['time'].iloc[0]
            
            # 1. TRADE / SWAP
            is_fiat_in = not incoming.empty and incoming['asset'].iloc[0] in FIAT
            is_fiat_out = not outgoing.empty and outgoing['asset'].iloc[0] in FIAT
            
            if is_fiat_out and not incoming.empty and incoming['asset'].iloc[0] not in FIAT:
                c = incoming.iloc[0]; f = outgoing.iloc[0]
                clean_trades.append({
                    'date': date_val, 'symbol': c['asset'], 'type': 'BUY',
                    'amount': c['amount'], 'price': abs(f['amount'])/c['amount'], 'fee': f['fee']+c['fee']
                })
            elif is_fiat_in and not outgoing.empty and outgoing['asset'].iloc[0] not in FIAT:
                c = outgoing.iloc[0]; f = incoming.iloc[0]
                clean_trades.append({
                    'date': date_val, 'symbol': c['asset'], 'type': 'SELL',
                    'amount': abs(c['amount']), 'price': f['amount']/abs(c['amount']), 'fee': f['fee']+c['fee']
                })
            elif not is_fiat_in and not is_fiat_out and not incoming.empty and not outgoing.empty:
                out_c = outgoing.iloc[0]; in_c = incoming.iloc[0]
                clean_trades.append({
                    'date': date_val, 'symbol': out_c['asset'], 'type': 'SELL',
                    'amount': abs(out_c['amount']), 'price': -1.0, 'fee': out_c['fee'], 'swap_ref': refid
                })
                clean_trades.append({
                    'date': date_val, 'symbol': in_c['asset'], 'type': 'BUY',
                    'amount': in_c['amount'], 'price': -1.0, 'fee': in_c['fee'], 'swap_ref': refid
                })
            # 2. WITHDRAWAL / DEPOSIT
            elif not outgoing.empty and incoming.empty:
                c = outgoing.iloc[0]
                if c['asset'] not in FIAT:
                    clean_trades.append({
                        'date': date_val, 'symbol': c['asset'], 'type': 'WITHDRAWAL',
                        'amount': abs(c['amount']), 'price': 0, 'fee': c['fee']
                    })
            elif not incoming.empty and outgoing.empty:
                c = incoming.iloc[0]
                if c['asset'] not in FIAT:
                    clean_trades.append({
                        'date': date_val, 'symbol': c['asset'], 'type': 'DEPOSIT',
                        'amount': c['amount'], 'price': 0, 'fee': c['fee']
                    })
                else:
                    ignored_fiat += 1
        except: continue
            
    return pd.DataFrame(clean_trades), ignored_fiat

# --- CALCULATOR ---
def calculate_tax(df, short_rate, long_rate, threshold, legacy_mode):
    df = df.sort_values(by=['date', 'type'], ascending=[True, False])
    portfolio = {} 
    events = []
    summary = {}
    snapshots = {} 
    swap_buffer = {}
    
    min_year = df['date'].dt.year.min()
    curr_scan_year = min_year
    
    for _, row in df.iterrows():
        trade_year = row['date'].year
        while curr_scan_year < trade_year:
            snapshots[curr_scan_year] = copy.deepcopy(portfolio)
            curr_scan_year += 1
        
        if trade_year not in summary: summary[trade_year] = {'p': 0.0, 'tb': 0.0, 'tax': 0.0}
        
        coin = row['symbol']
        typ = row['type']
        amt = float(row['amount'])
        price = float(row['price'])
        fee = float(row['fee'])
        date = row['date']
        
        if coin not in portfolio: portfolio[coin] = {'amt': 0.0, 'cost': 0.0, 'date': None}
        
        if typ in ['BUY', 'DEPOSIT']:
            val = 0.0
            if typ == 'BUY':
                if price == -1: val = swap_buffer.pop(row.get('swap_ref'), 0) 
                else: val = (amt * price) + fee
            
            curr_amt = portfolio[coin]['amt']
            curr_cost = portfolio[coin]['cost']
            new_amt = curr_amt + amt
            new_val = (curr_amt * curr_cost) + val
            
            portfolio[coin]['amt'] = new_amt
            portfolio[coin]['cost'] = new_val / new_amt if new_amt > 0 else 0
            if portfolio[coin]['date'] is None: portfolio[coin]['date'] = date
            
            events.append({
                'year': trade_year, 'date': date.date(), 'coin': coin, 'type': typ,
                'gain': 0.0, 'tax': 0.0, 'rate': 0.0, 'debug': True
            })
            
        elif typ in ['SELL', 'WITHDRAWAL']:
            curr_amt = portfolio[coin]['amt']
            cost = portfolio[coin]['cost']
            
            if typ == 'SELL':
                proceeds = 0.0
                is_swap = (price == -1)
                
                if is_swap: 
                    proceeds = amt * cost 
                    swap_buffer[row.get('swap_ref')] = proceeds
                else: 
                    proceeds = (amt * price) - fee
                
                gain = proceeds - (amt * cost)
                buy_date = portfolio[coin]['date']
                held = (date - buy_date).days if buy_date else 0
                
                is_taxable = True
                rate = 0.0
                if legacy_mode:
                    if buy_date and buy_date <= datetime(2021, 2, 28) and held > 365: is_taxable = False
                    else: rate = short_rate
                else:
                    if threshold > 0 and held > threshold: rate = long_rate
                    else: rate = short_rate
                
                tax = 0.0
                if is_taxable and not is_swap: tax = gain * rate
                else: is_taxable = False 
                
                events.append({
                    'year': trade_year, 'date': date.date(), 'coin': coin, 'type': 'SWAP' if is_swap else 'SELL',
                    'gain': gain, 'tax': tax, 'rate': rate, 'debug': False
                })
                
                summary[trade_year]['p'] += gain
                if is_taxable: summary[trade_year]['tb'] += gain
                summary[trade_year]['tax'] += tax
            
            elif typ == 'WITHDRAWAL':
                 events.append({
                    'year': trade_year, 'date': date.date(), 'coin': coin, 'type': 'WITHDRAWAL',
                    'gain': 0.0, 'tax': 0.0, 'rate': 0.0, 'debug': True
                })

            portfolio[coin]['amt'] = max(0, curr_amt - amt)
            if portfolio[coin]['amt'] < 0.00000001:
                portfolio[coin]['amt'] = 0; portfolio[coin]['cost'] = 0; portfolio[coin]['date'] = None

    snapshots[curr_scan_year] = copy.deepcopy(portfolio)
    return pd.DataFrame(events), summary, snapshots

# --- UI START ---
st.sidebar.header("Settings")
lang = st.sidebar.radio("Language", ["English", "Deutsch"], index=st.session_state.lang_index)
L = "DE" if lang == "Deutsch" else "EN"
T = TRANSLATIONS[L]

if lang == "Deutsch": st.session_state.country_choice = "Germany"
country = st.sidebar.selectbox("Country Profile", ["Austria", "Germany", "USA", "Custom"])

st.sidebar.markdown("---")
st.sidebar.subheader(T["settings_header"])
short_tax = st.sidebar.number_input("Short Term Rate (%)", value=27.5 if country == "Austria" else 42.0) / 100
long_tax = st.sidebar.number_input("Long Term Rate (%)", value=27.5 if country == "Austria" else 0.0) / 100
hold_days = st.sidebar.number_input("Hold Threshold (Days)", value=0 if country == "Austria" else 365)
legacy = st.sidebar.checkbox("AT Legacy Mode (< Feb 2021)", value=(country=="Austria"))

# --- SHOW ADS ---
st.sidebar.markdown("---")
show_ad_sidebar()

st.title(T["title"])

up_file = st.file_uploader(T["upload_label"], type="csv")

if up_file:
    df_raw, ignored_fiat = parse_csv_robust(up_file)
    if df_raw.empty:
        st.error(T["err_no_trades"])
    else:
        st.success(T["success_load"].format(len(df_raw)))
        if ignored_fiat > 0: st.info(T["ignored_fiat"].format(ignored_fiat))
        
        ev, summ, snaps = calculate_tax(df_raw, short_tax, long_tax, hold_days, legacy)
        
        # DEBUG CHECKBOX (For advanced users)
        with st.expander("🛠️ " + T["checkbox_debug"]):
            show_debug = st.checkbox("Enable", value=False)
        
        years = sorted(summ.keys(), reverse=True)
        tabs = st.tabs([str(y) for y in years])
        current_year_val = datetime.now().year
        
        for i, y in enumerate(years):
            with tabs[i]:
                s = summ[y]
                c1, c2, c3 = st.columns(3)
                c1.metric(T["total_profit"], f"€ {s['p']:,.2f}")
                c2.metric(T["taxable_base"], f"€ {s['tb']:,.2f}")
                c3.metric(T["tax_liab"], f"€ {s['tax']:,.2f}", delta_color="inverse" if s['tax']>0 else "off")
                
                if not ev.empty:
                    y_ev = ev[ev['year'] == y].copy()
                    if not show_debug: y_ev = y_ev[y_ev['debug'] == False]
                    
                    if not y_ev.empty:
                        view_df = y_ev[['date', 'coin', 'type', 'gain', 'tax', 'rate']]
                        view_df.columns = ['Date', 'Coin', T['col_type'], T['col_gain'], 'Tax', 'Rate']
                        st.dataframe(view_df.style.format({T['col_gain']: "{:.2f}", "Tax": "{:.2f}", "Rate": "{:.1%}"}), use_container_width=True)
                
                # --- PORTFOLIO ---
                st.markdown("---")
                
                is_current_year = (y == current_year_val)
                header_txt = T["port_header_now"].format(datetime.now().strftime("%d.%m.%Y")) if is_current_year else T["port_header_past"].format(y)
                
                st.subheader(header_txt)
                snap = snaps.get(y, {})
                rows = []
                syms_to_fetch = []
                total_cost = 0.0
                
                for coin, data in snap.items():
                    if data['amt'] > 0.001:
                        c_val = data['amt'] * data['cost']
                        total_cost += c_val
                        
                        row = {"Coin": coin, "Amount": data['amt'], "Cost": c_val}
                        
                        if is_current_year:
                            live = st.session_state.live_prices.get(coin, 0)
                            m_val = data['amt'] * live if live else 0
                            row["Live Price"] = live
                            row["Market Value"] = m_val
                            if m_val > 1 or c_val > 1:
                                rows.append(row)
                                if live == 0: syms_to_fetch.append(coin)
                        else:
                            if c_val > 1:
                                rows.append(row)

                if rows:
                    df_p = pd.DataFrame(rows)
                    
                    if is_current_year:
                        if st.button(T["btn_live"], key=f"btn_{y}"):
                            with st.spinner("Fetching..."):
                                fetch_current_prices(syms_to_fetch)
                            st.rerun()
                        
                        total_market = df_p['Market Value'].sum()
                        m1, m2 = st.columns(2)
                        m1.metric(T["port_cost"], f"€ {total_cost:,.2f}")
                        m2.metric(T["port_market"], f"€ {total_market:,.2f}")
                        
                        st.dataframe(df_p.style.format({"Amount": "{:.4f}", "Cost": "{:.2f}", "Live Price": "{:.2f}", "Market Value": "{:.2f}"}), use_container_width=True)
                        
                        if total_market > 0:
                            fig = px.pie(df_p, values='Market Value', names='Coin', hole=0.4, title=T["chart_market"])
                            st.plotly_chart(fig, use_container_width=True)
                        elif total_cost > 0:
                            fig = px.pie(df_p, values='Cost', names='Coin', hole=0.4, title=T["chart_cost"])
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Past Year View
                        st.metric(T["port_cost"], f"€ {total_cost:,.2f}")
                        st.dataframe(df_p.style.format({"Amount": "{:.4f}", "Cost": "{:.2f}"}), use_container_width=True)
                        
                        if total_cost > 0:
                            fig = px.pie(df_p, values='Cost', names='Coin', hole=0.4, title=T["chart_cost"])
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Empty Portfolio")
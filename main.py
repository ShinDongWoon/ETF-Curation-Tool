import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import os
from google.colab import drive
import chardet
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Union, List
import traceback
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import openai
from openai import OpenAI
from typing import Dict, Any



client=openai.OpenAI(api_key="your_api_key")
@dataclass
class StockData:
    ticker: str
    stock_info: str
    price_info: pd.DataFrame
    nh_data: pd.DataFrame
    dividend_info: pd.DataFrame

class EnhancedStockAnalyzer:
    def __init__(self):
        self.result_df = pd.DataFrame()
        drive.mount('/content/drive')
        self.file_paths = {
            'etf_holdings': '/content/NH_CONTEST_DATA_ETF_HOLDINGS.csv',
            'stock_info': '/content/NH_CONTEST_NHDATA_STK_DD_IFO.csv',
            'ticker': '/content/NH_CONTEST_STK_DT_QUT.csv',
            'company_info': '/content/NH_CONTEST_NW_FC_STK_IEM_IFO.csv',
            'etf_info': '/content/NH_CONTEST_ETF_SOR_IFO.csv',
            'dividend_data': '/content/NH_CONTEST_DATA_HISTORICAL_DIVIDEND.csv'
        }
        self.load_data()

    def load_data(self):
        self.etf_info = self._load_data(self.file_paths['company_info'])
        self.price_data = self._load_data(self.file_paths['ticker'])
        self.nh_data = self._load_data(self.file_paths['stock_info'])
        self.dividend_data = self._load_data(self.file_paths['dividend_data'])
        self.etf_holdings = self._load_data(self.file_paths['etf_holdings'])

    @staticmethod
    def _load_data(file_path: str) -> pd.DataFrame:
        with open(file_path, 'rb') as rawdata:
            result = chardet.detect(rawdata.read(100000))
            encoding_type = result['encoding']
        data = pd.read_csv(file_path, encoding=encoding_type)
        return data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    def get_stock_info(self, ticker: str) -> str:
        stock_data = self.etf_info[self.etf_info['tck_iem_cd'] == ticker]
        if stock_data.empty:
            return f"No information found for ticker: {ticker}"
        return f"""
        티커: {ticker}
        산업: {stock_data['ids_nm'].iloc[0]}
        섹터: {stock_data['ser_cfc_nm'].iloc[0]}
        업종: {stock_data['btp_cfc_nm'].iloc[0]}
        주식/ETF 구분: {stock_data['stk_etf_dit_cd'].iloc[0]}
        """

    def get_price_info(self, ticker: str) -> Union[str, pd.DataFrame]:
        price_info = self.price_data[self.price_data['tck_iem_cd'] == ticker]
        if price_info.empty:
            return f"No price information found for ticker: {ticker}"
        columns = ['bse_dt', 'iem_end_pr', 'bf_dd_cmp_ind_rt', 'acl_trd_qty']
        new_columns = ['거래일자', '종목종가', '전일대비증감율', '누적거래수량']
        return price_info[columns].rename(columns=dict(zip(columns, new_columns)))

    def get_nh_data(self, ticker: str) -> Union[str, pd.DataFrame]:
        nh_info = self.nh_data[self.nh_data['tck_iem_cd'] == ticker]
        if nh_info.empty:
            return f"No NH investment data found for ticker: {ticker}"
        columns = [
            'bse_dt', 'tco_avg_hld_wht_rt', 'tco_avg_eal_pls', 'tco_avg_phs_uit_pr', 'tco_avg_pft_rt',
            'tco_avg_hld_te_dd_cnt', 'dist_hnk_pct10_nmv', 'dist_hnk_pct30_nmv', 'dist_hnk_pct50_nmv',
            'dist_hnk_pct70_nmv', 'dist_hnk_pct90_nmv', 'bse_end_pr', 'lss_ivo_rt', 'pft_ivo_rt',
            'ifw_act_cnt', 'ofw_act_cnt', 'vw_tgt_cnt', 'rgs_tgt_cnt'
        ]
        new_columns = [
            '거래일자', '당사평균보유비중비율', '당사평균평가손익', '당사평균매입단가', '당사평균수익율', '당사평균보유기간일수',
            '분포상위10퍼센트수치', '분포상위30퍼센트수치', '분포상위50퍼센트수치', '분포상위70퍼센트수치', '분포상위90퍼센트수치',
            '기준종가', '손실투자자비율', '수익투자자비율', '신규매수계좌수', '전량매도계좌수', '종목조회건수', '관심종목등록건수'
        ]
        return nh_info[columns].rename(columns=dict(zip(columns, new_columns)))

    def get_dividend_info(self, ticker: str) -> Union[str, pd.DataFrame]:
        dividend_info = self.dividend_data[self.dividend_data['etf_tck_cd'] == ticker]
        if dividend_info.empty:
            return f"No dividend information found for ticker: {ticker}"
        columns = ['ediv_dt', 'aed_stkp_ddn_amt', 'ddn_pym_fcy_cd']
        new_columns = ['배당락일', '수정배당금', '배당주기']
        return dividend_info[columns].rename(columns=dict(zip(columns, new_columns)))

    def calculate_portfolio_metrics(self, portfolio: Dict[str, float], risk_free_rate: float = 0.037) -> pd.DataFrame:
        try:
            portfolio_returns = pd.DataFrame()
            weights = []

            for ticker, weight in portfolio.items():
                price_info = self.get_price_info(ticker)
                if isinstance(price_info, str):
                    print(f"Warning: {price_info}")
                    continue

                price_info['거래일자'] = pd.to_datetime(price_info['거래일자'], format='%Y%m%d')
                price_info.set_index('거래일자', inplace=True)
                price_info['종목종가'] = price_info['종목종가'].astype(float)
                returns = np.log(price_info['종목종가'] / price_info['종목종가'].shift(1)).dropna()
                portfolio_returns[ticker] = returns
                weights.append(float(weight))

            weights = np.array(weights, dtype=np.float64)

            if portfolio_returns.empty:
                return pd.DataFrame({'Error': ['Insufficient data to analyze portfolio.']})

            start_date = portfolio_returns.index.min().strftime('%Y-%m-%d')
            end_date = portfolio_returns.index.max().strftime('%Y-%m-%d')
            market_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close'].pct_change().dropna()

            common_dates = market_data.index.intersection(portfolio_returns.index)
            market_data = market_data.loc[common_dates]
            portfolio_returns = portfolio_returns.loc[common_dates]

            portfolio_return = np.sum(portfolio_returns.mean() * weights) * 252
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(portfolio_returns.cov() * 252, weights)))
            downside_stddev = np.sqrt(np.dot(weights.T, np.dot(portfolio_returns[portfolio_returns < 0].cov() * 252, weights)))

            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
            beta = np.cov(portfolio_returns.sum(axis=1), market_data)[0, 1] / np.var(market_data)
            treynor_ratio = (portfolio_return - risk_free_rate) / beta if beta != 0 else 0

            excess_return = portfolio_returns.sum(axis=1) - market_data
            tracking_error = excess_return.std() * np.sqrt(252)
            information_ratio = excess_return.mean() / tracking_error if tracking_error != 0 else 0

            correlation_matrix = portfolio_returns.corr()
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()

            cum_returns = (1 + portfolio_returns).cumprod(axis=0)
            max_drawdown = (cum_returns.cummax() - cum_returns).max()
            cdar_95 = max_drawdown.max()
            expected_shortfall = portfolio_returns[portfolio_returns < portfolio_returns.quantile(0.05)].mean().mean()

            portfolio_metrics = pd.DataFrame({
                'Portfolio Sharpe Ratio': [sharpe_ratio],
                'Portfolio Annual Return': [portfolio_return],
                'Portfolio Annual Std Dev': [portfolio_stddev],
                'Treynor Ratio': [treynor_ratio],
                'Information Ratio': [information_ratio],
                'CDaR (95%)': [cdar_95],
                'Expected Shortfall (5%)': [expected_shortfall],
                'Risk-free Rate': [risk_free_rate],
                'Average Correlation': [avg_correlation]
            })

            return portfolio_metrics

        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Error calculating portfolio metrics:\n{traceback_str}")
            return pd.DataFrame({'Error': [f"Error calculating portfolio metrics: {traceback_str}"]})

    def calculate_risk_metrics(self, ticker: str, risk_free_rate: float = 0.037) -> pd.DataFrame:
        price_info = self.get_price_info(ticker)
        if isinstance(price_info, str):
            return pd.DataFrame({'Error': [price_info]}, index=[ticker])

        price_info['daily_return'] = price_info['전일대비증감율'] / 100

        cvar = price_info['daily_return'].quantile(0.05)
        annual_return = price_info['daily_return'].mean() * 252
        annual_std = price_info['daily_return'].std() * np.sqrt(252)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_std

        risk_metrics = pd.DataFrame({
            'CVaR (5%)': [cvar],
            'Annual Return': [annual_return],
            'Annual Std Dev': [annual_std],
            'Sharpe Ratio': [sharpe_ratio],
        }, index=[ticker])

        return risk_metrics

    def analyze_investor_sentiment(self, nh_data: pd.DataFrame) -> str:
        if nh_data.empty:
            return "데이터가 충분하지 않습니다."

        latest_data = nh_data.iloc[-1]
        profit_ratio = latest_data['수익투자자비율']

        if profit_ratio > 70:
            return "Very Bullish"
        elif profit_ratio > 60:
            return "Bullish"
        elif profit_ratio > 40:
            return "Neutral"
        elif profit_ratio > 30:
            return "Bearish"
        else:
            return "Very Bearish"

    @staticmethod
    def plot_psychological_risk_vs_sharpe(stock_data: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=stock_data, x='Psychological Risk Index', y='Sharpe Ratio', hue='Ticker')
        plt.axvline(x=5, color='red', linestyle='--', label='x = 5')
        plt.axhline(y=0.5, color='blue', linestyle='--', label='y = 0.5')
        plt.title('Psychological Risk Index vs Sharpe Ratio (Stocks)')
        plt.xlabel('Psychological Risk Index')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_distance_from_origin_histogram(stock_data: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=stock_data, x='Distance from Origin', bins=20)
        plt.title('Distribution of Distance from Origin (Stocks)')
        plt.xlabel('Distance from Origin')
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def plot_top_bottom_stocks(stock_data: pd.DataFrame):
        top_5 = stock_data.nsmallest(5, 'Distance from Origin')
        bottom_5 = stock_data.nlargest(5, 'Distance from Origin')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        sns.barplot(data=top_5, x='Ticker', y='Distance from Origin', ax=ax1)
        ax1.set_title('Top 5 Stocks (Smallest Distance from Origin)')
        sns.barplot(data=bottom_5, x='Ticker', y='Distance from Origin', ax=ax2)
        ax2.set_title('Bottom 5 Stocks (Largest Distance from Origin)')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_portfolio_analysis(portfolio_data: pd.DataFrame):
        sharpe_ratio = portfolio_data['Sharpe Ratio'].values[0]
        info_ratio = portfolio_data['Information Ratio'].values[0]
        exp_shortfall = portfolio_data['Expected Shortfall (5%)'].values[0]

        colors = ['blue', 'red']
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)

        plt.figure(figsize=(12, 9))
        scatter = plt.scatter([sharpe_ratio], [info_ratio], c=[exp_shortfall],
                              cmap=cmap, s=200, vmin=0, vmax=max(0.1, exp_shortfall * 1.2))

        plt.xlabel('Portfolio Sharpe Ratio')
        plt.ylabel('Information Ratio')
        plt.title('Portfolio Analysis: Sharpe Ratio vs Information Ratio')

        cbar = plt.colorbar(scatter)
        cbar.set_label('Expected Shortfall (5%)', rotation=270, labelpad=15)

        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

        plt.annotate(
            f'Sharpe Ratio: {sharpe_ratio:.2f}\nInformation Ratio: {info_ratio:.2f}\nExpected Shortfall: {exp_shortfall:.4f}',
            (sharpe_ratio, info_ratio),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.tight_layout()
        plt.show()

    def analyze_stock_with_risk_metrics(self, ticker: str, user_avg_price: float, portfolio_weight: float) -> Dict[
        str, Union[str, pd.DataFrame, float]]:
        basic_data = self.analyze_stock(ticker)
        user_position = self.analyze_user_position(ticker, user_avg_price)
        psychological_risk_index = self.calculate_psychological_risk_index(ticker, user_avg_price)
        risk_metrics = self.calculate_risk_metrics(ticker)

        sharpe_ratio = risk_metrics['Sharpe Ratio'].iloc[0] if 'Sharpe Ratio' in risk_metrics.columns else None

        analysis_result = {
            "basic_data": basic_data,
            "user_position": user_position,
            "psychological_risk_index": psychological_risk_index,
            "sharpe_ratio": sharpe_ratio,
            "risk_metrics": risk_metrics
        }

        self.update_result_dataframe(ticker, analysis_result)
        return analysis_result

    def analyze_portfolio(self, portfolio: Dict[str, float], avg_prices: Dict[str, float]) -> pd.DataFrame:
        stock_data = []
        portfolio_weights = []

        for ticker, weight in portfolio.items():
            analysis_result = self.analyze_stock_with_risk_metrics(ticker, avg_prices[ticker], weight)
            stock_data.append({
                'Ticker': ticker,
                'Psychological Risk Index': analysis_result['psychological_risk_index'],
                'Sharpe Ratio': analysis_result['sharpe_ratio']
            })
            portfolio_weights.append(float(weight))

        stock_data_df = pd.DataFrame(stock_data)

        sorted_stocks = self.analyze_performance(stock_data_df)

        portfolio_metrics = self.calculate_portfolio_metrics(portfolio)

        portfolio_row = pd.DataFrame({
            'Ticker': ['PORTFOLIO'],
            'Psychological Risk Index': [None],
            'Sharpe Ratio': [portfolio_metrics['Portfolio Sharpe Ratio'].iloc[0]],
            'Distance from Origin': [None],
            'Average Correlation': [portfolio_metrics['Average Correlation'].iloc[0]],
            'Information Ratio': [portfolio_metrics['Information Ratio'].iloc[0]],
            'Expected Shortfall (5%)': [portfolio_metrics['Expected Shortfall (5%)'].iloc[0]]
        })

        result_df = pd.concat([sorted_stocks, portfolio_row], ignore_index=True)

        print("\nSorted stocks by closest distance from origin:")
        print(result_df)

        return result_df

    def analyze_stock(self, ticker: str) -> StockData:
        stock_info = self.get_stock_info(ticker)
        price_info = self.get_price_info(ticker)
        nh_data = self.get_nh_data(ticker)
        dividend_info = self.get_dividend_info(ticker)

        return StockData(
            ticker=ticker,
            stock_info=stock_info,
            price_info=price_info,
            nh_data=nh_data,
            dividend_info=dividend_info
        )

    def visualize_results(self, result_df: pd.DataFrame):
        stock_data = result_df.iloc[:-1]
        portfolio_data = result_df.iloc[-1:]

        self.plot_psychological_risk_vs_sharpe(stock_data)
        self.plot_distance_from_origin_histogram(stock_data)
        self.plot_top_bottom_stocks(stock_data)
        self.plot_portfolio_analysis(portfolio_data)

    def analyze_user_position(self, ticker: str, user_avg_price: float) -> str:
        nh_data = self.get_nh_data(ticker)
        if isinstance(nh_data, str):
            return nh_data

        latest_data = nh_data.iloc[-1]
        distribution_columns = ['분포상위10퍼센트수치', '분포상위30퍼센트수치', '분포상위50퍼센트수치', '분포상위70퍼센트수치', '분포상위90퍼센트수치']
        distribution = latest_data[distribution_columns].values

        position = np.searchsorted(distribution, user_avg_price)
        percentiles = [10, 30, 50, 70, 90]
        user_percentile = percentiles[position] if position < len(percentiles) else 100

        current_price = latest_data['기준종가']
        profit_loss = (current_price - user_avg_price) / user_avg_price * 100

        status = f"손실 중 (손실률: {profit_loss:.2f}%)" if profit_loss < 0 else f"수익 중 (수익률: {profit_loss:.2f}%)"
        return f"사용자 평균 매입가는 상위 {user_percentile}% 입니다. 현재 {status}"

    def calculate_psychological_risk_index(self, ticker: str, user_avg_price: float) -> float:
        nh_data = self.get_nh_data(ticker)
        if isinstance(nh_data, str):
            print(f"Error: {nh_data}")
            return float('nan')

        latest_data = nh_data.iloc[-1]
        distribution = [
            latest_data['분포상위10퍼센트수치'],
            latest_data['분포상위30퍼센트수치'],
            latest_data['분포상위50퍼센트수치'],
            latest_data['분포상위70퍼센트수치'],
            latest_data['분포상위90퍼센트수치']
        ]

        position = np.searchsorted(distribution, user_avg_price)

        risk_index = 10 - 2 * position if position < 5 else 1

        return round(risk_index, 2)

    def analyze_performance(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        stock_data['Distance from Origin'] = np.sqrt(
            stock_data['Sharpe Ratio'] ** 2 + stock_data['Psychological Risk Index'] ** 2)

        sorted_data = stock_data.sort_values(by='Distance from Origin', ascending=True).reset_index(drop=True)

        return sorted_data

    def update_result_dataframe(self, ticker: str, analysis_result: Dict[str, Union[str, pd.DataFrame, float]]):
        price_info = analysis_result['basic_data'].price_info
        latest_price = price_info['종목종가'].iloc[-1] if isinstance(price_info,
                                                                 pd.DataFrame) and not price_info.empty else None
        price_change = price_info['전일대비증감율'].iloc[-1] if isinstance(price_info,
                                                                    pd.DataFrame) and not price_info.empty else None

        risk_metrics = analysis_result.get('risk_metrics', pd.DataFrame())
        cvar_value = risk_metrics['CVaR (5%)'].iloc[
            0] if 'CVaR (5%)' in risk_metrics.columns and not risk_metrics.empty else None

        psychological_risk_index = analysis_result.get('psychological_risk_index', None)
        nh_data = analysis_result['basic_data'].nh_data
        investor_sentiment = self.analyze_investor_sentiment(nh_data) if isinstance(nh_data,
                                                                                    pd.DataFrame) and not nh_data.empty else None

        user_position = analysis_result.get('user_position', None)

        new_entry = pd.DataFrame({
            'Ticker': [ticker],
            'Latest Price': [latest_price],
            'Price Change (%)': [price_change],
            'CVaR (5%)': [cvar_value],
            'Psychological Risk Index': [psychological_risk_index],
            'Investor Sentiment': [investor_sentiment],
            'User Position': [user_position],
        })

        self.result_df = pd.concat([self.result_df, new_entry], ignore_index=True)

def classify_etf(etf_data: pd.DataFrame) -> pd.DataFrame:
    # 가장 최신 날짜 찾기
    latest_date = etf_data['bse_dt'].max()

    # 최신 날짜의 데이터만 필터링
    latest_etf_data = etf_data[etf_data['bse_dt'] == latest_date].copy()

    latest_etf_data.loc[:, 'profitability_score'] = latest_etf_data['ifo_rt_z_sor'] + latest_etf_data['shpr_z_sor']
    latest_etf_data.loc[:, 'risk_score'] = -latest_etf_data['crr_z_sor'] - latest_etf_data['vty_z_sor']

    conditions = [
        (latest_etf_data['profitability_score'] > 0) & (latest_etf_data['risk_score'] > 0),
        (latest_etf_data['profitability_score'] > 0) & (latest_etf_data['risk_score'] <= 0),
        (latest_etf_data['profitability_score'] <= 0) & (latest_etf_data['risk_score'] <= 0),
        (latest_etf_data['profitability_score'] <= 0) & (latest_etf_data['risk_score'] > 0)
    ]

    labels = [
        '1_quadrant_high_profit_low_risk',
        '2_quadrant_high_profit_high_risk',
        '3_quadrant_low_profit_high_risk',
        '4_quadrant_low_profit_low_risk'
    ]

    latest_etf_data.loc[:, 'classification'] = np.select(conditions, labels, default='unclassified')
    return latest_etf_data



def filter_low_performance_investments(result_df: pd.DataFrame, n: int = 3, beta_threshold: float = 1.5,
                                       pbr_threshold: float = 3, per_threshold: float = 30) -> pd.DataFrame:
    sorted_stocks = result_df[result_df['Ticker'] != 'PORTFOLIO'].sort_values('Distance from Origin')
    closest_stocks = sorted_stocks.head(n)

    low_performance_investments = []
    for _, stock in closest_stocks.iterrows():
        ticker = stock['Ticker']
        try:
            stock_info = yf.Ticker(ticker).info
            beta = stock_info.get('beta', np.nan)
            pbr = stock_info.get('priceToBook', np.nan)
            per = stock_info.get('trailingPE', np.nan)

            if ((not np.isnan(beta) and beta > beta_threshold) or
                    (not np.isnan(pbr) and pbr > pbr_threshold) or
                    (not np.isnan(per) and per > per_threshold)):
                low_performance_investments.append({
                    'Ticker': ticker,
                    'Distance from Origin': stock['Distance from Origin'],
                    'Beta': beta,
                    'P/B Ratio': pbr,
                    'P/E Ratio': per
                })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return pd.DataFrame(low_performance_investments)

def analyze_portfolio_position(result_df: pd.DataFrame) -> Dict[str, Union[int, str, float]]:
    portfolio_row = result_df[result_df['Ticker'] == 'PORTFOLIO'].iloc[0]
    ir = portfolio_row['Information Ratio']
    sr = portfolio_row['Sharpe Ratio']

    if ir > 0 and sr > 0.5:
        quadrant, profile = 1, "알파 생성자 (이상적인 전략)"
    elif ir > 0 and sr <= 0.5:
        quadrant, profile = 2, "시장 추종자 (높은 수익, 높은 위험)"
    elif ir <= 0 and sr <= 0.5:
        quadrant, profile = 3, "저성과자 (개선 필요)"
    else:
        quadrant, profile = 4, "안정 추구자 (낮은 위험, 낮은 수익)"

    plt.figure(figsize=(10, 8))
    for _, stock in result_df[result_df['Ticker'] != 'PORTFOLIO'].iterrows():
        plt.scatter(stock['Sharpe Ratio'], stock['Information Ratio'], s=50, alpha=0.7, label=stock['Ticker'])
    plt.scatter(sr, ir, s=200, c='red', marker='*', label='Portfolio')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.7)
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Information Ratio')
    plt.title('Portfolio Analysis')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return {
        'Quadrant': quadrant,
        'Investor Profile': profile,
        'Sharpe Ratio': sr,
        'Information Ratio': ir
    }

def comprehensive_portfolio_analysis(result_df: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, Dict]]:
    low_performance_investments = filter_low_performance_investments(result_df)
    portfolio_analysis = analyze_portfolio_position(result_df)

    print("저성과 투자로 판단되는 종목:")
    print(low_performance_investments)

    print("\n포트폴리오 분석 결과:")
    for key, value in portfolio_analysis.items():
        print(f"{key}: {value}")

    return {
        'Low Performance Investments': low_performance_investments,
        'Portfolio Analysis': portfolio_analysis
    }
def generate_personalized_analysis(investor_profile: Dict[str, Any], recommended_etfs: pd.DataFrame, portfolio_metrics: Dict[str, float]) -> str:
    # 분석을 위한 프롬프트 생성
    prompt = f"""
    투자자 프로필:
    - 투자자 유형: {investor_profile.get('Investor Profile', '정보 없음')}
    - Sharpe Ratio: {investor_profile.get('Sharpe Ratio', 0.0):.2f}
    - Information Ratio: {investor_profile.get('Information Ratio', 0.0):.2f}

    포트폴리오 메트릭스:
    - Sharpe Ratio: {portfolio_metrics.get('Sharpe Ratio', 0.0):.4f}
    - Information Ratio: {portfolio_metrics.get('Information Ratio', 0.0):.4f}
    - Expected Shortfall (5%): {portfolio_metrics.get('Expected Shortfall (5%)', 0.0):.4f}

    추천 ETF (상위 3개):
    {recommended_etfs[['etf_iem_cd', 'classification', 'profitability_score', 'risk_score']].head(3).to_string(index=False)}

    위 정보를 바탕으로, 투자자의 현재 포트폴리오 상태와 추천 ETF에 대한 상세한 분석과 조언을 제공해주세요.
    투자자의 프로필에 맞는 개인화된 조언을 포함하고, 각 지표가 의미하는 바를 설명해주세요.
    """

    max_retries = 5
    retry_delay = 10  # 초 단위 대기 시간

    for attempt in range(max_retries):
        try:
            # OpenAI Chat API 호출
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial advisor providing personalized investment advice."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                n=1,
                temperature=0.7,
            )

            # 응답에서 텍스트 추출
            return response.choices[0].message.content.strip()

        except openai.error.RateLimitError:
            print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    return "현재 개인화된 분석을 제공할 수 없습니다. 나중에 다시 시도해 주세요."


def main(analyzer: EnhancedStockAnalyzer) -> pd.DataFrame:
    file_path = '/content/cumulative_portfolio.csv'
    portfolio_data = pd.read_csv(file_path)

    portfolio_data = portfolio_data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    portfolio = portfolio_data.set_index('asset')['weight'].astype(float).to_dict()
    avg_prices = portfolio_data.set_index('asset')['avg_price'].astype(float).to_dict()

    sorted_stocks = analyzer.analyze_portfolio(portfolio, avg_prices)

    analyzer.result_df = sorted_stocks

    analyzer.visualize_results(analyzer.result_df)

    # 추가된 분석 함수들 실행
    analysis_results = comprehensive_portfolio_analysis(analyzer.result_df)

    print("\n포트폴리오 종합 분석 결과:")
    print(analysis_results)

    # ETF 분류 (ETF 데이터가 있다고 가정)
    etf_data = analyzer._load_data(analyzer.file_paths['etf_info'])
    classified_etfs = classify_etf(etf_data)

    print("\nETF 분류 결과:")
    print(classified_etfs['classification'].value_counts())

    return analyzer.result_df
if __name__ == "__main__":
    analyzer = EnhancedStockAnalyzer()
    result_df = main(analyzer)

    while True:
        user_input = input("조회할 투자자 ID를 입력하세요 (0-1999, 'q' 입력 시 종료): ").strip().lower()

        if user_input in ['q', 'quit', '-1']:
            print("프로그램을 종료합니다.")
            break

        try:
            investor_id = int(user_input)
            if 0 <= investor_id <= 1999:
                selected_portfolio = result_df[result_df['Ticker'] != 'PORTFOLIO']
                if not selected_portfolio.empty:
                    print(f"\n투자자 ID: {investor_id}")
                    print(f"포트폴리오 정보:")
                    print(selected_portfolio)

                    portfolio_metrics = result_df[result_df['Ticker'] == 'PORTFOLIO'].iloc[0]
                    print("\n포트폴리오 메트릭스:")
                    print(f"Sharpe Ratio: {portfolio_metrics['Sharpe Ratio']:.4f}")
                    print(f"Information Ratio: {portfolio_metrics['Information Ratio']:.4f}")
                    print(f"Expected Shortfall (5%): {portfolio_metrics['Expected Shortfall (5%)']:.4f}")

                    # ETF 추천
                    investor_profile = analyze_portfolio_position(result_df)
                    etf_data = analyzer._load_data(analyzer.file_paths['etf_info'])
                    classified_etfs = classify_etf(etf_data)

                    if investor_profile['Investor Profile'] == "알파 생성자 (이상적인 전략)":
                        recommended_etfs = classified_etfs[classified_etfs['classification'] == '1_quadrant_high_profit_low_risk']
                    elif investor_profile['Investor Profile'] == "시장 추종자 (높은 수익, 높은 위험)":
                        recommended_etfs = classified_etfs[classified_etfs['classification'].isin(['1_quadrant_high_profit_low_risk', '2_quadrant_high_profit_high_risk'])]
                    elif investor_profile['Investor Profile'] == "저성과자 (개선 필요)":
                        recommended_etfs = classified_etfs[classified_etfs['classification'] == '1_quadrant_high_profit_low_risk']
                    else:  # "안정 추구자 (낮은 위험, 낮은 수익)"
                        recommended_etfs = classified_etfs[classified_etfs['classification'].isin(['1_quadrant_high_profit_low_risk', '4_quadrant_low_profit_low_risk'])]

                    print("\n추천 ETF (최신 날짜 기준):")
                    print(recommended_etfs[['etf_iem_cd', 'classification', 'profitability_score', 'risk_score']].head())

                    personalized_analysis = generate_personalized_analysis(
                        investor_profile,
                        recommended_etfs,
                        {
                            'Sharpe Ratio': portfolio_metrics['Sharpe Ratio'],
                            'Information Ratio': portfolio_metrics['Information Ratio'],
                            'Expected Shortfall (5%)': portfolio_metrics['Expected Shortfall (5%)']
                        }
                    )

                    print("\n개인화된 포트폴리오 분석 및 조언:")
                    print(personalized_analysis)
                else:
                    print(f"투자자 ID {investor_id}에 해당하는 포트폴리오를 찾을 수 없습니다.")
            else:
                print("유효한 투자자 ID 범위는 0부터 1999까지입니다.")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")

        print("\n" + "=" * 50 + "\n")

    print("프로그램이 종료되었습니다.")
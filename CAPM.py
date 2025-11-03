import yfinance as yf
import numpy as np
import pandas as pd

# Set pandas display options for better output
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 10)


def fetch_data(tickers, start_date, end_date):
    """Fetch adjusted close prices for stocks and benchmark"""
    print("📊 Downloading data from Yahoo Finance...")
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']  # FIXED: Added space
    print(f"✅ Successfully downloaded {len(data)} days of data")
    return data


def calculate_returns(data):
    """Calculate daily log returns"""
    returns = np.log(data / data.shift(1))
    returns = returns.dropna()
    return returns


def simple_capm_beta(stock_returns, market_returns):
    """Simple CAPM beta calculation"""
    covariance = np.cov(stock_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    alpha = np.mean(stock_returns) - beta * np.mean(market_returns)

    # Calculate R-squared manually
    correlation = np.corrcoef(stock_returns, market_returns)[0, 1]
    r_squared = correlation ** 2

    return beta, alpha, r_squared


def portfolio_stats(weights, expected_returns, cov_matrix):
    """Calculate portfolio statistics"""
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio


def display_results_table(stock_tickers, betas, alphas, r_squareds, expected_returns):
    """Display results in a nice table format"""
    print("\n" + "=" * 70)
    print("CAPM BETA ANALYSIS RESULTS")
    print("=" * 70)
    print(f"{'Stock':<10} {'Beta':<8} {'Alpha':<12} {'R-squared':<10} {'Exp Return':<12} {'Volatility':<12}")
    print("-" * 70)

    for ticker in stock_tickers:
        beta = betas[ticker]
        alpha = alphas[ticker]
        r_sq = r_squareds[ticker]
        exp_ret = expected_returns[ticker]

        # Add volatility indicator
        if beta > 1.2:
            vol_indicator = "HIGH VOL"
        elif beta < 0.8:
            vol_indicator = "LOW VOL"
        else:
            vol_indicator = "MED VOL"

        print(f"{ticker:<10} {beta:<8.4f} {alpha:<12.6f} {r_sq:<10.4f} {exp_ret:<12.4f} {vol_indicator:<12}")


def main():
    print("🚀 CAPM BETA ESTIMATOR & PORTFOLIO OPTIMIZER")
    print("=============================================")

    # Configuration - CHANGE THESE STOCKS IF YOU WANT
    stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    benchmark_ticker = '^GSPC'  # S&P 500

    start_date = '2023-01-01'
    end_date = '2024-01-01'

    all_tickers = stock_tickers + [benchmark_ticker]

    try:
        # Step 1: Fetch data
        price_data = fetch_data(all_tickers, start_date, end_date)
        print(f"Downloaded data columns: {price_data.columns.tolist()}")
        print(f"First few rows:\n{price_data.head()}")

        # Step 2: Calculate returns
        returns_data = calculate_returns(price_data)
        stock_returns = returns_data[stock_tickers]
        market_returns = returns_data[benchmark_ticker]

        # Step 3: CAPM Beta Analysis
        betas = {}
        alphas = {}
        r_squareds = {}

        print("\n🔍 Calculating CAPM Betas...")
        for ticker in stock_tickers:
            beta, alpha, r_squared = simple_capm_beta(stock_returns[ticker], market_returns)
            betas[ticker] = beta
            alphas[ticker] = alpha
            r_squareds[ticker] = r_squared

        # Expected returns (annualized)
        expected_returns = stock_returns[stock_tickers].mean() * 252

        # Display CAPM results
        display_results_table(stock_tickers, betas, alphas, r_squareds, expected_returns)

        # Step 4: Portfolio Analysis
        print("\n" + "=" * 50)
        print("PORTFOLIO ANALYSIS")
        print("=" * 50)

        cov_matrix = stock_returns[stock_tickers].cov() * 252

        # Equal weight portfolio
        n_stocks = len(stock_tickers)
        equal_weights = np.array([1 / n_stocks] * n_stocks)
        equal_return, equal_vol, equal_sharpe = portfolio_stats(equal_weights, expected_returns, cov_matrix)

        print(f"📊 EQUAL WEIGHT PORTFOLIO ANALYSIS:")
        print(f"   Expected Annual Return: {equal_return:.2%}")
        print(f"   Expected Volatility: {equal_vol:.2%}")
        print(f"   Sharpe Ratio: {equal_sharpe:.4f}")
        print("\n   Portfolio Weights:")
        for ticker, weight in zip(stock_tickers, equal_weights):
            print(f"     {ticker}: {weight:.2%}")

        # Step 5: Investment Insights
        print("\n" + "=" * 50)
        print("💡 INVESTMENT INSIGHTS")
        print("=" * 50)

        highest_beta = max(betas.items(), key=lambda x: x[1])
        lowest_beta = min(betas.items(), key=lambda x: x[1])
        highest_return = max(expected_returns.items(), key=lambda x: x[1])
        lowest_return = min(expected_returns.items(), key=lambda x: x[1])

        print(f"📈 Most aggressive stock: {highest_beta[0]} (Beta: {highest_beta[1]:.4f})")
        print(f"🛡️  Most defensive stock: {lowest_beta[0]} (Beta: {lowest_beta[1]:.4f})")
        print(f"💰 Highest expected return: {highest_return[0]} ({highest_return[1]:.2%})")
        print(f"📉 Lowest expected return: {lowest_return[0]} ({lowest_return[1]:.2%})")

        # Risk assessment
        high_beta_stocks = [ticker for ticker, beta in betas.items() if beta > 1.2]
        low_beta_stocks = [ticker for ticker, beta in betas.items() if beta < 0.8]

        if high_beta_stocks:
            print(f"⚡ High volatility stocks (Beta > 1.2): {', '.join(high_beta_stocks)}")
        if low_beta_stocks:
            print(f"🛡️  Low volatility stocks (Beta < 0.8): {', '.join(low_beta_stocks)}")

        # Portfolio recommendation
        avg_beta = np.mean(list(betas.values()))
        if avg_beta > 1.1:
            print(f"\n🎯 Overall Portfolio: AGGRESSIVE (Avg Beta: {avg_beta:.3f})")
        elif avg_beta < 0.9:
            print(f"\n🎯 Overall Portfolio: DEFENSIVE (Avg Beta: {avg_beta:.3f})")
        else:
            print(f"\n🎯 Overall Portfolio: MODERATE (Avg Beta: {avg_beta:.3f})")

        print("\n✅ Analysis complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("This might be due to:")
        print("1. Internet connection issues")
        print("2. Invalid stock tickers")
        print("3. Yahoo Finance API temporary issues")


if __name__ == "__main__":
    main()
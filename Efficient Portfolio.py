import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import requests
from bs4 import BeautifulSoup

# URL for retrieving the 10-year Treasury constant maturity rate (risk-free rate)
url = 'https://fred.stlouisfed.org/series/DGS10'

# Send a GET request to the webpage and parse the HTML content
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract the risk-free rate by locating the relevant span element
div = soup.find('div', class_='float-start meta-col col-sm-5 col-5')
span = div.find('span', class_='series-meta-observation-value')

# Convert the extracted text to a float and adjust for percentage
risk_free_rate = float(span.text.strip()) / 100

# Display the current risk-free rate
print(f"Current Risk-Free Rate: {risk_free_rate:.4f} (retrieved from: {url})\n")

# User input for short selling permission and portfolio weight bounds
Shorting = input("Allow Short Selling? ('yes' or 'no'): ")
up = float(input("Enter upper bound: "))
down = float(input("Enter lower bound: "))

# Display available benchmark options and get user input for selection
benchmark_tickers = pd.DataFrame({
    "Number": [1, 2, 3, 4, 5, 6],
    "Benchmark": ["S&P 500", "Russell 1000", "Russell 2000", "Dow Jones Industrial Average", "NASDAQ Composite", "Russell 3000"],
    "Ticker": ["^GSPC", "^RUI", "^RUT", "^DJI", "^IXIC", "^RUA"]})
print(benchmark_tickers)
benchmark_choice = int(input("Enter a benchmark ticker number: "))

# Constants for portfolio analysis
RISK_FREE_RATE = risk_free_rate
TICKERS = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'UNH', 'LLY']
BENCHMARK = benchmark_tickers.iloc[benchmark_choice - 1, 2]
START_DATE = '2019-07-01'
END_DATE = '2024-08-01'

# Set portfolio weight bounds based on short selling permission
ALLOW_SHORT_SELLING = True if Shorting.lower() == "yes" else False
upper_bound = up  # Max upper limit for weights
lower_bound = down  # Max lower limit for weights
lower_bound = lower_bound if ALLOW_SHORT_SELLING else max(0.0, lower_bound)

def download_stock_data(tickers, start_date, end_date):
    """
    Downloads stock data for the given tickers and date range.
    Returns the monthly percentage returns.
    """
    data = yf.download(tickers, start=start_date, end=end_date, interval='1mo')['Adj Close']
    returns = data.pct_change().dropna()
    return returns

def calculate_portfolio_stats(weights, returns):
    """
    Calculates the portfolio's expected return and volatility.
    """
    portfolio_return = np.sum(returns.mean() * weights) * 12
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 12, weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, returns, risk_free_rate):
    """
    Calculates the negative Sharpe ratio (for minimization).
    """
    portfolio_return, portfolio_volatility = calculate_portfolio_stats(weights, returns)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

def optimize_portfolio(returns, risk_free_rate, target_function, bounds):
    """
    Optimizes the portfolio using the given target function.
    """
    num_assets = len(returns.columns)
    args = (returns, risk_free_rate)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights must sum to 1
    
    result = minimize(target_function, num_assets * [1. / num_assets], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

def get_efficient_frontier(returns, risk_free_rate, num_portfolios=100, bounds=None):
    """
    Calculates the Global Minimum Variance Portfolio (GMVP), Tangent Portfolio, 
    and the efficient frontier.
    """
    if bounds is None:
        bounds = tuple((0, 1) for _ in range(len(returns.columns)))

    # Global Minimum Variance Portfolio (GMVP)
    gmvp_weights = optimize_portfolio(returns, risk_free_rate, 
                                      lambda weights, returns, _: calculate_portfolio_stats(weights, returns)[1],
                                      bounds)
    gmvp_return, gmvp_volatility = calculate_portfolio_stats(gmvp_weights, returns)

    # Tangent Portfolio (maximizing Sharpe ratio)
    tangent_weights = optimize_portfolio(returns, risk_free_rate, negative_sharpe_ratio, bounds)
    tangent_return, tangent_volatility = calculate_portfolio_stats(tangent_weights, returns)

    # Generate Efficient Frontier
    target_returns = np.linspace(gmvp_return, tangent_return * 1.5, num_portfolios)
    efficient_portfolios = []

    for target_return in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights must sum to 1
            {'type': 'eq', 'fun': lambda x: calculate_portfolio_stats(x, returns)[0] - target_return}
        )
        result = minimize(lambda x: calculate_portfolio_stats(x, returns)[1], 
                          len(returns.columns) * [1. / len(returns.columns)],
                          method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            efficient_portfolios.append(result.x)

    returns_array = [calculate_portfolio_stats(weights, returns)[0] for weights in efficient_portfolios]
    volatilities_array = [calculate_portfolio_stats(weights, returns)[1] for weights in efficient_portfolios]

    return (gmvp_return, gmvp_volatility, gmvp_weights), (tangent_return, tangent_volatility, tangent_weights), (returns_array, volatilities_array)

def plot_efficient_frontier(returns, risk_free_rate, bounds=None):
    """
    Plots the efficient frontier along with the Global Minimum Variance Portfolio (GMVP),
    Tangent Portfolio, and Capital Market Line (CML).
    """
    gmvp, tangent, efficient_frontier = get_efficient_frontier(returns, risk_free_rate, bounds=bounds)
    gmvp_return, gmvp_volatility, _ = gmvp
    tangent_return, tangent_volatility, _ = tangent
    returns_array, volatilities_array = efficient_frontier

    plt.figure(figsize=(10, 6))
    plt.plot(volatilities_array, returns_array, 'b-', label='Efficient Frontier')
    plt.scatter(gmvp_volatility, gmvp_return, color='red', marker='*', s=200, label='Global Minimum Variance Portfolio')
    plt.scatter(tangent_volatility, tangent_return, color='green', marker='*', s=200, label='Tangent Portfolio')

    # Plot Capital Market Line (CML)
    max_volatility = max(volatilities_array) * 1.2
    cml_x = np.linspace(0, max_volatility, 100)
    cml_y = risk_free_rate + (tangent_return - risk_free_rate) / tangent_volatility * cml_x
    plt.plot(cml_x, cml_y, 'r--', label='Capital Market Line')

    # Plot individual assets
    for i, ticker in enumerate(returns.columns):
        asset_return = returns[ticker].mean() * 12
        asset_volatility = returns[ticker].std() * np.sqrt(12)
        plt.scatter(asset_volatility, asset_return, marker='o', label=ticker)

    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier with GMVP, Tangent Portfolio, and CML')
    plt.legend()
    plt.grid(True)
    plt.savefig('efficient_frontier.png')
    plt.show()

    # Print GMVP and Tangent Portfolio details
    print(f"\nGlobal Minimum Variance Portfolio:")
    print(f"Return: {gmvp_return:.4f}")
    print(f"Volatility: {gmvp_volatility:.4f}")
    print("Weights:")
    for ticker, weight in zip(returns.columns, gmvp[2]):
        print(f"  {ticker}: {weight:.4f}")

    print(f"\nTangent Portfolio:")
    print(f"Return: {tangent_return:.4f}")
    print(f"Volatility: {tangent_volatility:.4f}")
    print("Weights:")
    for ticker, weight in zip(returns.columns, tangent[2]):
        print(f"  {ticker}: {weight:.4f}")
    
    # Save GMVP and Tangent Portfolio stats to file
    with open('gmvp_tangent_stats.txt', 'w') as f:
        f.write(f"\nGlobal Minimum Variance Portfolio:\n")
        f.write(f"Return: {gmvp_return:.4f}\n")
        f.write(f"Volatility: {gmvp_volatility:.4f}\n")
        f.write("Weights:\n")
        for ticker, weight in zip(returns.columns, gmvp[2]):
            f.write(f"  {ticker}: {weight:.4f}\n")

        f.write(f"\nTangent Portfolio:\n")
        f.write(f"Return: {tangent_return:.4f}\n")
        f.write(f"Volatility: {tangent_volatility:.4f}\n")
        f.write("Weights:\n")
        for ticker, weight in zip(returns.columns, tangent[2]):
            f.write(f"  {ticker}: {weight:.4f}\n")

    return gmvp, tangent, efficient_frontier

def cumulative_returns_versus_benchmark(returns, benchmark, gmvp, tangent):
    """
    Plots cumulative returns of GMVP and Tangent Portfolio versus the benchmark.
    Also calculates the allocation for a $10 million investment.
    """
    gmvp_weights = gmvp[2]
    tangent_weights = tangent[2]
    
    # Ensure returns and benchmark use the same date range
    common_dates = returns.index.intersection(benchmark.index)
    returns = returns.loc[common_dates]
    benchmark = benchmark.loc[common_dates]
    
    # Calculate portfolio returns
    gmvp_returns = (returns * gmvp_weights).sum(axis=1)
    tangent_returns = (returns * tangent_weights).sum(axis=1)
    
    # Calculate cumulative returns
    cumulative_gmvp = (1 + gmvp_returns).cumprod()
    cumulative_tangent = (1 + tangent_returns).cumprod()
    cumulative_benchmark = (1 + benchmark).cumprod()
    
    # Create a DataFrame for plotting
    cumulative_returns = pd.DataFrame({
        'GMVP': cumulative_gmvp,
        'Tangent Portfolio': cumulative_tangent,
        'Benchmark': cumulative_benchmark.squeeze()  # Convert DataFrame with one column to Series
    })

    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns.index, cumulative_returns['GMVP'], label='GMVP', color='red')
    plt.plot(cumulative_returns.index, cumulative_returns['Tangent Portfolio'], label='Tangent Portfolio', color='orange')
    plt.plot(cumulative_returns.index, cumulative_returns['Benchmark'], label='Benchmark', linestyle='--', color='green')
    plt.title('Cumulative Returns: GMVP and Tangent Portfolio vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_returns.png')
    plt.show()
    
    # Calculate allocation for a $10 million investment
    initial_investment = 10_000_000
    gmvp_allocation = gmvp_weights * initial_investment
    tangent_allocation = tangent_weights * initial_investment
    
    # Create DataFrames for allocation
    gmvp_allocation_df = pd.DataFrame({'Stock': returns.columns, 'Allocation': gmvp_allocation})
    tangent_allocation_df = pd.DataFrame({'Stock': returns.columns, 'Allocation': tangent_allocation})
    
    # Sort allocations in descending order and format as currency
    gmvp_allocation_df = gmvp_allocation_df.sort_values('Allocation', ascending=False)
    gmvp_allocation_df['Allocation'] = gmvp_allocation_df['Allocation'].apply(lambda x: f"${x:,.2f}")
    tangent_allocation_df = tangent_allocation_df.sort_values('Allocation', ascending=False)
    tangent_allocation_df['Allocation'] = tangent_allocation_df['Allocation'].apply(lambda x: f"${x:,.2f}")
    
    # Print allocation details
    print("\nGMVP Allocation:")
    print(gmvp_allocation_df)
    print("\nTangent Portfolio Allocation:")
    print(tangent_allocation_df)

    # Save cumulative returns
    cumulative_returns.to_csv('cumulative_returns.csv')
    
    # Save GMVP and Tangent Portfolio allocation details to file
    with open('gmvp_tangent_stats.txt', 'a') as f:
        f.write("\nGMVP Allocation:\n")
        f.write(gmvp_allocation_df.to_string(index=False))
        f.write("\n\nTangent Portfolio Allocation:\n")
        f.write(tangent_allocation_df.to_string(index=False))
    
    return cumulative_returns, gmvp_allocation_df, tangent_allocation_df

# Main execution flow
returns = download_stock_data(TICKERS, START_DATE, END_DATE)
benchmark = download_stock_data(BENCHMARK, START_DATE, END_DATE)

# Define bounds for the optimization process
bounds = tuple((lower_bound, upper_bound) for _ in range(len(TICKERS)))

# Plot the efficient frontier and calculate portfolio statistics
gmvp, tangent, efficient_frontier = plot_efficient_frontier(returns, RISK_FREE_RATE, bounds)

# Compare cumulative returns with the benchmark
cumulative_returns_versus_benchmark(returns, benchmark, gmvp, tangent)

# Save stock and benchmark returns to CSV files
returns.to_csv('stock_returns.csv')
benchmark.to_csv('benchmark_returns.csv')

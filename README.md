# Efficient-Frontier
Efficient frontier portfolio optimization tool with visualization for S&amp;P 500 stocks

# Portfolio Optimization Tool

## Overview
This project implements an efficient frontier portfolio optimization tool for a selection of S&P 500 stocks. It allows users to analyze different investment scenarios, construct optimal portfolios, and visualize the results.

## Features
- Download historical stock data using yfinance
- Calculate portfolio statistics (return and volatility)
- Generate the efficient frontier
- Identify the Global Minimum Variance Portfolio (GMVP) and Tangent Portfolio
- Visualize the efficient frontier and portfolio performance
- Compare portfolio performance against a benchmark (S&P 500)
- Analyze different investment scenarios with various constraints

## Requirements
- Python 3.7+
- Libraries: numpy, pandas, yfinance, matplotlib, scipy, requests, beautifulsoup4

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/portfolio-optimization-tool.git
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the main script:
   ```
   python efficient_portfolio.py
   ```
2. Follow the prompts to input your preferences:
   - Allow short selling (yes/no)
   - Upper bound for weights
   - Lower bound for weights
   - Select a benchmark index

3. The script will generate:
   - Efficient frontier plot
   - Cumulative returns plot
   - Portfolio statistics and allocations

## Scenarios
The tool can analyze three main scenarios:
1. No constraints (unrestricted weights, with short selling)
2. No short selling (all weights non-negative)
3. Weight constraints (5% minimum, 60% maximum per stock)

## Output
- Efficient frontier graph (`efficient_frontier.png`)
- Cumulative returns graph (`cumulative_returns.png`)
- CSV files with stock and benchmark returns
- Text file with GMVP and Tangent Portfolio statistics

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.

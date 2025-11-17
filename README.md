# Euribor Future Options Pricer

A comprehensive interactive dashboard for pricing Euribor Future Options (RFOs) with support for both Mid-Curve and Non-Mid-Curve cases, featuring SABR-M volatility modeling.

## Features

### Core Pricing Capabilities
- **Dual Option Types**: Support for both Mid-Curve and Non-Mid-Curve Euribor Future Options
- **SABR-M Model**: Advanced volatility modeling with full parameter calibration
- **Black-76 Pricing**: Industry-standard pricing model for options on futures
- **Complete Greeks**: Delta, Gamma, Vega, Theta, and Rho calculations

### Interactive Dashboard Components
1. **Pricing Tab**: Real-time option pricing with scenario analysis
2. **Volatility Surface**: 3D visualization of implied volatility surface and smile curves
3. **Payoff Analysis**: Interactive P&L diagrams with position management
4. **Greeks Analysis**: Comprehensive sensitivity analysis across parameters
5. **Theory Documentation**: In-app theoretical overview and methodology

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone or download the project files
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## Usage Guide

### Market Data Input
- **Future Price**: Current Euribor future price (quoted as 100 - rate)
- **Risk-Free Rate**: Current risk-free interest rate for discounting

### Option Parameters
- **Option Type**: Choose between Call and Put options
- **Strike Price**: The strike price of the option
- **Option Style**: 
  - Mid-Curve: Option expires before the underlying future
  - Non-Mid-Curve: Option and future expire simultaneously
- **Time to Expiry**: Time until option expiration (in years)
- **Time to Future Expiry**: (Mid-Curve only) Time until future expiration

### SABR Model Parameters
- **α (Alpha)**: Initial volatility level
- **β (Beta)**: CEV exponent (elasticity parameter)
- **ρ (Rho)**: Correlation between asset and volatility
- **ν (Nu)**: Volatility of volatility

### Key Features Explained

#### Scenario Analysis
The pricer automatically calculates option values under different market scenarios:
- Bear market (-10%)
- Mild bear (-5%)
- Base case
- Mild bull (+5%)
- Bull market (+10%)

#### Volatility Surface Visualization
- 3D surface plot showing implied volatility across strikes and maturities
- 2D volatility smile for selected maturities
- Real-time updates based on SABR parameters

#### Position Management
- Support for long and short positions
- Customizable number of contracts and contract size
- Real-time P&L calculations

#### Greeks Dashboard
- Visual representation of all option Greeks
- Sensitivity analysis across different strikes and time to expiry
- Interactive charts showing Greek behaviors

## Technical Architecture

### Core Components

1. **SABRModel Class**
   - Implements SABR volatility model
   - Provides implied volatility calculations
   - Includes calibration functionality

2. **EuriborFutureOption Class**
   - Implements Black-76 pricing formula
   - Separate methods for Mid-Curve and Non-Mid-Curve options
   - Complete Greeks calculations

3. **Visualization Functions**
   - 3D volatility surface plotting
   - Option payoff diagrams
   - Greeks sensitivity charts

### Mathematical Framework

The implementation uses:
- **Black-76 Model**: Standard model for European options on futures
- **SABR Model**: Stochastic volatility model for smile modeling
- **Risk-Neutral Valuation**: Ensures arbitrage-free pricing

## Project Structure

```
euribor-future-options-pricer/
│
├── streamlit_app.py       # Main application file
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Extending the Project

This pricer can be extended to include:

### Additional Features
- American option pricing using binomial trees
- Monte Carlo simulation for path-dependent options
- Historical volatility analysis
- Market data feeds integration
- Risk metrics (VaR, CVaR)

### Model Enhancements
- Alternative volatility models (Heston, GARCH)
- Multi-curve framework
- Credit spread adjustments
- Collateral posting considerations

### UI Improvements
- Real-time market data integration
- Portfolio management features
- Batch pricing capabilities
- Export functionality for results

## Performance Considerations

- The SABR model uses analytical approximations for speed
- Vectorized NumPy operations for efficient calculations
- Plotly for interactive but performant visualizations
- Streamlit caching can be added for frequently computed values

## Dependencies

- **streamlit**: Web application framework
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scipy**: Statistical functions and optimization
- **plotly**: Interactive visualizations

## References

1. Black, F. (1976). "The pricing of commodity contracts"
2. Hagan, P. et al. (2002). "Managing Smile Risk"
3. CME Group. "Eurodollar Options Handbook"
4. Rebonato, R. (2004). "Volatility and Correlation"

## License

This project is provided as-is for educational and professional use.

## Support

For questions or issues, please refer to the in-app documentation or review the theoretical overview in the Theory & Documentation tab.

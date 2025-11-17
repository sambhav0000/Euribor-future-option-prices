"""
Euribor Future Options Pricer
Interactive Dashboard for pricing Euribor Future Options (RFOs)
Supports both Mid-Curve and Non-Mid-Curve cases with SABR-M volatility model
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Euribor Future Options Pricer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class SABRModel:
    """SABR-M (Modified SABR) Model for volatility modeling"""
    
    def __init__(self, alpha: float = 0.05, beta: float = 0.5, 
                 rho: float = -0.3, nu: float = 0.3):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
    
    def implied_vol(self, F: float, K: float, T: float) -> float:
        """Calculate SABR implied volatility"""
        if F == K:
            # ATM case
            v_atm = (self.alpha / (F ** (1 - self.beta))) * \
                    (1 + ((1 - self.beta) ** 2 / 24 * self.alpha ** 2 / (F ** (2 - 2 * self.beta)) +
                          0.25 * self.rho * self.beta * self.nu * self.alpha / (F ** (1 - self.beta)) +
                          (2 - 3 * self.rho ** 2) / 24 * self.nu ** 2) * T)
            return v_atm
        
        # General case
        z = (self.nu / self.alpha) * (F * K) ** ((1 - self.beta) / 2) * np.log(F / K)
        x_z = np.log((np.sqrt(1 - 2 * self.rho * z + z ** 2) + z - self.rho) / (1 - self.rho))
        
        A = self.alpha / ((F * K) ** ((1 - self.beta) / 2) * 
                         (1 + (1 - self.beta) ** 2 / 24 * (np.log(F / K)) ** 2 +
                          (1 - self.beta) ** 4 / 1920 * (np.log(F / K)) ** 4))
        
        B = 1 + ((1 - self.beta) ** 2 / 24 * self.alpha ** 2 / ((F * K) ** (1 - self.beta)) +
                0.25 * self.rho * self.beta * self.nu * self.alpha / ((F * K) ** ((1 - self.beta) / 2)) +
                (2 - 3 * self.rho ** 2) / 24 * self.nu ** 2) * T
        
        if abs(z) < 1e-10:
            return A * B
        
        return A * (z / x_z) * B
    
    def calibrate(self, market_vols: np.ndarray, forwards: np.ndarray, 
                  strikes: np.ndarray, maturities: np.ndarray) -> None:
        """Calibrate SABR parameters to market data"""
        def objective(params):
            self.alpha, self.beta, self.rho, self.nu = params
            model_vols = []
            for F, K, T in zip(forwards, strikes, maturities):
                try:
                    vol = self.implied_vol(F, K, T)
                    model_vols.append(vol)
                except:
                    model_vols.append(0.2)  # Default vol if calculation fails
            
            model_vols = np.array(model_vols)
            return np.sum((model_vols - market_vols) ** 2)
        
        # Constraints for SABR parameters
        bounds = [(0.001, 1.0),  # alpha
                  (0.0, 1.0),     # beta
                  (-0.999, 0.999), # rho
                  (0.001, 2.0)]   # nu
        
        initial_guess = [self.alpha, self.beta, self.rho, self.nu]
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            self.alpha, self.beta, self.rho, self.nu = result.x

class EuriborFutureOption:
    """Pricer for Euribor Future Options"""
    
    def __init__(self, sabr_model: SABRModel):
        self.sabr_model = sabr_model
    
    def black_formula(self, F: float, K: float, T: float, sigma: float, 
                      r: float, option_type: str = 'call') -> float:
        """Black formula for European options on futures"""
        if T <= 0:
            if option_type.lower() == 'call':
                return max(F - K, 0)
            else:
                return max(K - F, 0)
        
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            return np.exp(-r * T) * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
        else:
            return np.exp(-r * T) * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))
    
    def price_mid_curve_option(self, future_price: float, strike: float, 
                               time_to_expiry: float, time_to_future_expiry: float,
                               risk_free_rate: float, option_type: str = 'call') -> Dict:
        """Price Mid-Curve Euribor Future Option"""
        # Adjust forward price for mid-curve
        forward_adjustment = time_to_future_expiry - time_to_expiry
        adjusted_forward = future_price * np.exp(-risk_free_rate * forward_adjustment)
        
        # Get implied volatility from SABR model
        sigma = self.sabr_model.implied_vol(adjusted_forward, strike, time_to_expiry)
        
        # Calculate option price using Black formula
        price = self.black_formula(adjusted_forward, strike, time_to_expiry, 
                                  sigma, risk_free_rate, option_type)
        
        # Calculate Greeks
        greeks = self.calculate_greeks(adjusted_forward, strike, time_to_expiry, 
                                       sigma, risk_free_rate, option_type)
        
        return {
            'price': price,
            'implied_vol': sigma,
            'forward': adjusted_forward,
            **greeks
        }
    
    def price_non_mid_curve_option(self, future_price: float, strike: float, 
                                   time_to_expiry: float, risk_free_rate: float,
                                   option_type: str = 'call') -> Dict:
        """Price Non-Mid-Curve Euribor Future Option"""
        # Get implied volatility from SABR model
        sigma = self.sabr_model.implied_vol(future_price, strike, time_to_expiry)
        
        # Calculate option price using Black formula
        price = self.black_formula(future_price, strike, time_to_expiry, 
                                  sigma, risk_free_rate, option_type)
        
        # Calculate Greeks
        greeks = self.calculate_greeks(future_price, strike, time_to_expiry, 
                                       sigma, risk_free_rate, option_type)
        
        return {
            'price': price,
            'implied_vol': sigma,
            'forward': future_price,
            **greeks
        }
    
    def calculate_greeks(self, F: float, K: float, T: float, sigma: float, 
                        r: float, option_type: str) -> Dict:
        """Calculate option Greeks"""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
        
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            delta = np.exp(-r * T) * stats.norm.cdf(d1)
            theta = -np.exp(-r * T) * (F * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                                       r * K * stats.norm.cdf(d2))
        else:
            delta = -np.exp(-r * T) * stats.norm.cdf(-d1)
            theta = -np.exp(-r * T) * (F * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                                       r * K * stats.norm.cdf(-d2))
        
        gamma = np.exp(-r * T) * stats.norm.pdf(d1) / (F * sigma * np.sqrt(T))
        vega = F * np.exp(-r * T) * stats.norm.pdf(d1) * np.sqrt(T) / 100
        rho = K * T * np.exp(-r * T) * (stats.norm.cdf(d2) if option_type.lower() == 'call' 
                                        else -stats.norm.cdf(-d2)) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta / 365,  # Convert to daily theta
            'rho': rho
        }

def generate_volatility_surface(sabr_model: SABRModel, spot: float, 
                               strikes_range: Tuple[float, float], 
                               maturities: List[float]) -> pd.DataFrame:
    """Generate volatility surface using SABR model"""
    strikes = np.linspace(strikes_range[0] * spot, strikes_range[1] * spot, 20)
    
    surface_data = []
    for T in maturities:
        for K in strikes:
            vol = sabr_model.implied_vol(spot, K, T)
            surface_data.append({
                'Strike': K,
                'Maturity': T,
                'Implied_Vol': vol,
                'Moneyness': K / spot
            })
    
    return pd.DataFrame(surface_data)

def plot_volatility_surface(surface_df: pd.DataFrame):
    """Create 3D volatility surface plot"""
    pivot_data = surface_df.pivot(index='Strike', columns='Maturity', values='Implied_Vol')
    
    fig = go.Figure(data=[go.Surface(
        x=pivot_data.columns,
        y=pivot_data.index,
        z=pivot_data.values,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Implied Vol")
    )])
    
    fig.update_layout(
        title="SABR Implied Volatility Surface",
        scene=dict(
            xaxis_title="Time to Maturity (Years)",
            yaxis_title="Strike",
            zaxis_title="Implied Volatility",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=600
    )
    
    return fig

def plot_option_payoff(spot: float, strike: float, premium: float, 
                       option_type: str, is_long: bool = True):
    """Plot option payoff diagram"""
    spot_range = np.linspace(0.7 * strike, 1.3 * strike, 100)
    
    if option_type.lower() == 'call':
        payoff = np.maximum(spot_range - strike, 0)
    else:
        payoff = np.maximum(strike - spot_range, 0)
    
    if not is_long:
        payoff = -payoff
        premium = -premium
    
    profit = payoff - premium
    
    fig = go.Figure()
    
    # Add payoff line
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=profit,
        mode='lines',
        name='P&L',
        line=dict(color='blue', width=2)
    ))
    
    # Add break-even line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Break-even")
    
    # Add current spot marker
    fig.add_vline(x=spot, line_dash="dash", line_color="green", 
                  annotation_text=f"Current Spot: {spot:.4f}")
    
    # Add strike marker
    fig.add_vline(x=strike, line_dash="dash", line_color="red", 
                  annotation_text=f"Strike: {strike:.4f}")
    
    fig.update_layout(
        title=f"{'Long' if is_long else 'Short'} {option_type.capitalize()} Payoff Diagram",
        xaxis_title="Future Price at Expiry",
        yaxis_title="Profit/Loss",
        hovermode='x unified',
        height=500
    )
    
    return fig

# Main Streamlit App
def main():
    st.title("🏦 Euribor Future Options Pricer")
    st.markdown("**Advanced pricing model for Mid-Curve and Non-Mid-Curve Euribor Future Options using SABR-M volatility model**")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("⚙️ Model Parameters")
        
        st.subheader("Market Data")
        future_price = st.number_input("Future Price (%)", 
                                      min_value=0.0, max_value=110.0, 
                                      value=96.50, step=0.01,
                                      help="Current Euribor future price (100 - rate)")
        
        euribor_rate = 100 - future_price
        st.info(f"Implied Euribor Rate: {euribor_rate:.2f}%")
        
        risk_free_rate = st.number_input("Risk-Free Rate (%)", 
                                        min_value=0.0, max_value=10.0, 
                                        value=3.5, step=0.1) / 100
        
        st.subheader("Option Parameters")
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        
        strike = st.number_input("Strike Price (%)", 
                                min_value=90.0, max_value=100.0, 
                                value=96.50, step=0.01)
        
        option_style = st.selectbox("Option Style", 
                                   ["Mid-Curve", "Non-Mid-Curve"])
        
        time_to_expiry = st.slider("Time to Option Expiry (Years)", 
                                  min_value=0.1, max_value=5.0, 
                                  value=1.0, step=0.1)
        
        if option_style == "Mid-Curve":
            time_to_future_expiry = st.slider("Time to Future Expiry (Years)", 
                                             min_value=time_to_expiry + 0.1, 
                                             max_value=10.0, 
                                             value=time_to_expiry + 1.0, 
                                             step=0.1)
        
        st.subheader("SABR Parameters")
        col1, col2 = st.columns(2)
        with col1:
            alpha = st.number_input("α (Alpha)", min_value=0.001, max_value=1.0, 
                                   value=0.05, step=0.001, format="%.3f")
            beta = st.number_input("β (Beta)", min_value=0.0, max_value=1.0, 
                                  value=0.5, step=0.1)
        with col2:
            rho = st.number_input("ρ (Rho)", min_value=-0.999, max_value=0.999, 
                                 value=-0.3, step=0.1)
            nu = st.number_input("ν (Nu)", min_value=0.001, max_value=2.0, 
                                value=0.3, step=0.1)
        
        calibrate_sabr = st.checkbox("Auto-Calibrate SABR", value=False)
    
    # Initialize SABR model
    sabr_model = SABRModel(alpha=alpha, beta=beta, rho=rho, nu=nu)
    
    # Initialize pricer
    pricer = EuriborFutureOption(sabr_model)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Pricing", "📈 Volatility Surface", 
                                            "💹 Payoff Analysis", "🔬 Greeks Analysis",
                                            "📚 Theory & Documentation"])
    
    with tab1:
        st.header("Option Pricing Results")
        
        # Calculate option price
        if option_style == "Mid-Curve":
            result = pricer.price_mid_curve_option(
                future_price/100, strike/100, time_to_expiry, 
                time_to_future_expiry, risk_free_rate, option_type.lower()
            )
        else:
            result = pricer.price_non_mid_curve_option(
                future_price/100, strike/100, time_to_expiry, 
                risk_free_rate, option_type.lower()
            )
        
        # Display results in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Option Price", f"{result['price']*100:.4f}%")
            st.metric("Delta", f"{result['delta']:.4f}")
        
        with col2:
            st.metric("Implied Volatility", f"{result['implied_vol']*100:.2f}%")
            st.metric("Gamma", f"{result['gamma']:.6f}")
        
        with col3:
            st.metric("Forward Price", f"{result['forward']*100:.4f}%")
            st.metric("Vega", f"{result['vega']:.4f}")
        
        with col4:
            moneyness = strike / future_price
            if moneyness < 0.98:
                money_status = "ITM"
            elif moneyness > 1.02:
                money_status = "OTM"
            else:
                money_status = "ATM"
            st.metric("Moneyness", f"{moneyness:.3f} ({money_status})")
            st.metric("Theta (Daily)", f"{result['theta']:.6f}")
        
        # Additional pricing information
        st.subheader("Detailed Pricing Breakdown")
        
        pricing_df = pd.DataFrame({
            'Parameter': ['Option Type', 'Style', 'Current Future Price', 
                         'Strike Price', 'Time to Expiry', 'Risk-Free Rate',
                         'Implied Volatility', 'Option Premium'],
            'Value': [option_type, option_style, f"{future_price:.2f}%", 
                     f"{strike:.2f}%", f"{time_to_expiry:.2f} years", 
                     f"{risk_free_rate*100:.2f}%", f"{result['implied_vol']*100:.2f}%",
                     f"{result['price']*100:.4f}%"]
        })
        
        st.dataframe(pricing_df, use_container_width=True)
        
        # Scenario Analysis
        st.subheader("Scenario Analysis")
        
        scenarios_df = pd.DataFrame({
            'Scenario': ['Bear (-10%)', 'Mild Bear (-5%)', 'Base', 
                        'Mild Bull (+5%)', 'Bull (+10%)'],
            'Future Price': [future_price * 0.9, future_price * 0.95, 
                           future_price, future_price * 1.05, future_price * 1.1]
        })
        
        scenario_prices = []
        for fp in scenarios_df['Future Price']:
            if option_style == "Mid-Curve":
                res = pricer.price_mid_curve_option(
                    fp/100, strike/100, time_to_expiry, 
                    time_to_future_expiry, risk_free_rate, option_type.lower()
                )
            else:
                res = pricer.price_non_mid_curve_option(
                    fp/100, strike/100, time_to_expiry, 
                    risk_free_rate, option_type.lower()
                )
            scenario_prices.append(res['price'] * 100)
        
        scenarios_df['Option Price (%)'] = scenario_prices
        scenarios_df['P&L (%)'] = scenarios_df['Option Price (%)'] - result['price'] * 100
        
        st.dataframe(scenarios_df.style.format({
            'Future Price': '{:.2f}%',
            'Option Price (%)': '{:.4f}',
            'P&L (%)': '{:+.4f}'
        }), use_container_width=True)
    
    with tab2:
        st.header("Volatility Surface Analysis")
        
        # Generate volatility surface
        maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
        surface_df = generate_volatility_surface(
            sabr_model, future_price/100, (0.9, 1.1), maturities
        )
        
        # Plot 3D surface
        fig_surface = plot_volatility_surface(surface_df)
        st.plotly_chart(fig_surface, use_container_width=True)
        
        # Volatility smile for selected maturity
        st.subheader("Volatility Smile")
        
        selected_maturity = st.selectbox("Select Maturity", maturities)
        smile_data = surface_df[surface_df['Maturity'] == selected_maturity]
        
        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=smile_data['Moneyness'],
            y=smile_data['Implied_Vol'] * 100,
            mode='lines+markers',
            name=f'{selected_maturity}Y',
            line=dict(width=2)
        ))
        
        fig_smile.update_layout(
            title=f"Volatility Smile - {selected_maturity} Year Maturity",
            xaxis_title="Moneyness (K/F)",
            yaxis_title="Implied Volatility (%)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_smile, use_container_width=True)
        
        # SABR parameters table
        st.subheader("SABR Model Parameters")
        sabr_params_df = pd.DataFrame({
            'Parameter': ['Alpha (α)', 'Beta (β)', 'Rho (ρ)', 'Nu (ν)'],
            'Value': [sabr_model.alpha, sabr_model.beta, sabr_model.rho, sabr_model.nu],
            'Description': [
                'Initial volatility',
                'CEV exponent (elasticity)',
                'Correlation between asset and volatility',
                'Volatility of volatility'
            ]
        })
        st.dataframe(sabr_params_df, use_container_width=True)
    
    with tab3:
        st.header("Payoff & P&L Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Position Settings")
            position_type = st.radio("Position", ["Long", "Short"])
            num_contracts = st.number_input("Number of Contracts", 
                                           min_value=1, value=100)
            contract_size = st.number_input("Contract Size (€)", 
                                           min_value=1000, value=1000000)
        
        with col1:
            # Plot payoff diagram
            fig_payoff = plot_option_payoff(
                future_price/100, strike/100, result['price'], 
                option_type.lower(), position_type == "Long"
            )
            st.plotly_chart(fig_payoff, use_container_width=True)
        
        # P&L Table
        st.subheader("P&L at Different Future Prices")
        
        future_prices_range = np.linspace(0.9 * future_price, 1.1 * future_price, 11)
        pnl_data = []
        
        for fp in future_prices_range:
            if option_type.lower() == 'call':
                intrinsic = max((fp - strike)/100, 0)
            else:
                intrinsic = max((strike - fp)/100, 0)
            
            if position_type == "Long":
                pnl = (intrinsic - result['price']) * num_contracts * contract_size
            else:
                pnl = (result['price'] - intrinsic) * num_contracts * contract_size
            
            pnl_data.append({
                'Future Price (%)': fp,
                'Intrinsic Value': intrinsic * 100,
                'P&L (€)': pnl,
                'Return (%)': (pnl / (result['price'] * num_contracts * contract_size)) * 100 if result['price'] > 0 else 0
            })
        
        pnl_df = pd.DataFrame(pnl_data)
        
        # Highlight rows based on P&L
        def highlight_pnl(row):
            if row['P&L (€)'] > 0:
                return ['background-color: #90EE90'] * len(row)
            elif row['P&L (€)'] < 0:
                return ['background-color: #FFB6C1'] * len(row)
            else:
                return [''] * len(row)
        
        st.dataframe(
            pnl_df.style.apply(highlight_pnl, axis=1).format({
                'Future Price (%)': '{:.2f}',
                'Intrinsic Value': '{:.4f}%',
                'P&L (€)': '{:,.0f}',
                'Return (%)': '{:+.2f}%'
            }),
            use_container_width=True
        )
    
    with tab4:
        st.header("Greeks Analysis")
        
        # Display all Greeks with explanations
        greeks_explanation = {
            'Delta': (result['delta'], 'Rate of change of option price with respect to future price'),
            'Gamma': (result['gamma'], 'Rate of change of delta with respect to future price'),
            'Vega': (result['vega'], 'Sensitivity to changes in implied volatility (per 1% vol change)'),
            'Theta': (result['theta'], 'Time decay per day'),
            'Rho': (result['rho'], 'Sensitivity to interest rate changes (per 1% rate change)')
        }
        
        col1, col2 = st.columns(2)
        
        for i, (greek, (value, description)) in enumerate(greeks_explanation.items()):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{greek}</h4>
                    <h2>{value:.6f}</h2>
                    <p>{description}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Greeks sensitivity analysis
        st.subheader("Greeks Sensitivity Analysis")
        
        # Delta and Gamma across strikes
        strikes_range = np.linspace(0.9 * future_price, 1.1 * future_price, 20)
        deltas = []
        gammas = []
        
        for k in strikes_range:
            if option_style == "Mid-Curve":
                res = pricer.price_mid_curve_option(
                    future_price/100, k/100, time_to_expiry, 
                    time_to_future_expiry, risk_free_rate, option_type.lower()
                )
            else:
                res = pricer.price_non_mid_curve_option(
                    future_price/100, k/100, time_to_expiry, 
                    risk_free_rate, option_type.lower()
                )
            deltas.append(res['delta'])
            gammas.append(res['gamma'])
        
        # Create subplots for Greeks
        fig_greeks = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delta vs Strike', 'Gamma vs Strike', 
                          'Vega vs Time', 'Theta vs Time')
        )
        
        # Delta plot
        fig_greeks.add_trace(
            go.Scatter(x=strikes_range, y=deltas, mode='lines', name='Delta'),
            row=1, col=1
        )
        
        # Gamma plot
        fig_greeks.add_trace(
            go.Scatter(x=strikes_range, y=gammas, mode='lines', name='Gamma'),
            row=1, col=2
        )
        
        # Vega and Theta vs time
        times = np.linspace(0.1, 2.0, 20)
        vegas = []
        thetas = []
        
        for t in times:
            if option_style == "Mid-Curve":
                res = pricer.price_mid_curve_option(
                    future_price/100, strike/100, t, 
                    t + 1.0, risk_free_rate, option_type.lower()
                )
            else:
                res = pricer.price_non_mid_curve_option(
                    future_price/100, strike/100, t, 
                    risk_free_rate, option_type.lower()
                )
            vegas.append(res['vega'])
            thetas.append(res['theta'])
        
        # Vega plot
        fig_greeks.add_trace(
            go.Scatter(x=times, y=vegas, mode='lines', name='Vega'),
            row=2, col=1
        )
        
        # Theta plot
        fig_greeks.add_trace(
            go.Scatter(x=times, y=thetas, mode='lines', name='Theta'),
            row=2, col=2
        )
        
        fig_greeks.update_xaxes(title_text="Strike (%)", row=1, col=1)
        fig_greeks.update_xaxes(title_text="Strike (%)", row=1, col=2)
        fig_greeks.update_xaxes(title_text="Time to Expiry (Years)", row=2, col=1)
        fig_greeks.update_xaxes(title_text="Time to Expiry (Years)", row=2, col=2)
        
        fig_greeks.update_yaxes(title_text="Delta", row=1, col=1)
        fig_greeks.update_yaxes(title_text="Gamma", row=1, col=2)
        fig_greeks.update_yaxes(title_text="Vega", row=2, col=1)
        fig_greeks.update_yaxes(title_text="Theta (Daily)", row=2, col=2)
        
        fig_greeks.update_layout(height=700, showlegend=False)
        
        st.plotly_chart(fig_greeks, use_container_width=True)
    
    with tab5:
        st.header("📚 Theoretical Overview & Documentation")
        
        st.markdown("""
        ## Euribor Future Options (RFOs)
        
        Euribor Future Options are options on Euribor interest rate futures, which are among the most 
        liquid interest rate derivatives in Europe. They provide exposure to European short-term interest rates.
        
        ### Key Concepts
        
        #### 1. **Euribor Futures**
        - Euribor futures are contracts on the 3-month Euribor interest rate
        - Quoted as 100 minus the implied interest rate
        - Example: If 3-month Euribor is 3.5%, the future trades at 96.50
        
        #### 2. **Mid-Curve vs Non-Mid-Curve Options**
        
        **Mid-Curve Options:**
        - Options that expire before the underlying future
        - Provide exposure to forward-starting interest rate periods
        - Commonly used for hedging future rate resets
        - Require adjustment for the time gap between option and future expiry
        
        **Non-Mid-Curve Options:**
        - Standard options where option and future expire simultaneously
        - Simpler pricing structure
        - Direct exposure to the underlying future
        
        ### Pricing Methodology
        
        #### Black Model
        The pricing uses the Black model (Black-76), which is the standard for options on futures:
        
        For a Call option:
        $$C = e^{-rT}[F \\cdot N(d_1) - K \\cdot N(d_2)]$$
        
        For a Put option:
        $$P = e^{-rT}[K \\cdot N(-d_2) - F \\cdot N(-d_1)]$$
        
        Where:
        - $d_1 = \\frac{\\ln(F/K) + \\frac{1}{2}\\sigma^2 T}{\\sigma\\sqrt{T}}$
        - $d_2 = d_1 - \\sigma\\sqrt{T}$
        - $F$ = Future price
        - $K$ = Strike price
        - $r$ = Risk-free rate
        - $T$ = Time to expiry
        - $\\sigma$ = Implied volatility
        - $N(\\cdot)$ = Cumulative normal distribution function
        
        ### SABR-M Model
        
        The SABR (Stochastic Alpha Beta Rho) model is used for modeling the volatility smile:
        
        #### Model Dynamics:
        $$dF_t = \\alpha_t F_t^\\beta dW_t^F$$
        $$d\\alpha_t = \\nu \\alpha_t dW_t^\\alpha$$
        $$dW_t^F \\cdot dW_t^\\alpha = \\rho dt$$
        
        #### Parameters:
        - **α (Alpha)**: Initial volatility level
        - **β (Beta)**: CEV exponent, controls the relationship between forward rate and volatility
        - **ρ (Rho)**: Correlation between forward rate and volatility
        - **ν (Nu)**: Volatility of volatility
        
        #### SABR Implied Volatility Formula:
        The model provides closed-form approximation for implied volatility, capturing the smile effect 
        observed in interest rate options markets.
        
        ### Risk-Neutral Pricing
        
        Under the risk-neutral measure, the option price is the discounted expected payoff:
        
        $$V_0 = e^{-rT} \\mathbb{E}^Q[\\max(F_T - K, 0)]$$
        
        This framework ensures no-arbitrage pricing and is consistent with:
        - **Swaptions**: Options on interest rate swaps
        - **Cap/Floors**: Options on floating interest rates
        
        ### Greeks Interpretation
        
        | Greek | Interpretation | Use Case |
        |-------|---------------|----------|
        | **Delta (Δ)** | Price sensitivity to future price changes | Hedging directional risk |
        | **Gamma (Γ)** | Rate of change of Delta | Managing hedge ratios |
        | **Vega (ν)** | Sensitivity to volatility changes | Volatility risk management |
        | **Theta (Θ)** | Time decay | Understanding holding costs |
        | **Rho (ρ)** | Sensitivity to interest rate changes | Rate risk assessment |
        
        ### Practical Applications
        
        1. **Interest Rate Hedging**: Protect against adverse rate movements
        2. **Volatility Trading**: Express views on interest rate volatility
        3. **Curve Trading**: Take positions on different parts of the yield curve
        4. **Portfolio Management**: Manage duration and convexity risk
        
        ### Implementation Notes
        
        This implementation features:
        - Full SABR model calibration capabilities
        - Separate handling of Mid-Curve and Non-Mid-Curve options
        - Complete Greeks calculation
        - Interactive visualization of volatility surfaces
        - Scenario analysis tools
        
        The pricer can be extended to handle:
        - American-style options
        - Exotic option structures
        - Multi-curve environments
        - Credit spread adjustments
        """)

if __name__ == "__main__":
    main()

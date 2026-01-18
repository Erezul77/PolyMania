"""
PolyMania Bot Army Dashboard
============================

Real-time monitoring dashboard with:
- Portfolio performance
- Active positions
- Trading signals
- Risk metrics
- ML model performance
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import time
import os

# Try to import redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="PolyMania Bot Army HQ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .main-header {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00ff88, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #00ff88;
    }
    
    .metric-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .status-live {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #00ff88;
        font-family: 'Rajdhani', sans-serif;
    }
    
    .status-dot {
        width: 12px;
        height: 12px;
        background: #00ff88;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 10px #00ff88; }
        50% { opacity: 0.5; box-shadow: 0 0 20px #00ff88; }
    }
    
    .position-card {
        background: rgba(0,212,255,0.1);
        border-left: 4px solid #00d4ff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    
    .signal-buy { color: #00ff88; }
    .signal-sell { color: #ff4444; }
    
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(0,255,136,0.2);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def get_redis_client():
    """Get Redis client."""
    if not REDIS_AVAILABLE:
        return None
    try:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        client = redis.Redis(host=host, port=port, decode_responses=True)
        client.ping()
        return client
    except:
        return None


def get_system_status(redis_client):
    """Get system status from Redis."""
    if redis_client:
        try:
            data = redis_client.get("bot_army:status")
            if data:
                return json.loads(data)
        except:
            pass
    
    # Mock data for demo
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cycle_count": 1523,
        "signals_generated": 247,
        "trades_executed": 89,
        "portfolio_value": 12456.78,
        "position_count": 8,
        "risk_level": "NORMAL",
        "collectors": {
            "markets": {"collected_count": 15000},
            "trades": {"collected_count": 45000},
            "events": {"collected_count": 250}
        },
        "strategy": {
            "name": "ensemble",
            "signals_generated": 247,
            "win_rate": 58.4
        },
        "execution": {
            "total_trades": 89,
            "total_fees": 23.45,
            "paper_portfolio_value": 12456.78
        }
    }


def get_portfolio_history():
    """Get portfolio value history."""
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
    base = 10000
    returns = np.random.normal(0.001, 0.02, 100)
    values = base * np.cumprod(1 + returns)
    return pd.DataFrame({"timestamp": dates, "value": values})


def get_positions():
    """Get current positions."""
    # Sample positions
    return [
        {"market": "BTC > $50K", "side": "LONG", "size": 500, "entry": 0.45, "current": 0.52, "pnl": 77.78, "pnl_pct": 15.6},
        {"market": "ETH > $3K", "side": "LONG", "size": 300, "entry": 0.38, "current": 0.41, "pnl": 23.68, "pnl_pct": 7.9},
        {"market": "Trump 2024", "side": "LONG", "size": 400, "entry": 0.52, "current": 0.55, "pnl": 23.08, "pnl_pct": 5.8},
        {"market": "Fed Rate Cut", "side": "SHORT", "size": 250, "entry": 0.65, "current": 0.58, "pnl": 26.92, "pnl_pct": 10.8},
    ]


def get_signals_history():
    """Get recent signals."""
    return [
        {"time": "14:32", "market": "BTC > $50K", "signal": "BUY", "confidence": 0.78, "strategy": "momentum"},
        {"time": "14:28", "market": "ETH > $3K", "signal": "BUY", "confidence": 0.72, "strategy": "stat_arb"},
        {"time": "14:15", "market": "Fed Rate Cut", "signal": "SELL", "confidence": 0.81, "strategy": "mean_reversion"},
        {"time": "14:02", "market": "NVDA > $500", "signal": "BUY", "confidence": 0.69, "strategy": "ensemble"},
        {"time": "13:45", "market": "Trump 2024", "signal": "BUY", "confidence": 0.75, "strategy": "momentum"},
    ]


def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 POLYMANIA HQ</h1>', unsafe_allow_html=True)
    
    # Status bar
    redis_client = get_redis_client()
    status = get_system_status(redis_client)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f'''
        <div style="text-align: center; padding: 10px;">
            <span class="status-live">
                <span class="status-dot"></span>
                LIVE - {datetime.now().strftime("%H:%M:%S")}
            </span>
            <span style="margin-left: 20px; color: #888;">
                Cycle #{status.get("cycle_count", 0):,}
            </span>
        </div>
        ''', unsafe_allow_html=True)
    
    st.divider()
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        portfolio_val = status.get("portfolio_value", status.get("execution", {}).get("paper_portfolio_value", 10000))
        pnl = portfolio_val - 10000
        pnl_pct = (pnl / 10000) * 100
        st.metric(
            "💰 Portfolio Value",
            f"${portfolio_val:,.2f}",
            f"{pnl_pct:+.2f}%"
        )
    
    with col2:
        win_rate = status.get("strategy", {}).get("win_rate", 58)
        st.metric(
            "🎯 Win Rate",
            f"{win_rate:.1f}%",
            f"+{win_rate - 50:.1f}% vs random"
        )
    
    with col3:
        signals = status.get("signals_generated", 0)
        st.metric(
            "📊 Signals Generated",
            f"{signals:,}",
            f"+{signals // 24}/hr avg"
        )
    
    with col4:
        trades = status.get("trades_executed", 0)
        st.metric(
            "⚡ Trades Executed",
            f"{trades:,}",
            f"{status.get('position_count', 0)} open"
        )
    
    st.divider()
    
    # Charts row
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.subheader("📈 Portfolio Performance")
        
        df_history = get_portfolio_history()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_history['timestamp'],
            y=df_history['value'],
            mode='lines',
            fill='tozeroy',
            line=dict(color='#00ff88', width=2),
            fillcolor='rgba(0,255,136,0.1)'
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, color='#888'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='#888'),
            margin=dict(l=0, r=0, t=10, b=0),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.subheader("🎯 Strategy Performance")
        
        strategies = ['Momentum', 'StatArb', 'MeanRev']
        values = [35, 38, 27]
        
        fig = go.Figure(data=[go.Pie(
            labels=strategies,
            values=values,
            hole=0.6,
            marker_colors=['#00ff88', '#00d4ff', '#7c3aed']
        )])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(font=dict(color='#888')),
            margin=dict(l=0, r=0, t=10, b=0),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Positions and Signals
    pos_col, sig_col = st.columns(2)
    
    with pos_col:
        st.subheader("📍 Active Positions")
        
        positions = get_positions()
        
        for pos in positions:
            pnl_color = "signal-buy" if pos["pnl"] > 0 else "signal-sell"
            st.markdown(f'''
            <div class="position-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: #fff;">{pos["market"]}</strong>
                        <span style="color: #888; margin-left: 10px;">{pos["side"]}</span>
                    </div>
                    <div class="{pnl_color}" style="font-weight: bold;">
                        {pos["pnl_pct"]:+.1f}%
                    </div>
                </div>
                <div style="color: #888; font-size: 0.9rem; margin-top: 5px;">
                    Size: ${pos["size"]:,.0f} | Entry: {pos["entry"]:.2f} | Current: {pos["current"]:.2f}
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    with sig_col:
        st.subheader("📡 Recent Signals")
        
        signals = get_signals_history()
        
        for sig in signals:
            sig_color = "signal-buy" if sig["signal"] == "BUY" else "signal-sell"
            st.markdown(f'''
            <div style="background: rgba(255,255,255,0.03); padding: 0.8rem; border-radius: 8px; margin: 0.3rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #888;">{sig["time"]}</span>
                        <strong style="color: #fff; margin-left: 10px;">{sig["market"]}</strong>
                    </div>
                    <span class="{sig_color}" style="font-weight: bold;">
                        {sig["signal"]} {sig["confidence"]:.0%}
                    </span>
                </div>
                <div style="color: #666; font-size: 0.8rem;">
                    Strategy: {sig["strategy"]}
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.divider()
    
    # Risk metrics
    st.subheader("🛡️ Risk Dashboard")
    
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
    
    with risk_col1:
        st.metric("Risk Level", status.get("risk_level", "NORMAL"), "✅ OK")
    
    with risk_col2:
        st.metric("VaR (95%)", "2.3%", "-0.2%")
    
    with risk_col3:
        st.metric("Drawdown", "1.8%", "Below limit")
    
    with risk_col4:
        st.metric("Exposure", "45%", "Balanced")
    
    # Sidebar
    with st.sidebar:
        icon_path = Path(__file__).resolve().parent / "assets" / "polymarket.svg"
        if icon_path.exists():
            st.image(str(icon_path), width=50)
        st.title("🎯 PolyMania Bot Army")
        st.markdown("---")
        
        st.subheader("⚙️ System Status")
        st.write(f"**Mode:** {'🟢 Live' if redis_client else '🟡 Demo'}")
        st.write(f"**Uptime:** {status.get('cycle_count', 0) * 10 // 3600}h {(status.get('cycle_count', 0) * 10 % 3600) // 60}m")
        
        st.markdown("---")
        
        st.subheader("📊 Data Collection")
        collectors = status.get("collectors", {})
        st.write(f"Markets: {collectors.get('markets', {}).get('collected_count', 0):,}")
        st.write(f"Trades: {collectors.get('trades', {}).get('collected_count', 0):,}")
        st.write(f"Events: {collectors.get('events', {}).get('collected_count', 0):,}")
        
        st.markdown("---")
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        st.caption("PolyMania Bot Army v2.0")
        st.caption(f"© {datetime.now().year}")


if __name__ == "__main__":
    main()

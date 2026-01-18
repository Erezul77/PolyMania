"""
PolyMania Dashboard - Real-time Web Interface

A beautiful web dashboard to monitor:
- Bot status
- Portfolio & positions
- Recent signals
- Whale activity
- Intelligent systems
- Live logs
"""

import os
import json
import logging
from datetime import datetime, timezone
from flask import Flask, render_template_string, jsonify
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('polymania.dashboard')

app = Flask(__name__)

# Data directory
DATA_DIR = Path('/app/data') if os.path.exists('/app/data') else Path('data')


def read_json_file(filename):
    """Read a JSON file safely."""
    try:
        filepath = DATA_DIR / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f'Error reading {filename}: {e}')
    return {}


def read_csv_tail(filename, lines=20):
    """Read last N lines of a CSV file."""
    try:
        filepath = DATA_DIR / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()
                return all_lines[-lines:] if len(all_lines) > lines else all_lines
    except Exception as e:
        logger.debug(f'Error reading {filename}: {e}')
    return []


def get_portfolio_data():
    """Get portfolio data."""
    data = read_json_file('paper_portfolio.json')
    if not data:
        return {
            'cash': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'positions': [],
            'total_value': 0,
        }
    
    positions = []
    total_value = data.get('cash', 0)
    
    for event_id, pos in data.get('positions', {}).items():
        value = pos.get('shares', 0) * pos.get('current_price', 0)
        total_value += value
        positions.append({
            'title': pos.get('event_title', '')[:50],
            'shares': pos.get('shares', 0),
            'entry_price': pos.get('entry_price', 0),
            'current_price': pos.get('current_price', 0),
            'value': value,
            'pnl_pct': ((pos.get('current_price', 0) - pos.get('entry_price', 1)) / pos.get('entry_price', 1) * 100) if pos.get('entry_price', 0) > 0 else 0,
        })
    
    return {
        'cash': data.get('cash', 0),
        'trades': data.get('trades', 0),
        'wins': data.get('wins', 0),
        'losses': data.get('losses', 0),
        'positions': positions,
        'total_value': total_value,
    }


def get_signals_data():
    """Get recent signals."""
    lines = read_csv_tail('trading_signals.csv', 30)
    signals = []
    
    for line in reversed(lines):
        try:
            parts = line.strip().split(',')
            if len(parts) >= 6 and parts[2] != 'HOLD':
                signals.append({
                    'time': parts[0][:19] if len(parts[0]) > 19 else parts[0],
                    'type': parts[2],
                    'confidence': float(parts[3]) if parts[3] else 0,
                    'price': float(parts[4]) if parts[4] else 0,
                    'reasons': parts[7] if len(parts) > 7 else '',
                })
        except Exception:
            continue
    
    return signals[:15]


def get_whale_data():
    """Get whale signals."""
    lines = read_csv_tail('whale_signals.csv', 20)
    whales = []
    
    for line in reversed(lines):
        try:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                whales.append({
                    'time': parts[0][:19] if len(parts[0]) > 19 else parts[0],
                    'action': parts[1] if len(parts) > 1 else '',
                    'event': parts[2][:40] if len(parts) > 2 else '',
                    'amount': parts[3] if len(parts) > 3 else '',
                })
        except Exception:
            continue
    
    return whales[:10]


def get_brain_data():
    """Get ensemble brain data."""
    data = read_json_file('ensemble_brain.json')
    if not data:
        return {
            'decisions': 0,
            'wins': 0,
            'total_pnl': 0,
            'weights': {},
        }
    
    perf = data.get('ensemble_performance', {})
    weights = data.get('source_weights', {})
    
    return {
        'decisions': perf.get('decisions', 0),
        'wins': perf.get('wins', 0),
        'total_pnl': perf.get('total_pnl', 0),
        'max_drawdown': perf.get('max_drawdown', 0),
        'weights': weights,
    }


def get_risk_data():
    """Get risk manager data."""
    data = read_json_file('risk_manager.json')
    if not data:
        return {
            'risk_level': 'NORMAL',
            'equity': 1.0,
            'peak_equity': 1.0,
            'positions': 0,
        }
    
    return {
        'risk_level': data.get('risk_level', 'normal').upper(),
        'equity': data.get('current_equity', 1.0),
        'peak_equity': data.get('peak_equity', 1.0),
        'positions': len(data.get('positions', {})),
    }


# HTML Template with modern dark theme
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 PolyMania Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a24;
            --bg-hover: #22222e;
            --accent-green: #00ff88;
            --accent-red: #ff4466;
            --accent-blue: #4488ff;
            --accent-yellow: #ffcc00;
            --accent-purple: #aa66ff;
            --text-primary: #ffffff;
            --text-secondary: #888899;
            --border: #2a2a3a;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            background-image: 
                radial-gradient(ellipse at top left, rgba(0, 255, 136, 0.05) 0%, transparent 50%),
                radial-gradient(ellipse at bottom right, rgba(68, 136, 255, 0.05) 0%, transparent 50%);
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 30px;
        }
        
        .logo {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--bg-card);
            border-radius: 20px;
            font-size: 14px;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--accent-green);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        
        .card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 40px rgba(0, 255, 136, 0.1);
        }
        
        .card-title {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .big-number {
            font-family: 'JetBrains Mono', monospace;
            font-size: 42px;
            font-weight: 700;
            color: var(--accent-green);
            line-height: 1;
        }
        
        .big-number.red { color: var(--accent-red); }
        .big-number.blue { color: var(--accent-blue); }
        .big-number.yellow { color: var(--accent-yellow); }
        
        .stats-row {
            display: flex;
            gap: 20px;
            margin-top: 16px;
        }
        
        .stat {
            flex: 1;
        }
        
        .stat-label {
            font-size: 12px;
            color: var(--text-secondary);
        }
        
        .stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 18px;
            font-weight: 600;
        }
        
        .positions-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .position-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }
        
        .position-item:last-child {
            border-bottom: none;
        }
        
        .position-title {
            font-size: 13px;
            max-width: 200px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .position-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 600;
        }
        
        .pnl-positive { color: var(--accent-green); }
        .pnl-negative { color: var(--accent-red); }
        
        .signal-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 0;
            border-bottom: 1px solid var(--border);
        }
        
        .signal-badge {
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .signal-buy { background: rgba(0, 255, 136, 0.2); color: var(--accent-green); }
        .signal-sell { background: rgba(255, 68, 102, 0.2); color: var(--accent-red); }
        .signal-strong { border: 1px solid currentColor; }
        
        .signal-confidence {
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
        }
        
        .signal-reasons {
            font-size: 11px;
            color: var(--text-secondary);
            max-width: 300px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .whale-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 0;
            font-size: 13px;
        }
        
        .whale-icon { font-size: 18px; }
        
        .weight-bar {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 6px 0;
        }
        
        .weight-label {
            width: 120px;
            font-size: 12px;
            color: var(--text-secondary);
        }
        
        .weight-track {
            flex: 1;
            height: 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .weight-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
            border-radius: 4px;
            transition: width 0.3s;
        }
        
        .weight-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            width: 50px;
            text-align: right;
        }
        
        .risk-level {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
        }
        
        .risk-normal { background: rgba(0, 255, 136, 0.2); color: var(--accent-green); }
        .risk-conservative { background: rgba(255, 204, 0, 0.2); color: var(--accent-yellow); }
        .risk-defensive { background: rgba(255, 68, 102, 0.2); color: var(--accent-red); }
        
        .refresh-time {
            font-size: 12px;
            color: var(--text-secondary);
        }
        
        .card-wide {
            grid-column: span 2;
        }
        
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            .card-wide { grid-column: span 1; }
            .big-number { font-size: 32px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                🚀 PolyMania
            </div>
            <div class="status-badge">
                <div class="status-dot"></div>
                <span>Live</span>
                <span class="refresh-time" id="lastUpdate"></span>
            </div>
        </header>
        
        <div class="grid">
            <!-- Portfolio Value -->
            <div class="card">
                <div class="card-title">💰 Portfolio Value</div>
                <div class="big-number" id="totalValue">$0</div>
                <div class="stats-row">
                    <div class="stat">
                        <div class="stat-label">Cash</div>
                        <div class="stat-value" id="cash">$0</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Positions</div>
                        <div class="stat-value" id="positionCount">0</div>
                    </div>
                </div>
            </div>
            
            <!-- Win/Loss -->
            <div class="card">
                <div class="card-title">📊 Trading Stats</div>
                <div class="stats-row">
                    <div class="stat">
                        <div class="stat-label">Total Trades</div>
                        <div class="stat-value" id="trades">0</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Wins</div>
                        <div class="stat-value pnl-positive" id="wins">0</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Losses</div>
                        <div class="stat-value pnl-negative" id="losses">0</div>
                    </div>
                </div>
            </div>
            
            <!-- Risk Status -->
            <div class="card">
                <div class="card-title">⚠️ Risk Manager</div>
                <div id="riskLevel" class="risk-level risk-normal">NORMAL</div>
                <div class="stats-row">
                    <div class="stat">
                        <div class="stat-label">Equity</div>
                        <div class="stat-value" id="equity">1.00</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Peak</div>
                        <div class="stat-value" id="peakEquity">1.00</div>
                    </div>
                </div>
            </div>
            
            <!-- Brain Status -->
            <div class="card">
                <div class="card-title">🧠 Ensemble Brain</div>
                <div class="stats-row">
                    <div class="stat">
                        <div class="stat-label">Decisions</div>
                        <div class="stat-value" id="brainDecisions">0</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Brain Wins</div>
                        <div class="stat-value" id="brainWins">0</div>
                    </div>
                </div>
            </div>
            
            <!-- Positions -->
            <div class="card">
                <div class="card-title">📈 Open Positions</div>
                <div class="positions-list" id="positionsList"></div>
            </div>
            
            <!-- Source Weights -->
            <div class="card">
                <div class="card-title">⚖️ Signal Source Weights</div>
                <div id="weightsList"></div>
            </div>
            
            <!-- Recent Signals -->
            <div class="card card-wide">
                <div class="card-title">📡 Recent Signals</div>
                <div id="signalsList"></div>
            </div>
            
            <!-- Whale Activity -->
            <div class="card">
                <div class="card-title">🐋 Whale Activity</div>
                <div id="whalesList"></div>
            </div>
        </div>
    </div>
    
    <script>
        async function fetchData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
        
        function updateDashboard(data) {
            // Portfolio
            document.getElementById('totalValue').textContent = '$' + data.portfolio.total_value.toFixed(2);
            document.getElementById('cash').textContent = '$' + data.portfolio.cash.toFixed(2);
            document.getElementById('positionCount').textContent = data.portfolio.positions.length;
            document.getElementById('trades').textContent = data.portfolio.trades;
            document.getElementById('wins').textContent = data.portfolio.wins;
            document.getElementById('losses').textContent = data.portfolio.losses;
            
            // Risk
            const riskEl = document.getElementById('riskLevel');
            riskEl.textContent = data.risk.risk_level;
            riskEl.className = 'risk-level risk-' + data.risk.risk_level.toLowerCase();
            document.getElementById('equity').textContent = data.risk.equity.toFixed(2);
            document.getElementById('peakEquity').textContent = data.risk.peak_equity.toFixed(2);
            
            // Brain
            document.getElementById('brainDecisions').textContent = data.brain.decisions;
            document.getElementById('brainWins').textContent = data.brain.wins;
            
            // Positions
            const positionsHtml = data.portfolio.positions.map(p => `
                <div class="position-item">
                    <div class="position-title">${p.title}</div>
                    <div>
                        <span class="position-value">$${p.value.toFixed(2)}</span>
                        <span class="position-value ${p.pnl_pct >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                            ${p.pnl_pct >= 0 ? '+' : ''}${p.pnl_pct.toFixed(1)}%
                        </span>
                    </div>
                </div>
            `).join('');
            document.getElementById('positionsList').innerHTML = positionsHtml || '<div style="color: var(--text-secondary)">No open positions</div>';
            
            // Weights
            const weights = Object.entries(data.brain.weights).sort((a, b) => b[1] - a[1]);
            const weightsHtml = weights.map(([name, value]) => `
                <div class="weight-bar">
                    <div class="weight-label">${name}</div>
                    <div class="weight-track">
                        <div class="weight-fill" style="width: ${(value / 2) * 100}%"></div>
                    </div>
                    <div class="weight-value">${value.toFixed(2)}x</div>
                </div>
            `).join('');
            document.getElementById('weightsList').innerHTML = weightsHtml || '<div style="color: var(--text-secondary)">No weight data</div>';
            
            // Signals
            const signalsHtml = data.signals.map(s => {
                const isBuy = s.type.includes('BUY');
                const isStrong = s.type.includes('STRONG');
                return `
                    <div class="signal-item">
                        <div class="signal-badge ${isBuy ? 'signal-buy' : 'signal-sell'} ${isStrong ? 'signal-strong' : ''}">${s.type}</div>
                        <div class="signal-confidence">${(s.confidence * 100).toFixed(0)}%</div>
                        <div class="signal-reasons">${s.reasons}</div>
                    </div>
                `;
            }).join('');
            document.getElementById('signalsList').innerHTML = signalsHtml || '<div style="color: var(--text-secondary)">No recent signals</div>';
            
            // Whales
            const whalesHtml = data.whales.map(w => `
                <div class="whale-item">
                    <span class="whale-icon">🐋</span>
                    <span>${w.action}</span>
                    <span style="color: var(--text-secondary)">${w.event}</span>
                    <span style="color: var(--accent-yellow)">${w.amount}</span>
                </div>
            `).join('');
            document.getElementById('whalesList').innerHTML = whalesHtml || '<div style="color: var(--text-secondary)">No whale activity</div>';
            
            // Update time
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
        }
        
        // Initial fetch
        fetchData();
        
        // Refresh every 30 seconds
        setInterval(fetchData, 30000);
    </script>
</body>
</html>
'''


@app.route('/')
def dashboard():
    """Serve the dashboard."""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/data')
def api_data():
    """API endpoint for dashboard data."""
    return jsonify({
        'portfolio': get_portfolio_data(),
        'signals': get_signals_data(),
        'whales': get_whale_data(),
        'brain': get_brain_data(),
        'risk': get_risk_data(),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


def main():
    """Run the dashboard server."""
    port = int(os.environ.get('DASHBOARD_PORT', 8080))
    logger.info(f'Starting PolyMania Dashboard on port {port}')
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    main()

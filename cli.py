#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import sys
import json

VPS_HOST = "129.159.134.31"
VPS_USER = "ubuntu"
VPS_PATH = "~/PolyMania"
SSH_KEY = "C:/Users/erezs/.ssh/oci_polymania.key"

def ssh_cmd(cmd):
    full = "ssh -i " + SSH_KEY + " " + VPS_USER + "@" + VPS_HOST + " " + chr(34) + "cd " + VPS_PATH + " && " + cmd + chr(34)
    try:
        r = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=30)
        return r.stdout + r.stderr
    except:
        return "Error"

def show_status():
    print("\n[BOT STATUS]")
    print(ssh_cmd("docker compose ps"))

def show_portfolio():
    print("\n[PAPER PORTFOLIO]")
    out = ssh_cmd("cat data/paper_portfolio.json")
    try:
        d = json.loads(out)
        print("Cash: $" + str(round(d.get("cash",0),2)))
        print("Trades: " + str(d.get("trades",0)))
        print("Wins: " + str(d.get("wins",0)) + " Losses: " + str(d.get("losses",0)))
        pos = d.get("positions",{})
        if pos:
            print("Open Positions: " + str(len(pos)))
            tv = 0
            for k,p in pos.items():
                v = p["shares"] * p["current_price"]
                tv += v
                print("  " + p["event_title"][:35] + " $" + str(round(v,2)))
            print("Total: $" + str(round(d.get("cash",0)+tv,2)))
    except:
        print(out)

def show_trades():
    print("\n[RECENT TRADES]")
    print(ssh_cmd("tail -10 data/paper_trades.csv"))

def show_signals():
    print("\n[RECENT SIGNALS]")
    print(ssh_cmd("grep -v HOLD data/trading_signals.csv | tail -10"))

def show_ai():
    print("\n[AI INSIGHTS]")
    c = "docker compose exec -T market_analyzer python -c 'from polymania.ai_learner import get_ai_learner; print(get_ai_learner().get_insights())'"
    print(ssh_cmd(c))

def show_multi():
    print("\n[MULTI-STRATEGY STATUS - 100 strategies x $1000]")
    c = "docker compose exec -T market_analyzer python -c 'from polymania.multi_strategy import get_multi_trader; print(get_multi_trader().get_summary())'"
    print(ssh_cmd(c))

def show_logs():
    print("\n[RECENT LOGS]")
    print(ssh_cmd("docker compose logs --tail=30 market_analyzer"))

def show_telegram_hits():
    print("\n[TELEGRAM ALPHA HITS]")
    print(ssh_cmd("tail -20 data/telegram_hits.csv"))

def show_alpha_signals():
    print("\n[ALPHA SIGNALS]")
    print(ssh_cmd("tail -20 data/alpha_signals.csv"))

def show_contrarian():
    print("\n[CONTRARIAN SKIPS - Last 20 log entries]")
    print(ssh_cmd("docker compose logs --tail=100 market_analyzer | grep -i 'CONTRARIAN' | tail -20"))

def show_whales():
    print("\n[WHALE TRACKER STATUS]")
    c = "docker compose exec -T market_analyzer python -c 'from polymania.whale_tracker import get_whale_tracker; print(get_whale_tracker().get_summary())'"
    print(ssh_cmd(c))

def show_whale_signals():
    print("\n[RECENT WHALE SIGNALS]")
    print(ssh_cmd("tail -15 data/whale_signals.csv"))

def scan_whales():
    print("\n[SCANNING FOR WHALES...]")
    c = "docker compose exec -T market_analyzer python -m polymania.whale_tracker"
    print(ssh_cmd(c))

def show_strategy():
    print("\n[STRATEGY LEARNER STATUS]")
    c = "docker compose exec -T market_analyzer python -c 'from polymania.strategy_learner import get_strategy_learner; print(get_strategy_learner().get_summary())'"
    print(ssh_cmd(c))

def discover_traders():
    print("\n[DISCOVERING TOP TRADERS...]")
    c = "docker compose exec -T market_analyzer python -m polymania.strategy_learner"
    print(ssh_cmd(c))

def show_segments():
    print("\n[SEGMENT STRATEGIES]")
    c = "docker compose exec -T market_analyzer python -c 'from polymania.segment_router import get_segment_summary; print(get_segment_summary())'"
    print(ssh_cmd(c))

def show_regimes():
    print("\n[MARKET REGIME CONFIGS]")
    c = "docker compose exec -T market_analyzer python -c 'from polymania.market_regime import get_regime_summary; print(get_regime_summary())'"
    print(ssh_cmd(c))

# === NEW: Advanced Profit Optimization Commands ===

def show_optimizer():
    print("\n[PROFIT OPTIMIZER STATUS]")
    c = "docker compose exec -T market_analyzer python -c 'from polymania.profit_optimizer import get_profit_optimizer; print(get_profit_optimizer().get_summary())'"
    print(ssh_cmd(c))

def show_correlations():
    print("\n[CROSS-MARKET CORRELATIONS]")
    c = "docker compose exec -T market_analyzer python -c 'from polymania.correlation_matrix import get_correlation_matrix; print(get_correlation_matrix().get_summary())'"
    print(ssh_cmd(c))

def show_patterns():
    print("\n[OPPORTUNITY SCANNER - PATTERN DETECTION]")
    c = "docker compose exec -T market_analyzer python -c 'from polymania.opportunity_scanner import get_opportunity_scanner; print(get_opportunity_scanner().get_summary())'"
    print(ssh_cmd(c))

def show_brain():
    print("\n[ENSEMBLE BRAIN - DECISION ENGINE]")
    c = "docker compose exec -T market_analyzer python -c 'from polymania.ensemble_brain import get_ensemble_brain; print(get_ensemble_brain().get_summary())'"
    print(ssh_cmd(c))

def show_risk():
    print("\n[RISK MANAGER STATUS]")
    c = "docker compose exec -T market_analyzer python -c 'from polymania.risk_manager import get_risk_manager; print(get_risk_manager().get_summary())'"
    print(ssh_cmd(c))

def show_intelligence():
    """Show all intelligent systems at once"""
    print("\n" + "="*60)
    print("=== POLYMANIA INTELLIGENT TRADING SYSTEMS ===")
    print("="*60)
    show_optimizer()
    show_brain()
    show_risk()
    show_correlations()
    show_patterns()

def restart_bot():
    print("Restarting...")
    print(ssh_cmd("docker compose restart market_analyzer"))

def show_help():
    print("Commands:")
    print("  status      - Bot container status")
    print("  portfolio/p - Paper trading portfolio")
    print("  trades/t    - Recent trades")
    print("  signals/s   - Recent trading signals")
    print("  ai          - AI learning insights")
    print("  multi/m     - Multi-strategy status")
    print("  logs/l      - Recent analyzer logs")
    print("  --- ALPHA SIGNALS ---")
    print("  tg          - Telegram alpha hits")
    print("  alpha/a     - Alpha signals from external intel")
    print("  whales/w    - Whale tracker status")
    print("  wsignals    - Recent whale signals")
    print("  wscan       - Scan for new whales")
    print("  --- INTELLIGENT STRATEGIES ---")
    print("  segments    - Segment strategy configs (crypto/sports/etc)")
    print("  regimes     - Market regime configs (volatility/trend/etc)")
    print("  strategy    - Strategy learner (patterns from top traders)")
    print("  discover    - Discover top traders to follow")
    print("  contrarian/c - Contrarian logic skips")
    print("  --- ADVANCED PROFIT OPTIMIZATION ---")
    print("  optimizer/o - Multi-dimensional signal scorer")
    print("  brain/b     - Ensemble decision engine")
    print("  risk/r      - Risk manager (position sizing/drawdown)")
    print("  corr        - Cross-market correlation matrix")
    print("  patterns    - Pattern & opportunity scanner")
    print("  intel       - ALL intelligent systems summary")
    print("  --- SYSTEM ---")
    print("  restart     - Restart market analyzer")
    print("  ssh         - Open SSH to VPS")
    print("  help/h      - This help")
    print("  quit/q      - Exit CLI")

def open_ssh():
    subprocess.run("ssh -i " + SSH_KEY + " " + VPS_USER + "@" + VPS_HOST, shell=True)

def main():
    print("PolyMania CLI - type help")
    cmds = {"status":show_status,"portfolio":show_portfolio,"p":show_portfolio,"trades":show_trades,"t":show_trades,"signals":show_signals,"s":show_signals,"ai":show_ai,"multi":show_multi,"m":show_multi,"logs":show_logs,"l":show_logs,"tg":show_telegram_hits,"alpha":show_alpha_signals,"a":show_alpha_signals,"contrarian":show_contrarian,"c":show_contrarian,"whales":show_whales,"w":show_whales,"wsignals":show_whale_signals,"wscan":scan_whales,"strategy":show_strategy,"discover":discover_traders,"segments":show_segments,"regimes":show_regimes,"optimizer":show_optimizer,"o":show_optimizer,"brain":show_brain,"b":show_brain,"risk":show_risk,"r":show_risk,"corr":show_correlations,"patterns":show_patterns,"intel":show_intelligence,"restart":restart_bot,"ssh":open_ssh,"help":show_help,"h":show_help}
    if len(sys.argv)>1:
        c=sys.argv[1].lower()
        if c in cmds: cmds[c]()
        return
    while True:
        try:
            c=input("\npolymania> ").strip().lower()
            if c in ("quit","q"): break
            if c in cmds: cmds[c]()
        except: break

if __name__=="__main__": main()

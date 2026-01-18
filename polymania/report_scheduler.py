import logging, time, schedule
from datetime import datetime
from .reports import generate_daily_report, generate_weekly_report
from .backtester import Backtester
from .ml_signals import get_pattern_learner
from .notifier import send_telegram_message
from .config import settings

logger = logging.getLogger('polymania.scheduler')

def send_daily_report():
    try:
        report = generate_daily_report()
        if settings.telegram_bot_token and settings.telegram_chat_id:
            send_telegram_message(report)
            logger.info('Daily report sent')
        else:
            print(report)
    except Exception as e:
        logger.error('Failed to send daily report: ' + str(e))

def send_weekly_report():
    try:
        report = generate_weekly_report()
        
        # Add backtest results
        bt = Backtester()
        signals = bt.load_signals()
        if signals:
            prices = bt.load_price_history()
            result = bt.run_backtest(signals, prices)
            report += chr(10) + chr(10) + bt.format_report(result)
        
        # Add ML insights
        learner = get_pattern_learner()
        report += chr(10) + chr(10) + learner.get_insights()
        
        if settings.telegram_bot_token and settings.telegram_chat_id:
            send_telegram_message(report)
            logger.info('Weekly report sent')
        else:
            print(report)
    except Exception as e:
        logger.error('Failed to send weekly report: ' + str(e))

def run_scheduler():
    logger.info('Starting report scheduler')
    
    # Daily at 20:00 UTC
    schedule.every().day.at('20:00').do(send_daily_report)
    
    # Weekly on Sunday at 20:00 UTC
    schedule.every().sunday.at('20:00').do(send_weekly_report)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
    
    if '--daily' in sys.argv:
        send_daily_report()
    elif '--weekly' in sys.argv:
        send_weekly_report()
    else:
        run_scheduler()

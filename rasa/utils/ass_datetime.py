from datetime import date
import time

def get_date(inquire_date=None):
    
    lookups = {'大前天': -3, '前天': -2, '昨天': -1, '今天': 0, '明天': 1, '后天': 2, '大后天': 3}
    week_zh = ['一', '二', '三', '四','五', '六', '七']
    day_delta = lookups.get(inquire_date, 0)
    print('day_delta', day_delta)
    today_date = date.today()
    inquire_date = date(today_date.year, today_date.month, today_date.day+day_delta)
    inquire_week = inquire_date.weekday()
    
    return (f'{inquire_date.year}年{inquire_date.month}月{inquire_date.day}日 星期{week_zh[inquire_week]}')


def get_time():
    tm = time.localtime()
    return f'{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}'

if __name__ == '__main__':
    pass
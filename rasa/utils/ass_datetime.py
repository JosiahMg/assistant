from datetime import date, datetime, timedelta
import time

def get_date_by_entity(inquire_date=None):
    '''
    function: 给定实体,如果在lookups中则返回字符串格式的日期
    input: inquire_date in ['大前天', '前天', ...]
    output: strftime(string类型)的时间
    '''
    lookups = {'大前天': -3, '前天': -2, '昨天': -1, '今天': 0, '明天': 1, '后天': 2, '大后天': 3}
    # week_zh = ['一', '二', '三', '四','五', '六', '七']
    day_delta = lookups.get(inquire_date, 0)
    today_date = date.today()
    inquire_date = date(today_date.year, today_date.month, today_date.day+day_delta)
    # inquire_week = inquire_date.weekday()
    # return (f'{inquire_date.year}年{inquire_date.month}月{inquire_date.day}日 星期{week_zh[inquire_week]}')
    return inquire_date.strftime('%Y-%m-%d %H:%M:%S %A')

def get_datetime(value):
    '''
    function:
        使用字符串格式的时间生成datetime格式
    input:
        datetime_str: "2021-03-02T00:00:00.000+08:00"
    output:
        datetime()
    '''
    if isinstance(value, str):
        value = value[:19]
        inquire_date = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S')
        return inquire_date
    else:
        return datetime.now()

def get_date_by_value(value, mode='datetime'):
    '''
    function: 生成'年-月-日 时:分:秒 星期'格式的字符串数据
    input: 输入str或者dict{'to': str, 'from': str}
    output: "2021-03-02 00:00:00 monday"
    '''
    if mode == 'datetime':
        fmt_res = '%Y-%m-%d %H:%M:%S %A'
    elif mode == 'date':
        fmt_res = '%Y-%m-%d'
    if isinstance(value, str):
        inquire_date = get_datetime(value)
    elif isinstance(value, dict):
        date_from = get_datetime(value['from'])
        date_to = get_datetime(value['to'])
        print('date_from', date_from)
        print('date_to', date_to)
        print('datetime.today()', datetime.today())
        if date_from.day == datetime.today().day + 1:
            inquire_date =  date_to
        else:
            inquire_date =  date_from
    else:
        return datetime.now().strftime(fmt_res)
    
    return inquire_date.strftime(fmt_res)

def get_localtime():
    tm = time.localtime()
    return f'{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}'

if __name__ == '__main__':
    print(get_date_by_value(None))
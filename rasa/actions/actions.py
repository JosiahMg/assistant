# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from datetime import datetime
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from utils import ass_datetime


class ActionTellDate(Action):
    '''
    #TODO: 指定日期询问星期
    function: 询问日期的动作
    '''
    def name(self) -> Text:
        return "action_tell_date"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # 首先判断是否有DIETClassifier识别的实体 用于获取date的实体(大后天 大前天)
        entity_date = next(tracker.get_latest_entity_values("alien_date"), None)
        if entity_date:
            dispatcher.utter_message(text=ass_datetime.get_date_by_entity(entity_date))
        else:
            value_date = next(tracker.get_latest_entity_values("time"), None) # DucklingEntityExtractor
            dispatcher.utter_message(text=ass_datetime.get_date_by_value(value_date))

        return []

class ActionDateDifferent(Action):
    def name(self) -> Text:
        return "action_date_different"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dt_list = []
        
        for dt in tracker.get_latest_entity_values("time"):
            dt_list.append(dt)
        
        if len(dt_list) == 0:
            dispatcher.utter_message(response='utter_un_come_true')
        
        if len(dt_list) == 1:
            d0 = ass_datetime.get_datetime(dt_list[0])
            d1 = datetime.today()
        
        if len(dt_list) == 2:
            d0 = ass_datetime.get_datetime(dt_list[0])
            d1 = ass_datetime.get_datetime(dt_list[1])
        
        if d1 > d0:
            d0, d1 = d1, d0
        days = (d0 - d1).days
        
        dispatcher.utter_message(text=f"{d1.strftime('%Y-%m-%d')} 与 {d0.strftime('%Y-%m-%d')} 相差 {days} 天")
        
        return []


class ActionTellTime(Action):
    def name(self) -> Text:
        return "action_tell_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # 首先判断是否有DIETClassifier识别出Place实体 
        entity_local = next(tracker.get_latest_entity_values("place"), None)
        print('entity_local:', entity_local)
        if entity_local:
            dispatcher.utter_message(text=ass_datetime.get_time_by_entity(entity_local))
        else:
            value_date = next(tracker.get_latest_entity_values("time"), None)  # DucklingEntityExtractor
            
            dispatcher.utter_message(text=ass_datetime.get_time_by_value(value_date))

        return []


class ActionTimeDifferent(Action):
    def name(self) -> Text:
        return "action_time_different"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        place_list = []
        
        for dt in tracker.get_latest_entity_values("place"):
            place_list.append(dt)

        dispatcher.utter_message(text=ass_datetime.get_place_time_different(place_list))
        
        return []
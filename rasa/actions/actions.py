# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from utils import ass_datetime


class ActionTellDate(Action):
    #TODO: 指定日期询问星期  三天后几号 5天前
    def name(self) -> Text:
        return "action_tell_date"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # 获取date的实体(今天 明天 后天 ...)
        inquired_date = next(tracker.get_latest_entity_values("date"), None)
        dispatcher.utter_message(text=ass_datetime.get_date(inquired_date))

        return []

class ActionTellTime(Action):
    #TODO: 不同时区的时间  半小时的时间...
    def name(self) -> Text:
        return "action_tell_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text=ass_datetime.get_time())

        return []


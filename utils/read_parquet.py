import pandas as pd


def read_balances() -> pd.DataFrame:
    df = pd.read_parquet("data/02 Обортно-сальдовая ведомость ЛС ХК.parquet")
    return df

def read_payments() -> pd.DataFrame:
    df = pd.read_parquet("data/03 Оплаты ХК.parquet")
    return df

def read_general_information() -> pd.DataFrame:
    df = pd.read_parquet("data/01 Общая информация о ЛС ХК.parquet")
    return df


def read_actions() -> pd.DataFrame:
    df = pd.read_parquet("data/Меры.parquet")
    return df


information_actions = ['Автодозвон', 'E-mail', 'СМС', 'Обзвон оператором', 'Уведомление о введении ограничения', 'Выезд к абоненту', 'Заявление о выдаче судебного приказа']
restriction_actions = ['Ограничение']
court_actions = ['Получение судебного приказа или ИЛ']


# нулевой тип - "nothing"
action_type = {
    'Автодозвон':  "informing", 
    'E-mail':  "informing",
    'СМС':  "informing",
    'Обзвон оператором':  "informing",
    'Уведомление о введении ограничения':  "informing",
    'Выезд к абоненту':  "informing",
    'Заявление о выдаче судебного приказа':  "informing",
    'Претензия': "informing",
    'Ограничение': "restriction",
    'Получение судебного приказа или ИЛ': "court",
}

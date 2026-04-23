from typing import Dict, Tuple

import pandas as pd
import os

def read_balances() -> pd.DataFrame:
    df = pd.read_excel("data/02 Обортно-сальдовая ведомость ЛС ХК.xlsx", header=[0, 1], decimal=',')
    
    id_col = df.columns[0]
    df.rename(columns={id_col: ("ЛС", "")})
    
    # 1. первый столбец → id
    # 1. берём первую колонку (как есть)
    id_col = df.columns[0]

    # 2. делаем её индексом
    df = df.set_index(id_col)

    # 3. просто задаём имя индекса
    df.index.name = "ЛС"

    # 2. маппинг показателей
    metric_map = {
        "СЗ на начало": "start",
        "Начислено": "accr",
        "Оплачено": "paid",
        "Опалачено": "paid",  # на случай опечатки
    }

    # 3. преобразуем колонки
    new_cols = []
    for date, metric in df.columns:
        # переводим дату → номер месяца
        date = pd.to_datetime(date)
        year = pd.to_datetime(date).year
        month = pd.to_datetime(date).month
        
        # переводим метрику → короткое имя
        metric_short = metric_map.get(metric.strip(), metric.strip())
        
        new_cols.append(f"{year}_{month}_{metric_short}")

    df.columns = new_cols

    # если нужно вернуть id как колонку
    df = df.reset_index()

    return df

def read_payments() -> pd.DataFrame:
    df = pd.read_csv("data/03 Оплаты ХК.csv", sep=";",  decimal=",")
    return df

def read_general_information() -> pd.DataFrame:
    df = pd.read_excel("data/01 Общая информация о ЛС ХК.xlsx", header=0)
    df.drop(columns=["Адрес (ГУИД)"], inplace=True)
    return df

information_actions = ['Автодозвон', 'E-mail', 'СМС', 'Обзвон оператором', 'Уведомление о введении ограничения', 'Выезд к абоненту', 'Заявление о выдаче судебного приказа']
restriction_actions = ['Ограничение']
court_actions = ['Получение судебного приказа или ИЛ']

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

def read_actions() -> Dict[str, Dict[str,pd.DataFrame]]:
    

    # путь к главному файлу
    main_file = "data/14 Лимиты мер воздействия ХК.xlsx"

    # читаем главный файл
    limits_df = pd.read_excel(main_file)

    # словарь для результатов
    result = {}

    for _, row in limits_df.iterrows():
        file_name = row.iloc[0]
        limit = row.iloc[1]

        # пропускаем пустые строки
        if pd.isna(file_name):
            continue

        file_path = os.path.join("data", file_name+".xlsx")

        # читаем файл без заголовков
        df_raw = pd.read_excel(file_path, header=None)

        # 1 строка — название операции
        operation_name = df_raw.iloc[0, 0]

        # 2 строка — заголовки
        df = df_raw.iloc[2:].copy()
        df.columns = df_raw.iloc[1]

        # чистка: убираем #Н/Д
        df = df[df["ЛС"].notna()]

        # сохраняем
        result[operation_name] = {
            "limit": limit,
            "data": df,
            "stage": action_type[operation_name]
        }
    return result
    # теперь result — словарь с данными


    
    
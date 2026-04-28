"""
    Скрипт, перекодирущий набор данных из csv и xlsx в parquet.
    Так данные хранятся в более сжатом виде и их загрузка происходит быстрее.
"""

import pandas as pd
from utils.general_information import read_actions, read_balances, read_general_information, read_payments

def reform_balances():
    """
        Перекодирует оборотно-сальдовую ведомость. 
        Удаляет три строчки с пропущенными ЛС
    """
    df = read_balances()
    df = df[df["ЛС"].notna()]
    df.to_parquet('data/02 Обортно-сальдовая ведомость ЛС ХК.parquet')
    
    
def reform_payments():
    """
        Перекодирует табличку с оплатами. 
        Превращает столбец "Дата оплаты" в тип DateTime.
        Переименовывает столбец "Номер" в "ЛС".
        Столбец "Способ оплаты" удаляет.
    """
    df = read_payments()
    df.rename(columns={"Номер":"ЛС"}, inplace=True)
    df.drop(columns=["Способ оплаты"], inplace=True)
    
    df["Дата оплаты"] = pd.to_datetime(df["Дата оплаты"], dayfirst=True, errors="coerce")
    df.to_parquet('data/03 Оплаты ХК.parquet')
    
def reform_general_information():
    """
        Перекодирует общую информацию о ЛС.
        Удаляет строчки с NaN в ЛС. Удаляет столбец GUID.
        Перекодирует текстовые столбцы из "да" "нет" в 0 и 1.
    """
    df = read_general_information()
    cols = list(set(df.columns) - set(["ЛС"]))

    df[cols] = (df[cols] == "Да").astype(float)
    
    df.to_parquet('data/01 Общая информация о ЛС ХК.parquet')
    
def reform_actions():
    """
        Перекодирует табличку всех действий. 
        На выходе табличка с столбцами:
        ЛС; Мера; Дата; Стадия
    """
    
    df = read_actions()
    rows = []
    for action, data in df.items():
        for i,row in data["data"].iterrows():
            rows.append({"ЛС": row["ЛС"],
                            "Дата": row["Дата"],
                            "Стадия": data["stage"],
                            "Мера": action})
    df = pd.DataFrame(rows)
    df.to_parquet("data/Меры.parquet")
    
    
def reform_data():
    """
    Перекодирует в формат parquet все необходимые таблицы: оборотно-сальдовую ведомость, историю оплат, историю применённых мер и общую информацию.
    """
    print("Обработка действий...")
    reform_actions()
    print("Обработка оборотно-сальдовой ведомости...")
    reform_balances()
    print("Обработка общей информации...")
    reform_general_information()
    print("Обработка оплат...")
    reform_payments()
    print("Обработка успешно завершена!")
    
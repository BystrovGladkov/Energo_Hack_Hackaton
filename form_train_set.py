import pandas as pd
import numpy as np
import re
from form_time_features import extract_payment_features, calculate_complex_features, get_seasonality_features, actions_features
from general_information import read_balances, read_general_information, read_actions

def build_master_dataset(time_pay: int,
                         start_date = pd.to_datetime('2025-04-01'), end_date = pd.Timestamp.today()) -> pd.DataFrame:
    """
    Создает панельный датасет, скользя по времени с шагом time_pay, собирая признаки на каждую дату.
    
    Параметры:
    pay_df: Таблица платежей
    general_df: Общая таблица (содержит статические признаки)
    time_pay: Шаг сдвига даты и размер окна k (в месяцах)
    end_date: Конечная дата (текущий день) в формате 'YYYY-MM-DD'
    """
    # Количество месяцев, для подсчёта количества применённых ранее мер
    actions_months = 3
    # количество дней для подсчёта силы меры
    repay_days = 14

    # Читаем сальдовую ведомость и удаляем из неё нулевые строки
    balances = read_balances()
    balances = balances[balances["ЛС"].notna()]
    cols_to_check = balances.columns.drop('ЛС')
    balances = balances[(balances[cols_to_check] != 0).any(axis=1)]

    # Все другие таблицы должны соответствовать данным id.
    ids = balances['ЛС']

    # Читаем платёжную таблицу. Удаляем лишние id.
    df_pay = pd.read_csv("data/03 Оплаты ХК.csv", sep=";", decimal=",")
    df_pay = df_pay[df_pay['Номер'].isin(ids)]

    # Читаем информацию с булевыми признаками. Удаляем лишние id.
    general_df = read_general_information()
    general_df = general_df[general_df['ЛС'].isin(ids)]

    actions_df = read_actions()

    # Начало формирования признаков.
    snapshot_date = start_date
    all_snapshots = []
    print(f"Начинаем сборку панельных данных с шагом {time_pay} мес. от {snapshot_date.strftime('%Y-%m-%d')} до {end_date.strftime('%Y-%m-%d')}")
    
    # Скользим окном по времени
    while snapshot_date <= end_date:
        print(f"Обработка среза на дату: {snapshot_date.strftime('%Y-%m-%d')}...")
        
        # Признаки по платежам 
        df_pay_feats = extract_payment_features(df_pay, k=3, current_date=snapshot_date)
            
        # Сложные сальдовые признаки и долг
        df_complex_feats = calculate_complex_features(df_pay, balances, k=time_pay, curr_date=snapshot_date)
        
        # Признаки сезонности 
        df_season = get_seasonality_features(snapshot_date)
        
        # Признаки действий
        df_complex_feats = actions_features(df_complex_feats, actions_df, df_pay, balances, snapshot_date, actions_months, repay_days)
        
        df_complex_feats.rename(columns={"ЛС": "Id"}, inplace=True)
        # --- Объединение признаков ---
        # Соединяем платежи и сальдо по Id
        merged_snapshot = pd.merge(df_complex_feats, df_pay_feats, on='Id', how='inner')
        
        # Добавляем колонку с датой среза 
        merged_snapshot['Snapshot_Date'] = snapshot_date
        
        # Размножаем признаки сезонности (так как они одинаковы для всех ЛС в этот день)
        for col in df_season.columns:
            merged_snapshot[col] = df_season.iloc[0][col]
            
        # Сохраняем срез в список
        all_snapshots.append(merged_snapshot)
        
        # Шагаем вперед на time_pay месяцев
        snapshot_date += pd.DateOffset(months=time_pay)
        
    # Соединяем все временные срезы в одну огромную таблицу
    master_df = pd.concat(all_snapshots, ignore_index=True)
    
    # Приклеиваем статические признаки из general_df ко всем строкам
    master_df = pd.merge(master_df, general_df, left_on='Id', right_on='ЛС', how='left').drop(columns=['ЛС'])
    
    print(f"Сборка завершена! Размер датасета: {master_df.shape}")
    return master_df
import pandas as pd
import numpy as np
import re
from form_time_features import extract_payment_features, calculate_complex_features, get_seasonality_features, actions_features

from form_time_features import calculate_complex_features_actions_based, actions_features_dateless, compute_success
from general_information import read_balances, read_general_information, read_actions
import random

def build_dataset_random_sampling(pay_df, balances, actions_info, start_date, end_date, N=50000):
    """
    Собирает датасет случайным сэмплированием дат до достижения лимита N.
    """
    master_rows = []
    days_diff = (end_date - start_date).days
    
    print(f"Начинаем случайный сбор данных (Цель: {N} записей)...")
    
    # Пока не набрали нужное количество строк
    while len(master_rows) < N:
        # Выбираем случайную дату в заданном диапазоне
        random_days = random.randint(0, days_diff)
        curr_dt = start_date + pd.Timedelta(days=random_days)
        print(f"Срез на дату: {curr_dt.strftime('%Y-%m-%d')} | Собрано: {len(master_rows)}/{N}")
        
        # Считаем комплексные фичи на эту дату (для всех клиентов)
        complex_feats = calculate_complex_features(pay_df, balances, k=3, curr_date=curr_dt)
        
        # Переименовываем Id в ЛС для совместимости с вашим actions_features
        if 'Id' in complex_feats.columns:
            complex_feats = complex_feats.rename(columns={'Id': 'ЛС'})
            
        # Рассматриваем только тех, у кого долг > 0
        complex_feats = complex_feats[complex_feats['Current_Debt'] > 0]
            
        if complex_feats.empty:
            continue
            
        act_feats = actions_features(complex_feats, actions_info, pay_df, balances, check_date=curr_dt, k_months=3)
        
        # Выделяем колонки с таймерами мер
        days_since_cols = [c for c in act_feats.columns if c.startswith('days_since_')]
        
        # Проходим по клиентам в этом срезе и определяем Treatment
        for _, row in act_feats.iterrows():
            ls = row['ЛС']
            
            # Ищем, есть ли хотя бы одна мера, где days_since_{action} == 0
            applied_today = [col for col in days_since_cols if row[col] == 0]
            
            if not applied_today:
                # Ни одна мера сегодня не применялась
                t_assigned = 0
                row_dict = row.to_dict()
                row_dict['curr_date'] = curr_dt
                row_dict['Treatment'] = t_assigned
                master_rows.append(row_dict)
            
            # Жесткий выход, если достигли лимита
            if len(master_rows) >= N:
                break

    print(f"\nСбор завершен! Собрано ровно {len(master_rows)} записей.")
    return pd.DataFrame(master_rows)

def build_master_dataset(time_pay: int,
                         start_date = pd.to_datetime('2025-04-01'), end_date = pd.Timestamp.today()) -> pd.DataFrame:
    """
    Создает панельный датасет, скользя по времени с шагом time_pay, собирая признаки на каждую дату.
    
    Параметры:
    pay_df: Таблица платежей
    general_df: Общая таблица (содержит статические признаки)
    time_pay: Время в днях, в которые ожидается выплата долга
    end_date: Конечная дата (текущий день) в формате 'YYYY-MM-DD'
    """
    # Количество месяцев, для подсчёта количества применённых ранее мер
    actions_months = 3
    # количество дней для подсчёта силы меры

    # Читаем сальдовую ведомость и удаляем из неё нулевые строки
    balances = read_balances()
    balances = balances[balances["ЛС"].notna()]
    cols_to_check = balances.columns.drop('ЛС')
    balances = balances[(balances[cols_to_check] != 0).any(axis=1)]

    # Все другие таблицы должны соответствовать данным id.
    ids = balances['ЛС']

    ids = ids[:1000]
    balances = balances[balances["ЛС"].isin(ids)]

    # Читаем платёжную таблицу. Удаляем лишние id.
    df_pay = pd.read_csv("data/03 Оплаты ХК.csv", sep=";", decimal=",")
    df_pay = df_pay[df_pay['Номер'].isin(ids)]

    # Читаем информацию с булевыми признаками. Удаляем лишние id.
    general_df = read_general_information()
    general_df = general_df[general_df['ЛС'].isin(ids)]

    actions = read_actions()
    for k in actions.keys():
        d = actions[k]["data"]
        actions[k]["data"] = d[d["ЛС"].isin(ids)]

    master_df = calculate_complex_features_actions_based(df_pay, balances, actions, k=actions_months)

    master_df = actions_features_dateless(master_df, actions, df_pay, balances, k_days=time_pay)

    succes_prior = compute_success(actions, df_pay, balances, k_days=time_pay)
    succes_prior = succes_prior["mean"].to_dict()
        


    cols = [c for c in master_df.columns if c.startswith("success_rate_")]

    for col in cols:
        action = col.replace("success_rate_", "")
        master_df[col] = master_df[col].fillna(succes_prior.get(action, 0))

    seasonal_features = get_seasonality_features(master_df["curr_date"])

    # Соединяем все временные срезы в одну огромную таблицу
    master_df = pd.concat([master_df, seasonal_features], axis=1)
    # master_df = pd.merge(master_df, seasonal_features, left_on='ЛС', right_on='ЛС', how='left')

    # Приклеиваем статические признаки из general_df ко всем строкам
    master_df = pd.merge(master_df, general_df, left_on='ЛС', right_on='ЛС', how='left')
    
    print(f"Сборка завершена! Размер датасета: {master_df.shape}")
    return master_df
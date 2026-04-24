
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import pandas as pd

from form_time_features import extract_payment_features, calculate_complex_features, get_seasonality_features, actions_features

from form_time_features import calculate_complex_features_actions_based, actions_features_dateless, compute_success
from general_information import read_balances, read_general_information, read_actions

def prepare_data(df):
    
    stage_le = LabelEncoder()
    df["current_stage"] = stage_le.fit_transform(df["current_stage"])

    cols = [
        "Возможность дистанционного отключения",
        "Наличие телефона",
        "Наличие льгот",
        "Газификация дома",
        "Город",
        "ЯрОблИЕРЦ квитанция",
        "Почта России квитанция",
        "электронная квитанция",
        "не проживает",
        "ЧД",
        "МКД",
        "Общежитие",
        "Установка Тамбур",
        "Установка опора",
        "Установка в квартире/доме",
        "Установка лестничкая клетка"
    ]

    df[cols] = df[cols].replace({
        "Да": 1,
        "Нет": 0
    })
    df[cols] = df[cols].astype(float)

    le = LabelEncoder()
    mark = le.fit_transform(df["current_action"])
    df["current_action"] = mark

    X = df.drop(columns=["ЛС", "curr_date", "debt_start_date"]) 
    Y = df["target"]
    return X, Y


def build_master_dataset(time_pay: int, current_day:pd.Timestamp) -> pd.DataFrame:
    """
    Создает панельный датасет, скользя по времени с шагом time_pay, собирая признаки на каждую дату.
    
    Параметры:
    time_pay: Время в днях, в которые ожидается выплата долга
    current_day: Текущая дата, в которую мы хотим получить лучшую меру для должников.
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

    measures = list(actions.keys())
    df_ids = pd.DataFrame({'ЛС': ids})

    all_possible_actions = {}

    for t in measures:

        df = df_ids.copy()
        df["Дата"] = current_day
        all_possible_actions[t] = {}
        all_possible_actions[t]["data"] = df
        all_possible_actions[t]["stage"] = actions[t]["stage"]


    master_df = calculate_complex_features_actions_based(df_pay, balances, all_possible_actions, k=actions_months)

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

if __name__ == "__main__":

    outcome_model = xgb.XGBRegressor(
        objective='reg:logistic',
        n_estimators=100,
        random_state=44
    )

    outcome_model.load_model("xgboost.xgb")

    """Дата в которую мы хотим предсказать действия"""
    timestamp = pd.Timestamp("2026-01-04")

    data = build_master_dataset(time_pay=3, current_day=timestamp)
    # data = pd.read_parquet("data/training_data.parquet")

    X,Y = prepare_data(data)

    indexes = data["ЛС"]
    actions = data["current_action"]

    # probability_of_debt_pay = outcome_model.predict(X)
    print("Точность оценки вероятности выплаты долга в течении 14 дней:")
    print(outcome_model.score(X,Y))
    probs = outcome_model.predict(X)

    probs = pd.Series(probs, name="repay_prop")
    results = pd.concat([indexes, actions, probs],axis=1)
    results.groupby(by=["ЛС", "current_action"], as_index=False).agg({"repay_prop": "max"})

    print("Сочетание лучших мер для каждого пользователя и вероятность их выплаты соранены в data/best_treatment.csv")
    print("Набор индексов пользователей образан на 1000 для экономии времени")
    results.to_csv("data/best_treatment.csv", sep=";")
    


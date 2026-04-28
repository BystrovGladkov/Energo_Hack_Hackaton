import pandas as pd
import numpy as np
from utils.general_information import read_balances

def extract_payment_features(payments: pd.DataFrame, k_months: int, current_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Извлекает число платежей за последние k месяцев и число дней с последнего платежа.
    
    Параметры:
    k_months (int): Окно в месяцах для подсчета частоты платежей.
    current_date (pd.Timestamp): Точка отсчета для вычисления давности. 
                                 Если None, берется максимальная дата из датасета.
    """
    
    # Если текущая дата не задана, берём максимальную из датасета
    if current_date is None:
        current_date = payments['Дата оплаты'].max()
    
    # Делаем копию, чтобы не испортить таблицу
    payments = payments.copy()
    # оставляем только оплаты, произведённые до указанной даты
    payments = payments[payments['Дата оплаты'] <= current_date]
    
    # Находим последнюю дату оплаты для каждого клиента
    last_payments = payments.groupby('ЛС')['Дата оплаты'].max().reset_index()
    last_payments['Дней_с_последнего_платежа'] = (current_date - last_payments['Дата оплаты']).dt.days
    
    # Оставляем только нужные столбцы
    recency_df = last_payments[['ЛС', 'Дней_с_последнего_платежа']]
    
    # Определяем границу отсечения дат (k месяцев назад от текущей даты)
    cutoff_date = current_date - pd.DateOffset(months=k_months)
    
    # Фильтруем платежи, попавшие в это временное окно
    recent_data = payments[payments['Дата оплаты'] >= cutoff_date]
    freq_col_name = f'Платежей_за_последние_{k_months}_мес'
    frequency_df = recent_data.groupby('ЛС').size().reset_index(name=freq_col_name)
    
    result_df = pd.merge(recency_df, frequency_df, on='ЛС', how='left')
    
    # Заполняем NaN нулями для клиентов без платежей в окне и переводим в int
    result_df[freq_col_name] = result_df[freq_col_name].fillna(0).astype(int)
    
    return result_df

def get_seasonality_features(dates: pd.Series) -> pd.DataFrame:
    """
    Вычисляет сезонные признаки (отопительный сезон и циклическое время) 
    для серии дат (pd.Series).
    """
    month = dates.dt.month
    day_of_year = dates.dt.dayofyear
    
    days_in_year = 365 + dates.dt.is_leap_year.astype(int)
    
    # Отопительный сезон
    is_heating = month.isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
    
    month_cos = np.cos((month - 1) * (2 * np.pi / 12))
    month_sin = np.sin((month - 1) * (2 * np.pi / 12))
    
    day_cos = np.cos((day_of_year - 1) * (2 * np.pi / days_in_year))
    day_sin = np.sin((day_of_year - 1) * (2 * np.pi / days_in_year))
    
    result_df = pd.DataFrame({
        'Is_Heating_Season': is_heating,
        'Season_Temperature_Cos': month_cos.round(4),
        'Season_Day_Cos': day_cos.round(4),
        'Season_Temperature_Sin': month_sin.round(4),
        'Season_Day_Sin': day_sin.round(4)
    })
    
    result_df.index = dates.index
    
    return result_df
    
    
def prepare_balances(balances):
    records = []

    for col in balances.columns:
        if col == "ЛС":
            continue

        year, month, metric = col.split("_")
        records.append((col, int(year), int(month), metric))

    meta = pd.DataFrame(records, columns=["col", "year", "month", "metric"])

    long = []
    for _, row in meta.iterrows():
        tmp = balances[["ЛС", row["col"]]].copy()
        tmp["year"] = row["year"]
        tmp["month"] = row["month"]
        tmp["metric"] = row["metric"]
        tmp = tmp.rename(columns={row["col"]: "value"})
        long.append(tmp)

    df = pd.concat(long)

    df = df.pivot_table(
        index=["ЛС", "year", "month"],
        columns="metric",
        values="value"
    ).reset_index()

    return df

def compute_success(actions:pd.DataFrame, payments:pd.DataFrame, balances:pd.DataFrame, k_days=14):

    all_actions = list(actions.keys())

    # --- собираем действия ---
    action_rows = []
    for name, info in actions.items():
        tmp = info["data"].copy()
        tmp = tmp.rename(columns={tmp.columns[0]: "ЛС"})
        tmp["date"] = pd.to_datetime(tmp.iloc[:, 1])
        tmp["action"] = name
        tmp["stage"] = info.get("stage", "unknown")
        action_rows.append(tmp[["ЛС", "action", "stage", "date"]])

    actions_df = pd.concat(action_rows, ignore_index=True)

    payments = payments.copy()
    payments["Дата оплаты"] = pd.to_datetime(payments["Дата оплаты"], dayfirst=True, errors='coerce')
    payments.rename(columns={"Номер": "ЛС"}, inplace=True)

    actions_df = actions_df.copy()
    actions_df["date"] = pd.to_datetime(actions_df["date"])

    balances_long = prepare_balances(balances)

    # --- добавим год/месяц к действиям ---
    actions_df["year"] = actions_df["date"].dt.year
    actions_df["month"] = actions_df["date"].dt.month

    # --- джойним стартовый долг ---
    actions_df = actions_df.merge(
        balances_long,
        on=["ЛС", "year", "month"],
        how="left"
    )

    # --- считаем оплаты до действия (в том же месяце) ---
    pay_before = payments.merge(
        actions_df[["ЛС", "date"]],
        on="ЛС",
        how="inner"
    )

    pay_before = pay_before[
        (pay_before["Дата оплаты"] < pay_before["date"]) &
        (pay_before["Дата оплаты"].dt.month == pay_before["date"].dt.month) &
        (pay_before["Дата оплаты"].dt.year == pay_before["date"].dt.year)
    ]

    pay_before_sum = (
        pay_before.groupby(["ЛС", "date"])["Сумма"]
        .sum()
        .rename("paid_before")
        .reset_index()
    )

    actions_df = actions_df.merge(
        pay_before_sum,
        on=["ЛС", "date"],
        how="left"
    )

    actions_df["paid_before"] = actions_df["paid_before"].fillna(0)

    # --- долг на момент действия ---
    actions_df["debt_at_action"] = (
        actions_df["start"] - actions_df["paid_before"]
    ).clip(lower=0)

    # --- оплаты после действия (14 дней) ---
    merged = actions_df.merge(payments, on="ЛС", how="left")

    merged["delta_days"] = (
        merged["Дата оплаты"] - merged["date"]
    ).dt.days

    pay_after = merged[
        (merged["delta_days"] >= 0) &
        (merged["delta_days"] <= k_days)
    ]

    pay_after_sum = (
        pay_after.groupby(["ЛС", "action", "date"])["Сумма"]
        .sum()
        .rename("paid_after")
        .reset_index()
    )

    actions_df = actions_df.merge(
        pay_after_sum,
        on=["ЛС", "action", "date"],
        how="left"
    )

    actions_df["paid_after"] = actions_df["paid_after"].fillna(0)

    # --- success ---
    actions_df["success"] = (
        actions_df["paid_after"] / actions_df["debt_at_action"]
    ).replace([np.inf, -np.inf], 0)

    actions_df["success"] = actions_df["success"].clip(upper=1).fillna(0)

    # --- агрегация по типу действия ---
    result = (
        actions_df.groupby("action")["success"]
        .agg(["mean"])
        .sort_values("mean", ascending=False)
    )

    return result    

def actions_features_dateless(
    user_features: pd.DataFrame,
    actions: dict,
    payments: pd.DataFrame,
    balance: pd.DataFrame,
    k_months=3,
    k_days=14
):

    """
    После генерации признаков нужно заполнить пропуски в succes_rate_

    succes_prior = compute_success(actions, payments, balance)
    succes_prior = succes_prior["mean"].to_dict()

    cols = [c for c in df.columns if c.startswith("success_rate_")]

    for col in cols:
        action = col.replace("success_rate_", "")
        df[col] = df[col].fillna(succes_prior.get(action, 0))
    
        succes_prior - можно вычислять 1 раз.
    """

    uf = user_features.copy()
    uf = uf.rename(columns={"Id": "ЛС", "action": "current_action"})
    uf["curr_date"] = pd.to_datetime(uf["curr_date"])

    # --- старт долга ---
    uf["debt_start_date"] = (
        uf["curr_date"] - pd.to_timedelta(uf["Days_Since_Clearance"], unit="D")
    )

    # --- платежи ---
    payments = payments.copy()
    payments["Дата оплаты"] = pd.to_datetime(payments["Дата оплаты"], dayfirst=True, errors="coerce")
    payments = payments.rename(columns={"Номер": "ЛС"})

    # --- действия ---
    action_rows = []
    for name, info in actions.items():
        tmp = info["data"].copy()
        tmp = tmp.rename(columns={tmp.columns[0]: "ЛС"})
        tmp["date"] = pd.to_datetime(tmp.iloc[:, 1])
        tmp["action"] = name
        tmp["stage"] = info.get("stage", "unknown")
        action_rows.append(tmp[["ЛС", "action", "stage", "date"]])

    actions_df = pd.concat(action_rows, ignore_index=True)

    # --- привязка событий ---
    act_ev = uf.merge(actions_df, on="ЛС", how="left")

    # только прошлые действия
    act_ev = act_ev[act_ev["date"] <= act_ev["curr_date"]]

    # --- внутри долга ---
    act_ev["in_debt"] = act_ev["date"] >= act_ev["debt_start_date"]

    # --- TARGET (vectorized через merge) ---

    pay = payments.rename(columns={"Номер": "ЛС"}).copy()
    pay["Дата оплаты"] = pd.to_datetime(pay["Дата оплаты"], dayfirst=True, errors="coerce")

    # джойним платежи к событиям
    target_df = uf.merge(pay, on="ЛС", how="left")

    # считаем лаг
    target_df["delta"] = (target_df["Дата оплаты"] - target_df["curr_date"]).dt.days

    # оставляем только нужное окно
    target_df = target_df[
        (target_df["delta"] >= 0) &
        (target_df["delta"] <= k_days)
    ]

    # сумма платежей в окне
    paid_after = (
        target_df.groupby(["ЛС", "curr_date"])["Сумма"]
        .sum()
        .rename("paid_after_k_days")
    )

    # добавляем обратно
    uf = uf.merge(paid_after, on=["ЛС", "curr_date"], how="left")

    uf["paid_after_k_days"] = uf["paid_after_k_days"].fillna(0)

    # --- target ---
    uf["Current_Debt"] = uf["Current_Debt"].replace(0, np.nan)

    uf["target"] = uf["paid_after_k_days"] / uf["Current_Debt"]
    uf["target"] = uf["target"].clip(0, 1).fillna(0)

    # --- (1) дни с момента последнего действия ---
    last_actions = (
        act_ev.sort_values("date")
        .groupby(["ЛС", "curr_date", "action"])
        .last()
        .reset_index()
    )

    last_actions["days_since"] = (
        last_actions["curr_date"] - last_actions["date"]
    ).dt.days

    days_since = last_actions.pivot(
        index=["ЛС", "curr_date"],
        columns="action",
        values="days_since"
    )

    days_since.columns = [f"days_since_{c}" for c in days_since.columns]

    # --- (2) текущий этап ---
    stage_order = {"informing": 1, "restriction": 2, "court": 3}
    act_ev["stage_rank"] = act_ev["stage"].map(stage_order)

    current_stage = (
        act_ev.sort_values(["ЛС", "curr_date", "stage_rank", "date"])
        .groupby(["ЛС", "curr_date"])
        .last()
        .reset_index()
    )

    current_stage["days_in_stage"] = (
        current_stage["curr_date"] - current_stage["date"]
    ).dt.days

    stage_info = current_stage[["ЛС", "curr_date", "stage", "days_in_stage"]]
    stage_info = stage_info.rename(columns={"stage": "current_stage"})

    # --- (3) число действий ---
    cutoff = uf["curr_date"] - pd.DateOffset(months=k_months)

    act_ev["recent"] = act_ev["date"] >= act_ev["curr_date"] - pd.DateOffset(months=k_months)

    actions_in = (
        act_ev[act_ev["recent"] & act_ev["in_debt"]]
        .groupby(["ЛС", "curr_date"])
        .size()
        .rename("actions_last_km_in_debt")
    )

    actions_out = (
        act_ev[act_ev["recent"] & ~act_ev["in_debt"]]
        .groupby(["ЛС", "curr_date"])
        .size()
        .rename("actions_last_km_out_debt")
    )

    # --- (4) success ---
    pay_ev = act_ev.merge(payments, on="ЛС", how="left")

    pay_ev["delta"] = (pay_ev["Дата оплаты"] - pay_ev["date"]).dt.days

    # оставляем только платежи после действия
    pay_ev = pay_ev[
        (pay_ev["delta"] >= 0) &
        (pay_ev["delta"] <= k_days) &
        (pay_ev["Дата оплаты"] <= pay_ev["curr_date"])
    ]

    # 👉 был ли хотя бы один платёж после действия
    pay_ev["has_payment"] = 1

    success_flag = (
        pay_ev.groupby(["ЛС", "curr_date", "action"])["has_payment"]
        .max()   # если был хоть один → 1
    )

    # теперь считаем среднее по действиям
    success = (
        success_flag
        .groupby(["ЛС", "curr_date", "action"])
        .mean()   # по сути доля успешных применений
        .unstack()
    )

    success.columns = [f"success_rate_{c}" for c in success.columns]

    # --- сборка ---
    features = uf.set_index(["ЛС", "curr_date"])

    for df_ in [days_since, stage_info.set_index(["ЛС", "curr_date"]), actions_in, actions_out, success]:
        features = features.join(df_, how="left")

    # --- fill ---
    days_cols = [c for c in features.columns if c.startswith("days_since_")]
    features[days_cols] = features[days_cols].fillna(-9999)

    features["current_stage"] = features["current_stage"].fillna("nothing")
    features["days_in_stage"] = features["days_in_stage"].fillna(features["Days_Since_Clearance"])

    features[["actions_last_km_in_debt", "actions_last_km_out_debt"]] = \
        features[["actions_last_km_in_debt", "actions_last_km_out_debt"]].fillna(0)

    return features.reset_index()
    
    
def actions_features(user_features:pd.DataFrame, actions:pd.DataFrame, payments:pd.DataFrame, balance: pd.DataFrame, check_date:pd.Timestamp, k_months=3, k_days=14):
    """
        Требует чтобы в user_features были столбцы Days_Since_Clearance и сумма долга Current_Debt
    
    Вычисляет признаки: 
        1) Сколько дней назад применялась мера A после появления долга?
        2) Время с момента перехода на текущий этап (этапы: информирование, ограничение, суд)
        3) Число применённых мер за последние k месяцев
        4) Количество успешных реакций после меры A (поступление платежа произошло через неделю после меры). Измеряем суммированием долей платежа от долга, а не просто 1 или 0.
        5) Текущий этап клиента (информирование, ограничение, суд)
    """
    check_date = pd.to_datetime(check_date)
    
    # --- старт долга ---
    uf = user_features.copy()
    uf.rename(columns={"Id": "ЛС"}, inplace=True)
    uf["debt_start_date"] = check_date - pd.to_timedelta(uf["Days_Since_Clearance"], unit="D")

    # --- платежи ---
    payments = payments.copy()
    payments["Дата оплаты"] = pd.to_datetime(payments["Дата оплаты"], dayfirst=True, errors='coerce')


    all_actions = list(actions.keys())

    # --- собираем действия ---
    action_rows = []
    for name, info in actions.items():
        tmp = info["data"].copy()
        tmp = tmp.rename(columns={tmp.columns[0]: "ЛС"})
        tmp["date"] = pd.to_datetime(tmp.iloc[:, 1])
        tmp["action"] = name
        tmp["stage"] = info.get("stage", "unknown")
        action_rows.append(tmp[["ЛС", "action", "stage", "date"]])

    actions_df = pd.concat(action_rows, ignore_index=True)

    # --- TARGET ---
    pay_after = payments.rename(columns={"Номер": "ЛС"}).copy()

    pay_after = pay_after[
        (pay_after["Дата оплаты"] >= check_date) &
        (pay_after["Дата оплаты"] <= check_date + pd.Timedelta(days=k_days))
    ]

    pay_after_sum = (
        pay_after.groupby("ЛС")["Сумма"]
        .sum()
        .rename("paid_after_k_days")
    )

    # добавляем в фичи
    uf = uf.merge(pay_after_sum, on="ЛС", how="left")

    uf["paid_after_k_days"] = uf["paid_after_k_days"].fillna(0)

    # защита от деления на 0
    uf["Current_Debt"] = uf["Current_Debt"].replace(0, np.nan)

    # target
    uf["target"] = uf["paid_after_k_days"] / uf["Current_Debt"]
    uf["target"] = uf["target"].clip(lower=0, upper=1).fillna(0)

    # Обрезаем и оставляем только историческую информацию
    actions_df = actions_df[actions_df["date"] <= check_date]

    actions_df_all = actions_df.copy()

    # --- привязываем старт долга ---
    actions_df = actions_df.merge(
        uf[["ЛС", "debt_start_date"]],
        on="ЛС",
        how="left"
    )

    # ❗ оставляем только действия внутри текущего долга
    actions_df = actions_df[actions_df["date"] >= actions_df["debt_start_date"]]
    

    # --- то же для платежей ---
    payments = payments.rename(columns={"Номер": "ЛС"})
    payments = payments.merge(
        uf[["ЛС", "debt_start_date"]],
        on="ЛС",
        how="left"
    )
    payments = payments[payments["Дата оплаты"] >= payments["debt_start_date"]]

    # --- merge действия + оплаты ---
    merged = actions_df.merge(payments, on="ЛС", how="left")
    merged = merged[merged["Дата оплаты"] >= merged["date"]]

    merged["delta_days"] = (merged["Дата оплаты"] - merged["date"]).dt.days

    # --- (4) успехи ---
    # добавляем долг
    merged = merged.merge(
        uf[["ЛС", "Current_Debt"]],
        on="ЛС",
        how="left"
    )

    # защита от деления на 0
    merged["Current_Debt"] = merged["Current_Debt"].replace(0, np.nan)

    # считаем нормированный успех
    merged["success"] = np.where(
        merged["delta_days"] <= k_days, # TODO: константа на сколько дней смотрим успешность действия
        merged["Сумма"] / merged["Current_Debt"],
        0
    )

    # можно ограничить сверху (чтобы не было >1)
    merged["success"] = merged["success"].clip(upper=1)

    success = (
        merged.groupby(["ЛС", "action"])["success"]
        .sum()
        .unstack(fill_value=pd.NA)
        .reindex(columns=all_actions, fill_value=pd.NA)
    )
    
    success.columns = [
        f"success_rate_{col}" for col in success.columns
    ]
    

    # --- (1) дни с момента меры (в рамках долга!) ---
    last_actions = (
        actions_df.sort_values("date")
        .groupby(["ЛС", "action"])
        .last()
        .reset_index()
    )

    last_actions["days_since_action"] = (check_date - last_actions["date"]).dt.days

    days_since_action = (
        last_actions.pivot(index="ЛС", columns="action", values="days_since_action")
        .fillna(-9999)
    )
    
    # Делаем так, чтобы столбцы были для всех действий, даже которых нет (не применялись)
    # days_since_action.reindex(columns=all_actions, fill_value=-9999)

    days_since_action.columns = [
        f"days_since_{col}" for col in days_since_action.columns
    ]

    # --- (2) время на текущем этапе ---
    stage_order = {"informing": 1, "restriction": 2, "court": 3}
    actions_df["stage_rank"] = actions_df["stage"].map(stage_order)

    current_stage = (
        actions_df.sort_values(["ЛС", "stage_rank", "date"])
        .groupby("ЛС")
        .last()
    )

    current_stage["days_in_stage"] = (check_date - current_stage["date"]).dt.days

    stage_info = current_stage[["stage", "days_in_stage"]].rename(
        columns={"stage": "current_stage"}
    )

    # --- (3) число мер за k месяцев (НО внутри долга) ---
    cutoff = check_date - pd.DateOffset(months=k_months)

    # все действия за последние k месяцев
    recent_actions_all = actions_df_all[actions_df_all["date"] >= cutoff]

    # --- внутри долга ---
    recent_in_debt = recent_actions_all.merge(
        uf[["ЛС", "debt_start_date"]],
        on="ЛС",
        how="left"
    )

    recent_in_debt = recent_in_debt[
        recent_in_debt["date"] >= recent_in_debt["debt_start_date"]
    ]

    actions_in_debt = (
        recent_in_debt.groupby("ЛС")
        .size()
        .rename(f"actions_last_{k_months}m_in_debt")
    )

    # --- вне долга ---
    recent_out_debt = recent_actions_all.merge(
        uf[["ЛС", "debt_start_date"]],
        on="ЛС",
        how="left"
    )

    recent_out_debt = recent_out_debt[
        recent_out_debt["date"] < recent_out_debt["debt_start_date"]
    ]

    actions_out_debt = (
        recent_out_debt.groupby("ЛС")
        .size()
        .rename(f"actions_last_{k_months}m_out_debt")
    )

    # --- сборка ---
    features = uf.copy()

    features = features.merge(days_since_action, on="ЛС", how="left")
    features = features.merge(stage_info, on="ЛС", how="left")
    features = features.merge(actions_in_debt, on="ЛС", how="left")
    features = features.merge(actions_out_debt, on="ЛС", how="left")
    features = features.merge(success, on="ЛС", how="left")

    days_since_cols = [c for c in features.columns if c.startswith("days_since_")]
    features[days_since_cols] = features[days_since_cols].fillna(value=-9999)

    features["current_stage"] = features["current_stage"].fillna(value="nothing")
    features["days_in_stage"] = features["days_in_stage"].fillna(value=features["Days_Since_Clearance"])
    
    act_counts_cols = ["actions_last_3m_in_debt", "actions_last_3m_out_debt"]
    
    features[act_counts_cols] = features[act_counts_cols].fillna(value=0)

    fill_success = compute_success(actions_df_all, payments, balance, k_days=k_days)
    
    fill_success = fill_success["mean"].to_dict()
    
    cols = [c for c in features.columns if c.startswith("success_rate_")]

    for col in cols:
        action = col.replace("success_rate_", "")
        features[col] = features[col].fillna(fill_success.get(action, 0))

    return features

def get_all_actions(actions: pd.DataFrame):
    all_actions = list(actions.keys())

    # --- собираем действия ---
    action_rows = []
    for name, info in actions.items():
        tmp = info["data"].copy()
        tmp = tmp.rename(columns={tmp.columns[0]: "ЛС"})
        tmp["date"] = pd.to_datetime(tmp.iloc[:, 1])
        tmp["action"] = name
        tmp["stage"] = info.get("stage", "unknown")
        action_rows.append(tmp[["ЛС", "action", "stage", "date"]])

    actions_df = pd.concat(action_rows, ignore_index=True)





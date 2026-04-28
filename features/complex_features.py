import numpy as np
import pandas as pd

# TODO не совпадает вычисление Наклона тренда в этих методах

def calculate_complex_features(payments: pd.DataFrame, balances: pd.DataFrame, k_months: int, curr_date: pd.Timestamp, no_payment_const = 9999) -> pd.DataFrame:
    """
    Вычисляет сложные агрегированные признаки на основе начислений и истории платежей:
    - Время с последнего полного погашения 
    - Доля месяцев, среди последних 12, когда клиент платил хоть сколько-то
    - Количество месяцев, которое длится долг
    - Доля выплат за последние k_month месяцев
    - Наклон линии тренда сальдо за k_month месяцев
    - Величина текущего долга
    - Отношение долга с среднему зачислению за 3 месяца
    - Время с предполагаемых дат зарплаты и аванса (5 и 20 число месяца)
    
    
    Параметры:
    balances: DataFrame с начислениями (столбцы: ЛС, 2025_1_start, 2025_1_accr, 2025_1_paid...)
    payments: DataFrame с платежами (столбцы: ЛС, Дата оплаты, Сумма)
    k_months: Количество месяцев для расчета коэффициента оплат (ratio)
    curr_date: Текущая дата (строка 'YYYY-MM-DD') для отсчета времени
    """
    payments = payments.copy()

    # Добавляем колонки года и месяца для связи таблиц
    payments['Year'] = payments['Дата оплаты'].dt.year
    payments['Month'] = payments['Дата оплаты'].dt.month
    
    # Преобразование сальдовой информации из широкого формата в длинный (Unpivot)
    # Оставляем ЛС индексом и плавим остальные колонки
    melted_balances = balances.melt(id_vars=['ЛС'], var_name='Period', value_name='Value')
    
    # Извлекаем год, месяц и тип метрики из названия колонки (например, '2025_1_start')
    # Используем регулярное выражение для разделения
    extracted = melted_balances['Period'].str.extract(r'(?P<Year>\d{4})_(?P<Month>\d+)_(?P<Metric>[a-zA-Z]+)')
    melted_balances = pd.concat([melted_balances, extracted], axis=1)
    
    # Переводим в числа
    melted_balances['Year'] = melted_balances['Year'].astype(int)
    melted_balances['Month'] = melted_balances['Month'].astype(int)
    
    # Сворачиваем метрики в столбцы: start, accr, paid
    long_df = melted_balances.pivot_table(index=['ЛС', 'Year', 'Month'], 
                                 columns='Metric', 
                                 values='Value', 
                                 aggfunc='first').reset_index()
    
    # Создаем дату на 1 число каждого месяца для удобной фильтрации
    long_df['Period_Date'] = pd.to_datetime(long_df['Year'].astype(str) + '-' + long_df['Month'].astype(str) + '-01')
    
    # Оставляем только те данные, которые были ДО текущей даты (предотвращаем утечку будущего)
    long_df = long_df[long_df['Period_Date'] < curr_date].copy()
    
    # Вычисляем исходящее сальдо на конец месяца
    # Текущие начисления учитывать не нужно. Они платятся в следующий месяц.
    long_df['end_balance'] = long_df['start'] - long_df['paid']
    
    # Подготавливаем итоговый DataFrame
    result_df = pd.DataFrame({'ЛС': long_df['ЛС'].unique().astype(int)})
    
    # --- Дней с последнего полного закрытия сальдо ---
    # Месяц закрытия: исходящее сальдо <= 0
    cleared_months = long_df[long_df['end_balance'] <= 0].copy()
    
    if not cleared_months.empty:
        # Находим самый последний месяц закрытия для каждого ЛС
        last_cleared_month = cleared_months.loc[cleared_months.groupby('ЛС')['Period_Date'].idxmax()]
        
        # Джойним с таблицей платежей, чтобы найти точную дату платежа в этом месяце
        merged_clearance = pd.merge(
            last_cleared_month[['ЛС', 'Year', 'Month']], 
            payments, 
            left_on=['ЛС', 'Year', 'Month'], 
            right_on=['ЛС', 'Year', 'Month'], 
            how='inner'
        )
        
        # Находим последнюю дату платежа в месяце закрытия
        last_clearance_date = merged_clearance.groupby('ЛС')['Дата оплаты'].max().reset_index()
        last_clearance_date['Days_Since_Clearance'] = (curr_date - last_clearance_date['Дата оплаты']).dt.days
        
        result_df = pd.merge(result_df, last_clearance_date[['ЛС', 'Days_Since_Clearance']], on='ЛС', how='left')
    else:
        result_df['Days_Since_Clearance'] = np.nan
        
    # Заполняем пропуски для тех, кто никогда не закрывал долг большим числом
    result_df['Days_Since_Clearance'] = result_df['Days_Since_Clearance'].fillna(no_payment_const) # TODO возможно, тут лучше оставить NaN
    
    # --- Доля месяцев с ненулевой оплатой за последние 12 месяцев ---
    date_12m_ago = curr_date - pd.DateOffset(months=12)
    last_12m_df = long_df[long_df['Period_Date'] >= date_12m_ago]
    
    # Считаем долю (среднее от булевой маски)
    paid_fraction = (
        (last_12m_df['paid'] > 0)
        .groupby(last_12m_df['ЛС'])
        .mean()
        .reset_index(name='Payment_Fraction_12M')
    )
    
    result_df = pd.merge(result_df, paid_fraction, on='ЛС', how='left')
    result_df['Payment_Fraction_12M'] = result_df['Payment_Fraction_12M'].fillna(0)

    # --- Возраст долга (число последних месяцев с долгом подряд) ---
    # Сортируем от самых свежих месяцев к старым
    sorted_long_df = long_df.sort_values(by=['ЛС', 'Period_Date'], ascending=[True, False]).copy()
    
    # Индикатор наличия долга на конец месяца
    sorted_long_df['has_debt'] = (sorted_long_df['end_balance'] > 0).astype(int)
    
    # Как только встретится 0 (долга нет), произведение навсегда станет 0 для более старых месяцев.
    sorted_long_df['debt_streak'] = sorted_long_df.groupby('ЛС')['has_debt'].cumprod()
    
    # Сумма этих произведений и есть число подряд идущих месяцев с долгом от текущего момента в прошлое
    debt_age = sorted_long_df.groupby('ЛС')['debt_streak'].sum().reset_index(name='Consecutive_Debt_Months')
    
    result_df = pd.merge(result_df, debt_age, on='ЛС', how='left')
    result_df['Consecutive_Debt_Months'] = result_df['Consecutive_Debt_Months'].fillna(0).astype(int)

    # --- Сумма оплат / Сумма начислений за последние k месяцев ---
    date_km_ago = curr_date - pd.DateOffset(months=k_months+1)
    last_km_df = long_df[long_df['Period_Date'] >= date_km_ago]
    last_km_sorted = last_km_df.sort_values(by=['ЛС', 'Period_Date'])
    
    # Считаем суммы
    sum_k = last_km_sorted.groupby('ЛС')[['paid', 'accr']].sum().reset_index()

    # Извлекаем первый платёж и последнее начисление для каждого ЛС
    first_paid = last_km_sorted.groupby('ЛС')['paid'].first().reset_index(name='first_paid')
    last_accr = last_km_sorted.groupby('ЛС')['accr'].last().reset_index(name='last_accr')

    # Безопасно объединяем (merge гарантирует, что данные совпадут по нужным ЛС)
    sum_k = pd.merge(sum_k, first_paid, on='ЛС')
    sum_k = pd.merge(sum_k, last_accr, on='ЛС')

    # Корректируем суммы согласно бизнес-логике
    sum_k['paid'] = sum_k['paid'] - sum_k['first_paid']
    sum_k['accr'] = sum_k['accr'] - sum_k['last_accr']

    # Защита от отрицательных значений
    # На случай аномалий в данных (например, если клиент был в выборке всего 1 месяц),
    # чтобы суммы не ушли в минус.
    sum_k['paid'] = sum_k['paid'].clip(lower=0)
    sum_k['accr'] = sum_k['accr'].clip(lower=0)

    # Удаляем вспомогательные колонки
    sum_k = sum_k.drop(columns=['first_paid', 'last_accr'])
    
    # Рассчитываем отношение. Используем np.where для безопасного деления на ноль (если начислений не было)
    sum_k['Payment_Accrual_Ratio_kM'] = np.where(
        sum_k['accr'] > 0, 
        sum_k['paid'] / sum_k['accr'], 
        1 # Если начислений 0, но оплаты были, ставим 1
    )
    
    result_df = pd.merge(result_df, sum_k[['ЛС', 'Payment_Accrual_Ratio_kM']], on='ЛС', how='left')
    result_df['Payment_Accrual_Ratio_kM'] = result_df['Payment_Accrual_Ratio_kM'].fillna(1)

    # --- Наклон линии тренда сальдо за последние k месяцев ---
    def calc_trend(y):
        if len(y) < 2: return 0.0
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    
    trend_df = last_km_sorted.groupby('ЛС')['end_balance'].apply(calc_trend).reset_index(name=f'Balance_Trend_Slope_kM')
    result_df = pd.merge(result_df, trend_df, on='ЛС', how='left')
    result_df[f'Balance_Trend_Slope_kM'] = result_df[f'Balance_Trend_Slope_kM'].fillna(0.0)

    # --- СУММА ДОЛГА НА ТЕКУЩИЙ ДЕНЬ (Current_Debt) ---
    # Извлекаем последний доступный баланс И его дату (1-ое число месяца)
    latest_balance = sorted_long_df.groupby('ЛС').first().reset_index()[['ЛС', 'start', 'Period_Date']]
    past_payments = payments[payments['Дата оплаты'] <= curr_date]
    
    # Привязываем к каждому платежу дату последнего начисления (Period_Date) конкретного ЛС
    pays_with_period = pd.merge(past_payments, latest_balance[['ЛС', 'Period_Date']], on='ЛС', how='inner')
    
    # Оставляем только свежие платежи (от 1-го числа "текущего" месяца до curr_date)
    current_month_pays = pays_with_period[pays_with_period['Дата оплаты'] >= pays_with_period['Period_Date']]
    
    # Суммируем эти платежи
    recent_pays_sum = current_month_pays.groupby('ЛС')['Сумма'].sum().reset_index(name='Recent_Payments')
    
    # Вычитаем недавние платежи из баланса на начало месяца
    debt_df = pd.merge(latest_balance, recent_pays_sum, on='ЛС', how='left')
    debt_df['Recent_Payments'] = debt_df['Recent_Payments'].fillna(0)
    debt_df['Current_Debt'] = debt_df['start'] - debt_df['Recent_Payments']
    
    # Записываем в общий результат
    result_df = pd.merge(result_df, debt_df[['ЛС', 'Current_Debt']], on='ЛС', how='left')

    # --- Текущее сальдо / среднее начисление за последние 3 месяца ---
    latest_balance = sorted_long_df.groupby('ЛС').first().reset_index()[['ЛС', 'end_balance']]
    
    # Считаем среднее начисление за 3 месяца
    date_4m_ago = curr_date - pd.DateOffset(months=4)
    date_1m_ago = curr_date - pd.DateOffset(months=1)
    last_3m_df = long_df[(long_df['Period_Date'] >= date_4m_ago) & (long_df['Period_Date'] < date_1m_ago)]
    avg_accr_3m = last_3m_df.groupby('ЛС')['accr'].mean().reset_index(name='Avg_Accrual_3M')
    
    # Объединяем и считаем ratio
    feat_6_df = pd.merge(latest_balance, avg_accr_3m, on='ЛС', how='left')
    feat_6_df['Avg_Accrual_3M'] = feat_6_df['Avg_Accrual_3M'].fillna(0)
    feat_6_df['Debt_to_Avg_Accrual_3M'] = np.where(
        feat_6_df['Avg_Accrual_3M'] > 0,
        feat_6_df['end_balance'] / feat_6_df['Avg_Accrual_3M'],
        feat_6_df['end_balance'] # Если начислений не было, вернем саму сумму долга как экстремальный показатель
    )
    result_df = pd.merge(result_df, feat_6_df[['ЛС', 'Debt_to_Avg_Accrual_3M']], on='ЛС', how='left')

    # --- Число дней после условных аванса (5 число) и зарплаты (20 число) ---
    # Поскольку current_date едина для всего датафрейма в рамках вызова функции,
    # мы рассчитываем эти значения один раз и присваиваем всем
    def get_days_since_target_day(curr_dt, target_day):
        if curr_dt.day >= target_day:
            last_target = curr_dt.replace(day=target_day)
        else:
            # Если сегодня 3 число, то последнее 5-е было в прошлом месяце
            last_target = (curr_dt - pd.DateOffset(months=1)).replace(day=target_day)
        return (curr_dt - last_target).days

    days_since_5th = get_days_since_target_day(curr_date, 5)
    days_since_20th = get_days_since_target_day(curr_date, 20)
    
    result_df['Days_Since_Advance_5th'] = days_since_5th
    result_df['Days_Since_Salary_20th'] = days_since_20th

    return result_df



def calculate_complex_features_actions_based(
    payments: pd.DataFrame,
    balances: pd.DataFrame,
    actions: pd.DataFrame,
    k: int,
    no_payment_const=9999
) -> pd.DataFrame:
    """
    Вычисляет сложные агрегированные признаки на основе начислений и истории платежей:
    - Время с последнего полного погашения +
    - Доля месяцев, среди последних 12, когда клиент платил хоть сколько-то 
    - Количество месяцев, которое длится долг
    - Доля выплат за последние k_month месяцев
    - Наклон линии тренда сальдо за k_month месяцев
    - Величина текущего долга
    - Отношение долга с среднему зачислению за 3 месяца
    - Время с предполагаемых дат зарплаты и аванса (5 и 20 число месяца)
    
    
    Параметры:
    balances: DataFrame с начислениями (столбцы: ЛС, 2025_1_start, 2025_1_accr, 2025_1_paid...)
    payments: DataFrame с платежами (столбцы: ЛС, Дата оплаты, Сумма)
    k_months: Количество месяцев для расчета коэффициента оплат (ratio)
    """

    payments = payments.copy()
    payments["Year"] = payments["Дата оплаты"].dt.year
    payments["Month"] = payments["Дата оплаты"].dt.month

    actions = actions.copy()
    actions = actions[["ЛС", "Дата", "Мера"]].sort_values(["ЛС", "Дата"])

    # --- balances → long ---
    balances["ЛС"] = balances["ЛС"].astype(int)
    melted_balances = balances.melt(id_vars=["ЛС"], var_name="Period", value_name="Value")
    extracted = melted_balances["Period"].str.extract(r"(?P<Year>\d{4})_(?P<Month>\d+)_(?P<Metric>[a-zA-Z]+)")
    melted_balances = pd.concat([melted_balances, extracted], axis=1)

    melted_balances["Year"] = melted_balances["Year"].astype(int)
    melted_balances["Month"] = melted_balances["Month"].astype(int)

    long_df = melted_balances.pivot_table(
        index=["ЛС", "Year", "Month"],
        columns="Metric",
        values="Value",
        aggfunc="first"
    ).reset_index()

    long_df["Period_Date"] = pd.to_datetime(
        long_df["Year"].astype(str) + "-" + long_df["Month"].astype(str) + "-01"
    )

    long_df["end_balance"] = long_df["start"] - long_df["paid"]

    # --- 🔴 КЛЮЧ: находим месяцы полного закрытия ---
    cleared = long_df[long_df["end_balance"] <= 0][["ЛС", "Year", "Month", "Period_Date"]]

    # --- находим дату последнего платежа в месяце закрытия ---
    clearance_pay = cleared.merge(
        payments,
        on=["ЛС", "Year", "Month"],
        how="left"
    )

    clearance_dates = (
        clearance_pay
        .groupby(["ЛС", "Period_Date"])["Дата оплаты"]
        .max()
        .reset_index()
        .rename(columns={"Дата оплаты": "clearance_date"})
    )

    clearance_dates = clearance_dates.dropna(subset=["clearance_date"])
    clearance_dates["clearance_date"] = pd.to_datetime(
        clearance_dates["clearance_date"], errors="coerce"
    )

    # --- теперь делаем asof join по событиям ---
    events_sorted = actions.sort_values(["Дата", "ЛС"])
    clearance_dates = clearance_dates.sort_values(["clearance_date", "ЛС"])

    last_clearance = pd.merge_asof(
        events_sorted,
        clearance_dates,    
        left_on="Дата",
        right_on="clearance_date",
        by="ЛС",
        direction="backward"
    )

    last_clearance["Days_Since_Clearance"] = (
        last_clearance["Дата"] - last_clearance["clearance_date"]
    ).dt.days

    last_clearance["Days_Since_Clearance"] = last_clearance["Days_Since_Clearance"].fillna(no_payment_const)

    # --- остальные признаки (по той же логике, но через фильтр по curr_date) ---

    # приклеим long_df к событиям
    long_ev = actions.merge(long_df, on="ЛС", how="left")
    long_ev = long_ev[long_ev["Period_Date"] < long_ev["Дата"]]

    # --- Payment_Fraction_12M ---
    mask_12m = long_ev["Period_Date"] >= (long_ev["Дата"] - pd.DateOffset(months=12))
    tmp_12m = long_ev[mask_12m]

    paid_fraction = (
        (tmp_12m["paid"] > 0)
        .groupby([tmp_12m["ЛС"], tmp_12m["Дата"]])
        .mean()
        .reset_index(name="Payment_Fraction_12M")
    )


    # --- debt streak ---
    long_ev = long_ev.sort_values(["ЛС", "Дата", "Period_Date"], ascending=[True, True, False])
    long_ev["has_debt"] = (long_ev["end_balance"] > 0).astype(int)
    long_ev["debt_streak"] = long_ev.groupby(["ЛС", "Дата"])["has_debt"].cumprod()

    debt_age = (
        long_ev.groupby(["ЛС", "Дата"])["debt_streak"]
        .sum()
        .reset_index(name="Consecutive_Debt_Months")
    )

    # --- ratio ---
    mask_k = long_ev["Period_Date"] >= (long_ev["Дата"] - pd.DateOffset(months=k+1))
    tmp_k = long_ev[mask_k].sort_values(["ЛС", "Дата", "Period_Date"])

    sum_k = tmp_k.groupby(["ЛС", "Дата"])[["paid", "accr"]].sum().reset_index()

    sum_k["Payment_Accrual_Ratio_kM"] = np.where(
        sum_k["accr"] > 0,
        sum_k["paid"] / sum_k["accr"],
        1
    )

    # --- Наклон линии тренда сальдо за последние k месяцев ---
    
    def calc_trend(y):
        if len(y) < 2: return 0.0
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    
    trend_df = long_ev.groupby(["ЛС", "Дата"])['end_balance'].apply(calc_trend).reset_index(name=f'Balance_Trend_Slope_kM')

    # --- current debt ---
    latest_balance = (
        long_ev.sort_values(["ЛС", "Дата", "Period_Date"], ascending=[True, True, False])
        .groupby(["ЛС", "Дата"])
        .first()
        .reset_index()[["ЛС", "Дата", "start", "Period_Date","end_balance"]]
    )

    pays = payments.copy()

    pays_ev = actions.merge(pays, on="ЛС", how="left")
    pays_ev = pays_ev[pays_ev["Дата оплаты"] <= pays_ev["Дата"]]

    pays_ev = pays_ev.merge(
        latest_balance[["ЛС", "Дата", "Period_Date"]],
        on=["ЛС", "Дата"],
        how="left"
    )

    pays_ev = pays_ev[pays_ev["Дата оплаты"] >= pays_ev["Period_Date"]]

    recent_pays = (
        pays_ev.groupby(["ЛС", "Дата"])["Сумма"]
        .sum()
        .reset_index(name="Recent_Payments")
    )

    debt = latest_balance.merge(recent_pays, on=["ЛС", "Дата"], how="left")
    debt["Recent_Payments"] = debt["Recent_Payments"].fillna(0)
    debt["Current_Debt"] = debt["start"] - debt["Recent_Payments"]

    # Считаем среднее начисление за 3 месяца
    mask_last_3m = ( 
        (long_ev["Period_Date"] >= (long_ev["Дата"] - pd.DateOffset(months=4)))
        &
        (long_ev["Period_Date"] < (long_ev["Дата"] - pd.DateOffset(months=1)))
    )
    last_3m_df = long_ev[mask_last_3m]
    avg_accr_3m = last_3m_df.groupby(['ЛС', "Дата"])['accr'].mean().reset_index(name='Avg_Accrual_3M')
    
    # Объединяем и считаем ratio
    feat_6_df = pd.merge(latest_balance, avg_accr_3m, on=['ЛС', 'Дата'], how='left')
    feat_6_df['Avg_Accrual_3M'] = feat_6_df['Avg_Accrual_3M'].fillna(0)
    feat_6_df['Debt_to_Avg_Accrual_3M'] = np.where(
        feat_6_df['Avg_Accrual_3M'] > 0,
        feat_6_df['end_balance'] / feat_6_df['Avg_Accrual_3M'],
        feat_6_df['end_balance'] # Если начислений не было, вернем саму сумму долга как экстремальный показатель
    )


    # --- финальная сборка ---
    result = actions.copy()

    for df_ in [
        last_clearance[["ЛС", "Дата", "Days_Since_Clearance"]],
        paid_fraction,
        debt_age,
        sum_k[["ЛС", "Дата", "Payment_Accrual_Ratio_kM"]],
        debt[["ЛС", "Дата", "Current_Debt"]],
        trend_df[["ЛС", "Дата", "Balance_Trend_Slope_kM"]],
        feat_6_df[['ЛС', 'Дата', 'Debt_to_Avg_Accrual_3M']]
    ]:
        result = result.merge(df_, on=["ЛС", "Дата"], how="left")

    result["Payment_Fraction_12M"] = result["Payment_Fraction_12M"].fillna(0)
    result["Consecutive_Debt_Months"] = result["Consecutive_Debt_Months"].fillna(0)
    result["Payment_Accrual_Ratio_kM"] = result["Payment_Accrual_Ratio_kM"].fillna(1)
    result['Balance_Trend_Slope_kM'] = result['Balance_Trend_Slope_kM'].fillna(0.0)


    # --- Число дней после условных аванса (5 число) и зарплаты (20 число) ---
    # Поскольку current_date едина для всего датафрейма в рамках вызова функции,
    # мы рассчитываем эти значения один раз и присваиваем всем
    def days_since_target_day(dates: pd.Series, target_day: int) -> pd.Series:
        dates = pd.to_datetime(dates)

        # день месяца
        day = dates.dt.day

        # дата с тем же месяцем
        this_month_target = dates.dt.to_period("M").dt.to_timestamp() + pd.offsets.Day(target_day - 1)

        # дата в прошлом месяце
        prev_month = (dates - pd.DateOffset(months=1))
        prev_month_target = prev_month.dt.to_period("M").dt.to_timestamp() + pd.offsets.Day(target_day - 1)

        # выбираем
        last_target = np.where(
            day >= target_day,
            this_month_target,
            prev_month_target
        )

        last_target = pd.to_datetime(last_target)

        return (dates - last_target).dt.days



    days_since_5th = days_since_target_day(result["Дата"], 5)
    days_since_20th = days_since_target_day(result["Дата"], 20)
    
    result['Days_Since_Advance_5th'] = days_since_5th
    result['Days_Since_Salary_20th'] = days_since_20th

    # Отбрасываем людей, которым звонили 1 января 2025 года <- мы не знаем их долг
    resilt = result[result["Current_Debt"].notna()]

    return result

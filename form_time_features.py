import pandas as pd
import numpy as np
from general_information import read_balances

def extract_payment_features(data: pd.DataFrame, k: int, current_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Извлекает число платежей за последние k месяцев и число дней с последнего платежа.
    
    Параметры:
    k (int): Окно в месяцах для подсчета частоты платежей.
    current_date (pd.Timestamp): Точка отсчета для вычисления давности. 
                                 Если None, берется максимальная дата из датасета.
    """

    data['Дата оплаты'] = pd.to_datetime(data['Дата оплаты'], dayfirst=True, errors='coerce')
    data = data.dropna(subset=['Дата оплаты'])
    
    # Если текущая дата не задана, берём максимальную из датасета.
    if current_date is None:
        current_date = data['Дата оплаты'].max()
    
    # Находим последнюю дату оплаты для каждого клиента
    last_payments = data.groupby('Номер')['Дата оплаты'].max().reset_index()
    last_payments['Дней_с_последнего_платежа'] = (current_date - last_payments['Дата оплаты']).dt.days
    
    # Оставляем только нужные столбцы
    recency_df = last_payments[['Номер', 'Дней_с_последнего_платежа']]
    
    # Определяем границу отсечения дат (k месяцев назад от текущей даты)
    cutoff_date = current_date - pd.DateOffset(months=k)
    
    # Фильтруем платежи, попавшие в это временное окно
    recent_data = data[data['Дата оплаты'] >= cutoff_date]
    freq_col_name = f'Платежей_за_последние_{k}_мес'
    frequency_df = recent_data.groupby('Номер').size().reset_index(name=freq_col_name)
    
    result_df = pd.merge(recency_df, frequency_df, on='Номер', how='left')
    
    # Заполняем NaN нулями для клиентов без платежей в окне и переводим в int
    result_df[freq_col_name] = result_df[freq_col_name].fillna(0).astype(int)
    result_df.rename(columns={'Номер': 'Id'})
    
    return result_df

def calculate_complex_features(pay_df: pd.DataFrame, gen_info: pd.DataFrame, k: int, curr_date: pd.Timestamp, no_payment_const = 9999) -> pd.DataFrame:
    """
    Вычисляет сложные агрегированные признаки на основе начислений и истории платежей.
    
    Параметры:
    gen_info: DataFrame с начислениями (столбцы: ЛС, 2025_1_start, 2025_1_accr, 2025_1_paid...)
    payments: DataFrame с платежами (столбцы: Номер, Дата оплаты, Сумма, Способ оплаты)
    k: Количество месяцев для расчета коэффициента оплат (ratio)
    current_date: Текущая дата (строка 'YYYY-MM-DD') для отсчета времени
    """
    # Подготовка таблиц и приведение типов
    pay_df['Дата оплаты'] = pd.to_datetime(pay_df['Дата оплаты'], dayfirst=True, errors='coerce')
    pay_df = pay_df.dropna(subset=['Дата оплаты'])

    # Добавляем колонки года и месяца для связи таблиц
    pay_df['Year'] = pay_df['Дата оплаты'].dt.year
    pay_df['Month'] = pay_df['Дата оплаты'].dt.month
    
    # Преобразование сальдовой информации из широкого формата в длинный (Unpivot)
    # Оставляем ЛС индексом и плавим остальные колонки
    melted = gen_info.melt(id_vars=['ЛС'], var_name='Period', value_name='Value')
    
    # Извлекаем год, месяц и тип метрики из названия колонки (например, '2025_1_start')
    # Используем регулярное выражение для разделения
    extracted = melted['Period'].str.extract(r'(?P<Year>\d{4})_(?P<Month>\d+)_(?P<Metric>[a-zA-Z]+)')
    melted = pd.concat([melted, extracted], axis=1)
    
    # Переводим в числа
    melted['Year'] = melted['Year'].astype(int)
    melted['Month'] = melted['Month'].astype(int)
    
    # Сворачиваем метрики в столбцы: start, accr, paid
    long_df = melted.pivot_table(index=['ЛС', 'Year', 'Month'], 
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
    result_df = pd.DataFrame({'Id': long_df['ЛС'].unique().astype(int)})
    
    # --- Дней с последнего полного закрытия сальдо ---
    # Месяц закрытия: исходящее сальдо <= 0
    cleared_months = long_df[long_df['end_balance'] <= 0].copy()
    
    if not cleared_months.empty:
        # Находим самый последний месяц закрытия для каждого ЛС
        last_cleared_month = cleared_months.loc[cleared_months.groupby('ЛС')['Period_Date'].idxmax()]
        
        # Джойним с таблицей платежей, чтобы найти точную дату платежа в этом месяце
        merged_clearance = pd.merge(
            last_cleared_month[['ЛС', 'Year', 'Month']], 
            pay_df, 
            left_on=['ЛС', 'Year', 'Month'], 
            right_on=['Номер', 'Year', 'Month'], 
            how='inner'
        )
        
        # Находим последнюю дату платежа в месяце закрытия
        last_clearance_date = merged_clearance.groupby('ЛС')['Дата оплаты'].max().reset_index()
        last_clearance_date['Days_Since_Clearance'] = (curr_date - last_clearance_date['Дата оплаты']).dt.days
        
        result_df = pd.merge(result_df, last_clearance_date[['ЛС', 'Days_Since_Clearance']], 
                             left_on='Id', right_on='ЛС', how='left').drop(columns=['ЛС'])
    else:
        result_df['Days_Since_Clearance'] = np.nan
        
    # Заполняем пропуски для тех, кто никогда не закрывал долг большим числом
    result_df['Days_Since_Clearance'] = result_df['Days_Since_Clearance'].fillna(no_payment_const)
    
    # --- Доля месяцев с ненулевой оплатой за последние 12 месяцев ---
    date_12m_ago = curr_date - pd.DateOffset(months=12)
    last_12m_df = long_df[long_df['Period_Date'] >= date_12m_ago]
    
    # Считаем долю (среднее от булевой маски)
    paid_fraction = last_12m_df.groupby('ЛС').apply(
        lambda x: (x['paid'] > 0).mean()
    ).reset_index(name='Payment_Fraction_12M')
    
    result_df = pd.merge(result_df, paid_fraction, left_on='Id', right_on='ЛС', how='left').drop(columns=['ЛС'])
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
    
    result_df = pd.merge(result_df, debt_age, left_on='Id', right_on='ЛС', how='left').drop(columns=['ЛС'])
    result_df['Consecutive_Debt_Months'] = result_df['Consecutive_Debt_Months'].fillna(0).astype(int)

    # --- Сумма оплат / Сумма начислений за последние k месяцев ---
    date_km_ago = curr_date - pd.DateOffset(months=k+1)
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
    
    result_df = pd.merge(result_df, sum_k[['ЛС', 'Payment_Accrual_Ratio_kM']], 
                         left_on='Id', right_on='ЛС', how='left').drop(columns=['ЛС'])
    result_df['Payment_Accrual_Ratio_kM'] = result_df['Payment_Accrual_Ratio_kM'].fillna(1)

    # --- Наклон линии тренда сальдо за последние k месяцев ---
    def calc_trend(y):
        if len(y) < 2: return 0.0
        x = np.arange(len(y))
        # np.polyfit возвращает [slope, intercept]
        return np.polyfit(x, y, 1)[0]
    
    trend_df = last_km_sorted.groupby('ЛС')['end_balance'].apply(calc_trend).reset_index(name=f'Balance_Trend_Slope_{k}M')
    result_df = pd.merge(result_df, trend_df, left_on='Id', right_on='ЛС', how='left').drop(columns=['ЛС'])
    result_df[f'Balance_Trend_Slope_{k}M'] = result_df[f'Balance_Trend_Slope_{k}M'].fillna(0.0)

    # --- СУММА ДОЛГА НА ТЕКУЩИЙ ДЕНЬ (Current_Debt) ---
    # Извлекаем последний доступный баланс И его дату (1-ое число месяца)
    latest_balance = sorted_long_df.groupby('ЛС').first().reset_index()[['ЛС', 'start', 'Period_Date']]
    past_payments = pay_df[pay_df['Дата оплаты'] <= curr_date]
    
    # Привязываем к каждому платежу дату последнего начисления (Period_Date) конкретного ЛС
    pays_with_period = pd.merge(past_payments, latest_balance[['ЛС', 'Period_Date']], left_on='Номер', right_on='ЛС', how='inner')
    
    # Оставляем только свежие платежи (от 1-го числа "текущего" месяца до curr_date)
    current_month_pays = pays_with_period[pays_with_period['Дата оплаты'] >= pays_with_period['Period_Date']]
    
    # Суммируем эти платежи
    recent_pays_sum = current_month_pays.groupby('ЛС')['Сумма'].sum().reset_index(name='Recent_Payments')
    
    # Вычитаем недавние платежи из баланса на начало месяца
    debt_df = pd.merge(latest_balance, recent_pays_sum, on='ЛС', how='left')
    debt_df['Recent_Payments'] = pd.to_numeric(debt_df['Recent_Payments'], errors='coerce').fillna(0)
    debt_df['Current_Debt'] = debt_df['start'] - debt_df['Recent_Payments']
    
    # Записываем в общий результат
    result_df = pd.merge(result_df, debt_df[['ЛС', 'Current_Debt']], left_on='Id', right_on='ЛС', how='left').drop(columns=['ЛС'])

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
    result_df = pd.merge(result_df, feat_6_df[['ЛС', 'Debt_to_Avg_Accrual_3M']], left_on='Id', right_on='ЛС', how='left').drop(columns=['ЛС'])

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

def get_seasonality_features(dt: pd.Timestamp) -> pd.DataFrame:
    """
    Вычисляет сезонные признаки (отопительный сезон и циклическое время).
    """
    month = dt.month
    day_of_year = dt.dayofyear
    days_in_year = 366 if dt.is_leap_year else 365
    
    # Отопительный сезон (октябрь-апрель)
    is_heating = int(month in [10, 11, 12, 1, 2, 3, 4])
    
    # Циклический месяц
    # Формула сдвинута так, чтобы Январь (1) давал 1, а Июль (7) давал -1.
    month_cos = np.cos((month - 1) * (2 * np.pi / 12))
    month_sin = np.sin((month - 1) * (2 * np.pi / 12))
    
    # Более точный вариант: циклический день года (плавное изменение каждый день)
    day_cos = np.cos((day_of_year - 1) * (2 * np.pi / days_in_year))
    day_sin = np.sin((day_of_year - 1) * (2 * np.pi / days_in_year))
    
    return pd.DataFrame([{
        'Is_Heating_Season': is_heating,
        'Season_Temperature_Cos': round(month_cos, 4),
        'Season_Day_Cos': round(day_cos, 4),
        'Season_Temperature_Sin': round(month_sin, 4),
        'Season_Day_Sin': round(day_sin, 4)
    }])
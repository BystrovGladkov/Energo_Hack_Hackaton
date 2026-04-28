import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.general_information import read_balances

if __name__ == '__main__':
    balances = read_balances()
    print(balances.shape)
    cols_to_check = balances.columns.drop('ЛС')
    balances = balances[(balances[cols_to_check] != 0).any(axis=1)].dropna()

    melted = balances.melt(id_vars=['ЛС'], var_name='Period', value_name='Value')

    # Вытаскиваем Год, Месяц и Метрику (start, paid, accr) из названий столбцов
    extracted = melted['Period'].str.extract(r'(?P<Year>\d{4})_(?P<Month>\d+)_(?P<Metric>[a-zA-Z]+)')
    melted = pd.concat([melted, extracted], axis=1)

    # Броня от ошибок: жестко чистим финансы (убираем пробелы, меняем запятые на точки)
    melted['Value'] = pd.to_numeric(
        melted['Value'].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.'), 
        errors='coerce'
    ).fillna(0.0)

    # Сворачиваем обратно, чтобы Год и Месяц стали строками, а метрики - столбцами
    long_df = melted.pivot_table(
        index=['ЛС', 'Year', 'Month'], 
        columns='Metric', 
        values='Value', 
        aggfunc='first'
    ).reset_index()

    # ==========================================
    # 2. РАСЧЕТ ДОЛЖНИКОВ И АГРЕГАЦИЯ
    # ==========================================

    # Гарантируем, что нужные столбцы существуют
    for col in ['start', 'paid']:
        if col not in long_df.columns:
            long_df[col] = 0.0

    # Определяем статус должника
    long_df['is_debtor'] = (long_df['start'] - long_df['paid']) > 0

    # Оставляем только должников
    debtors_df = long_df[long_df['is_debtor']].copy()

    # Создаем удобную дату (1-е число месяца) для сортировки по оси X
    debtors_df['Period_Date'] = pd.to_datetime(
        debtors_df['Year'].astype(str) + '-' + debtors_df['Month'].astype(str) + '-01'
    )

    # Считаем количество уникальных должников (ЛС) по месяцам
    monthly_debtors = debtors_df.groupby('Period_Date')['ЛС'].nunique().reset_index(name='Debtors_Count')
    monthly_debtors = monthly_debtors.sort_values('Period_Date')

    # ==========================================
    # 3. ОТРИСОВКА ГРАФИКА
    # ==========================================

    plt.figure(figsize=(12, 6))

    # Преобразуем даты в красивый формат (например, "01-2025" или "Янв 2025")
    x_labels = monthly_debtors['Period_Date'].dt.strftime('%m-%Y')

    # Рисуем гистограмму (bar chart)
    bars = plt.bar(x_labels, monthly_debtors['Debtors_Count'], color='#4C72B0', edgecolor='black')

    # Настройка внешнего вида
    plt.title('Динамика количества должников по месяцам', fontsize=16, pad=15)
    plt.xlabel('Месяц и Год', fontsize=12)
    plt.ylabel('Количество уникальных ЛС с долгом', fontsize=12)
    plt.xticks(rotation=45, ha='right') # Поворачиваем подписи, чтобы не слипались
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Сетка только по горизонтали

    # Добавляем цифры (количество) прямо над каждым столбиком
    for bar in bars:
        yval = bar.get_height()
        # Размещаем текст по центру столбика чуть выше его границы
        plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.015), 
                int(yval), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout() # Чтобы ничего не обрезалось
    plt.show()


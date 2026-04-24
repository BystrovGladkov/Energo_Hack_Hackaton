import pandas as pd
import numpy as np
import xgboost as xgb
import pulp
import joblib
from causalml.inference.meta import BaseXRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Стоимости каждой меры (в рублях).
COSTS = {
    0: 0.0,      # T=0: Нет воздействия
    1: 15.0,     # T=1: Автодозвон
    2: 0.0,      # T=2: Email
    3: 60.0,     # T=3: СМС
    4: 75.0,     # T=4: Обзвон оператором
    5: 30.0,     # T=5: Уведомление о введени ограничения
    6: 300.0,    # T=6: Выезд к абоненту
    7: 400.0,    # T=7: Заявление о выдаче судебного приказа
    8: 400.0,    # T=8: Претензия
    9: 600.0,    # T=9: Ограничение
    10: 500.0,   # T=10: Получение судебного приказа или ИЛ
}

Limits = {
    0: np.inf,
    1: 8000,
    2: np.inf,
    3: 2150,
    4: 1150,
    5: 6200,
    6: 500,
    7: 400,
    8: 400,
    9: 200,
    10: 250
}

def evaluate_and_save_results(optimized_results, Y_test, T_test, filename="optimized_decisions.csv"):
    """
    Оценивает соответствие предсказанных долей реальным данным и строит график калибровки.
    """
    # Подготовка данных
    optimized_results['Historical_T'] = T_test.values if isinstance(T_test, pd.Series) else T_test
    optimized_results['Actual_Fraction_Y'] = Y_test.values if isinstance(Y_test, pd.Series) else Y_test
    
    # Собираем предсказания именно для тех мер, которые были применены в истории
    preds = []
    for idx, row in optimized_results.iterrows():
        hist_t = int(row['Historical_T'])
        preds.append(row[f'Expected_Fraction_T{hist_t}'])
    
    optimized_results['Predicted_Historical_Fraction'] = preds
    optimized_results.to_csv(filename, index=False)

    # --- Построение графика калибровки ---
    plt.figure(figsize=(8, 8))
    
    # Идеальная линия (y = x)
    plt.plot([0, 1], [0, 1], "k--", label="Идеальная калибровка")
    
    # Точки данных
    plt.scatter(optimized_results['Predicted_Historical_Fraction'], 
                optimized_results['Actual_Fraction_Y'], 
                alpha=0.1, s=10, color='gray', label='Индивидуальные ЛС')
    
    # Линия тренда (насколько мы в среднем отклоняемся)
    lr = LinearRegression()
    X_reg = optimized_results[['Predicted_Historical_Fraction']]
    y_reg = optimized_results['Actual_Fraction_Y']
    lr.fit(X_reg, y_reg)
    plt.plot(X_reg, lr.predict(X_reg), color='red', label=f'Линия регрессии (R²={lr.score(X_reg, y_reg):.2f})')

    plt.xlabel('Предсказанная вероятность (доля) платежа')
    plt.ylabel('Фактическая доля платежа (Y_test)')
    plt.title('График калибровки предсказаний')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return optimized_results

def generate_shap_explanation(client_idx, X_test, x_learner, optimized_results, costs_dict):
    """
    Использует встроенные методы causalml для получения SHAP-значений и формирования текста.
    """
    row = optimized_results.iloc[client_idx]
    optimal_t = int(row['Optimal_Action'])
    
    # 1. Визуализация встроенными средствами для всей выборки (опционально)
    # x_learner.plot_shap_values(X=X_test, tau=x_learner.predict(X_test))
    
    explanation_text = f"Анализ решения для клиента (Долг: {row['Current_Debt']:.2f} руб.)\n"
    explanation_text += f"Решение: Мера T={optimal_t}. Ожидаемая прибыль: {row['Optimal_Expected_Profit']:.2f} руб.\n"
    explanation_text += "-"*60 + "\n"

    if optimal_t == 0:
        return explanation_text + "Действие не требуется: потенциальный доход ниже стоимости мер.\n"

    # 2. Получение векторов Шепли через встроенный метод
    # get_shap_values возвращает словарь {treatment_name: shap_array}
    shap_dict = x_learner.get_shap_values(X=X_test)
    
    # Названия воздействий в causalml обычно соответствуют тем, что были в T_train
    # Если мы хотим объяснить эффект меры T=optimal_t
    treatment_key = optimal_t 
    if treatment_key not in shap_dict:
        # Если ключи в словаре — строки, пробуем преобразовать
        treatment_key = str(optimal_t)

    # Достаем SHAP-значения для конкретного клиента и конкретной меры
    client_shap_values = shap_dict[treatment_key][client_idx]
    feature_names = X_test.columns
    
    # Сортировка признаков по вкладу
    feature_importance = []
    for name, val in zip(feature_names, client_shap_values):
        feature_importance.append((name, val, X_test.iloc[client_idx][name]))
    
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # 3. Формирование текста
    explanation_text += f"Почему выбрана мера T={optimal_t} (Uplift-анализ):\n"
    
    top_pos = [f for f in feature_importance if f[1] > 0][:3]
    explanation_text += "\nКлючевые факторы роста вероятности оплаты:\n"
    for name, impact, val in top_pos:
        explanation_text += f"  • {name} ({val}): +{impact*100:.2f}% к эффективности\n"

    top_neg = [f for f in feature_importance if f[1] < 0]
    top_neg.sort(key=lambda x: x[1]) # Самые сильные негативные факторы
    
    if top_neg:
        explanation_text += "\nФакторы, снижающие эффективность меры:\n"
        for name, impact, val in top_neg[:2]:
            explanation_text += f"  • {name} ({val}): {impact*100:.2f}% к эффективности\n"

    return explanation_text

def train_x_learner(X_train, T_train, Y_train, random_state=42):
    """
    Обучает X-Learner для предсказания CATE (инкрементального эффекта).
    """
    print("Обучение модели склонности (Propensity Score)...")
    propensity_model = xgb.XGBClassifier(
        objective='multi:softprob', 
        num_class=11, 
        eval_metric='mlogloss',
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    propensity_model.fit(X_train, T_train)
    p_scores = propensity_model.predict_proba(X_train)
    
    print("Инициализация базовых моделей XGBoost")
    outcome_model = xgb.XGBRegressor(
        objective='reg:logistic',
        n_estimators=100,
        random_state=random_state
    )

    effect_model = xgb.XGBRegressor(n_estimators=100, random_state=random_state)
    
    x_learner = BaseXRegressor(
        learner=outcome_model,
        effect_model=effect_model,
        control_name=0
    )
    print(f"Обучение X-Learner")
    x_learner.fit(X=X_train, treatment=T_train, y=Y_train, p=p_scores)

    filename = 'x_learner_model.joblib'
    joblib.dump(x_learner, filename)
    print(f"Модель успешно сохранена в файл {filename}")
    
    return x_learner, propensity_model, x_learner.model_mu_c

def assign_optimal_actions(results_df, run_limits):
    """
    Решает задачу линейного программирования для оптимального назначения мер.
    
    results_df: DataFrame, содержащий колонки 'Id' и 'Expected_Profit_T0' ... 'Expected_Profit_T10'
    run_limits: Словарь квот на текущий запуск (например, {0: 999999, 1: 1500, 2: 999999 ...})
    """
    # 1. Инициализация задачи максимизации
    prob = pulp.LpProblem("Maximize_Campaign_Profit", pulp.LpMaximize)
    
    clients = results_df.index.tolist() # Используем индексы датафрейма
    actions = list(run_limits.keys())
    
    # 2. Создаем переменные решения: x[i, t] = 1, если клиенту i назначена мера t, иначе 0
    # Используем словари для быстрого доступа
    x = pulp.LpVariable.dicts("assign", 
                              ((i, t) for i in clients for t in actions), 
                              cat='Binary')
    
    # Преобразуем датафрейм в словарь для сверхбыстрого доступа к прибылям
    profit_dict = {}
    for t in actions:
        profit_cols = results_df[f'Expected_Profit_T{t}'].to_dict()
        for i in clients:
            profit_dict[(i, t)] = profit_cols[i]

    # 3. Целевая функция: Сумма (Прибыль * x[i, t])
    prob += pulp.lpSum(profit_dict[(i, t)] * x[(i, t)] for i in clients for t in actions), "Total_Expected_Profit"
    
    # 4. Ограничение 1: Каждому клиенту ровно 1 мера
    for i in clients:
        prob += pulp.lpSum(x[(i, t)] for t in actions) == 1, f"One_Action_Per_Client_{i}"
        
    # 5. Ограничение 2: Лимиты ресурсов
    for t in actions:
        # Для бесконечных лимитов (T=0, T=2) ограничение не добавляем
        if run_limits[t] != np.inf:
            prob += pulp.lpSum(x[(i, t)] for i in clients) <= run_limits[t], f"Limit_Action_{t}"
            
    # 6. Решение задачи
    print("Запуск солвера (CBC)...")
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    print(f"Статус оптимизации: {pulp.LpStatus[prob.status]}")
    
    # 7. Извлечение результатов
    best_actions = []
    for i in clients:
        assigned_t = 0
        for t in actions:
            # Значение переменной будет 1.0 для выбранной меры
            if pulp.value(x[(i, t)]) == 1.0:
                assigned_t = t
                break
        best_actions.append(assigned_t)
        
    # Записываем оптимальную меру обратно в датафрейм
    results_df['Optimal_Action'] = best_actions
    
    # Считаем итоговую оптимизированную прибыль
    results_df['Optimal_Expected_Profit'] = [
        profit_dict[(i, t)] for i, t in zip(clients, best_actions)
    ]
    
    return results_df

def optimize_decisions(x_learner: BaseXRegressor, X_test, current_debts):
    """
    Рассчитывает ожидаемую прибыль для каждой меры и выбирает оптимальную.
    """
    print("Вычисление предсказаний (CATE)...")
    
    # Базовая вероятность (что клиент заплатит сам, если ничего не делать - T=0)
    base_pred_fraction = x_learner.model_mu_c.predict(X_test) 
    
    # Инкрементальный эффект (насколько мера УВЕЛИЧИТ долю возврата)
    # Возвращает матрицу N x 10 (CATE для T=1, T=2, T=3, ... относительно T=0)
    cate_estimates = x_learner.predict(X=X_test)
    
    results = pd.DataFrame({
        'Current_Debt': current_debts,
        'Expected_Fraction_T0': base_pred_fraction
    })
    
    # Рассчитываем итоговую долю возврата для каждого сценария (База + Инкремент)
    # Обнуляем действия для людей с несоответствующей категорией.
    for i in range(1, 11):
        results[f'Expected_Fraction_T{i}'] = np.clip(base_pred_fraction + cate_estimates[:, i-1], 0, 1)
        if i > 0 and i < 9:
            results.loc[~X_test['current_stage'].isin(['nothing', 'informing']), f'Expected_Fraction_T{i}'] = 0

    results.loc[~X_test['current_stage'].isin(['informing', 'restriction']), 'Expected_Fraction_T9'] = 0
    results.loc[~X_test['current_stage'].isin(['restriction', 'court']), 'Expected_Fraction_T10'] = 0

    # Нельзя использовать телефонную связь с теми, у кого его нет, и отключить энергию, если это невозможно.
    results['Expected_Fraction_T1'] *= X_test['Наличие телефона']
    results['Expected_Fraction_T3'] *= X_test['Наличие телефона']
    results['Expected_Fraction_T4'] *= X_test['Наличие телефона']
    results['Expected_Fraction_T9'] *= X_test['Возможность дистанционного отключения']

    # (Ожидаемая Прибыль = Ожидаемый Возврат - Затраты)
    profit_cols = []
    for t in range(11):
        col_name = f'Expected_Profit_T{t}'
        profit_cols.append(col_name)
        
        # (Доля * Сумма Долга) - Стоимость меры
        results[col_name] = (results[f'Expected_Fraction_T{t}'] * results['Current_Debt']) - COSTS[t]
    
    # Процент лимитов, которые можно сейчас израсходовать.
    current_limits_fraction = 1
    run_limits = {
        k: (np.inf if v == np.inf else int(v * current_limits_fraction)) 
        for k, v in Limits.items()
    }

    # Запускаем ЦЛП и получаем решение
    results = results.reset_index(drop=True) 
    optimized_results = assign_optimal_actions(results, run_limits)
        
    return optimized_results
import pandas as pd
import numpy as np
import xgboost as xgb
from causalml.inference.meta import BaseXRegressor
from sklearn.model_selection import train_test_split

# =========================================================
# 1. КОНФИГУРАЦИЯ БИЗНЕС-ОГРАНИЧЕНИЙ И ЗАТРАТ
# =========================================================

# Заглушки стоимости каждой меры (в рублях). Вы можете изменить их позже.
COSTS = {
    0: 0.0,      # T=0: Нет воздействия
    1: 15.0,     # T=1: Информирование (звонок, email, SMS)
    2: 500.0,    # T=2: Ограничение (выезд электрика, отключение)
    3: 3000.0    # T=3: Суд (госпошлина, работа юриста)
}


# =========================================================
# 3. ОБУЧЕНИЕ X-LEARNER И PROPENSITY SCORE
# =========================================================

def train_x_learner(X_train, T_train, Y_train, random_state=42):
    """
    Обучает X-Learner для предсказания CATE (инкрементального эффекта).
    """
    print("1. Обучение модели склонности (Propensity Score)...")
    # В наблюдаемых данных меры применяются не случайно. Нам нужно оценить вероятность
    # назначения каждой из 4 мер для каждого клиента, чтобы снять смещение выборки.
    propensity_model = xgb.XGBClassifier(
        objective='multi:softprob', 
        num_class=12, 
        eval_metric='mlogloss',
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    propensity_model.fit(X_train, T_train)
    # Получаем матрицу N x 12 с вероятностями каждого Treatment
    p_scores = propensity_model.predict_proba(X_train)
    
    print("2. Инициализация базовых моделей XGBoost (Fractional Regression)...")
    # Используем reg:logistic, так как таргет - это доля от 0 до 1. 
    outcome_model = xgb.XGBRegressor(objective='reg:logistic', n_estimators=100, random_state=random_state)

    # 2. Модели эффектов (Учатся предсказывать разницу D, которая лежит в диапазоне [-1, 1])
    # Используем обычную MSE (squarederror), так как таргет может быть отрицательным
    effect_model = xgb.XGBRegressor(n_estimators=100, random_state=random_state)

    # 3. Собираем X-Learner с правильным разделением
    x_learner = BaseXRegressor(
        learner=outcome_model,
        effect_model=effect_model,
        control_name=0
    )
    
    # Обучаем, передавая матрицу склонностей p
    x_learner.fit(X=X_train, treatment=T_train, y=Y_train, p=p_scores)
    
    return x_learner, propensity_model, x_learner.model_mu_c


# =========================================================
# 4. ДИСКРЕТНАЯ ОПТИМИЗАЦИЯ И ПРИНЯТИЕ РЕШЕНИЙ
# =========================================================

def optimize_decisions(x_learner, base_learner, X_test, current_debts):
    """
    Рассчитывает ожидаемую прибыль для каждой меры и выбирает оптимальную.
    """
    print("Вычисление предсказаний (CATE)...")
    
    # Базовая вероятность (что клиент заплатит сам, если ничего не делать - T=0)
    # X-Learner извлекает ее из модели, обученной на контрольной группе
    base_pred_fraction = base_learner.predict(X_test) 
    
    # Инкрементальный эффект (насколько мера УВЕЛИЧИТ долю возврата)
    # Возвращает матрицу N x 11 (CATE для T=1, T=2, T=3, ... относительно T=0)
    cate_estimates = x_learner.predict(X=X_test)
    
    # Создаем датафрейм для результатов
    results = pd.DataFrame({
        'Current_Debt': current_debts,
        'Expected_Fraction_T0': base_pred_fraction
    })
    
    # Рассчитываем итоговую долю возврата для каждого сценария (База + Инкремент)
    # Ограничиваем от 0 до 1 с помощью np.clip
    results['Expected_Fraction_T1'] = np.clip(base_pred_fraction + cate_estimates[:, 0], 0, 1)
    results['Expected_Fraction_T2'] = np.clip(base_pred_fraction + cate_estimates[:, 1], 0, 1)
    results['Expected_Fraction_T3'] = np.clip(base_pred_fraction + cate_estimates[:, 2], 0, 1)
    
    # Дискретная оптимизация (Ожидаемая Прибыль = Ожидаемый Возврат - Затраты)
    profit_cols = []
    for t in [0, 1, 2, 3]:
        col_name = f'Expected_Profit_T{t}'
        profit_cols.append(col_name)
        
        # Расчет: (Доля * Сумма Долга) - Стоимость меры
        results[col_name] = (results[f'Expected_Fraction_T{t}'] * results['Current_Debt']) - COSTS[t]
        
    # 4. Выбор наилучшего действия (argmax по прибыли)
    results['Best_Action'] = results[profit_cols].idxmax(axis=1).str.replace('Expected_Profit_T', '').astype(int)
    results['Max_Expected_Profit'] = results[profit_cols].max(axis=1)
    
    # =========================================================
    # ВАЖНО: ХАРДКОД-ОГРАНИЧЕНИЯ (ПРАВИЛА ПЕРЕХОДОВ И НЕВОЗМОЖНЫЕ МЕРЫ)
    # =========================================================
    # Пример: Если нет телефона (has_phone == 0), звонок (T=1) невозможен.
    # Мы принудительно заменяем Лучшее Действие.
    
    # Извлекаем индексы тех, у кого нет телефона, и кто при этом получил T=1 в качестве оптимального
    no_phone_mask = (X_test['has_phone'] == 0) & (results['Best_Action'] == 1)
    
    # Если T=1 нельзя, пересчитываем argmax только среди [T=0, T=2, T=3]
    if no_phone_mask.any():
        alt_cols = ['Expected_Profit_T0', 'Expected_Profit_T2', 'Expected_Profit_T3']
        results.loc[no_phone_mask, 'Best_Action'] = results.loc[no_phone_mask, alt_cols].idxmax(axis=1).str.replace('Expected_Profit_T', '').astype(int)

    # То же самое можно сделать для строгих переходов:
    # Если current_stage == 'informing' (нет еще информирования), то T=2 и T=3 недоступны,
    # значит насильно выбираем между T=0 и T=1.
        
    return results

# =========================================================
# ПРИМЕР ЗАПУСКА НА ДАННЫХ
# =========================================================
# X, T, Y, debts = prepare_data_for_uplift(master_df)
# X_train, X_test, T_train, T_test, Y_train, Y_test, debts_train, debts_test = train_test_split(X, T, Y, debts, test_size=0.2)
# 
# x_learner, prop_model, base_model = train_x_learner(X_train, T_train, Y_train)
# decision_df = optimize_decisions(x_learner, base_model, X_test, debts_test)
# 
# print(decision_df[['Current_Debt', 'Best_Action', 'Max_Expected_Profit']].head())
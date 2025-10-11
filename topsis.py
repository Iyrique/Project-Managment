import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import textwrap


class FuzzyTOPSIS:
    def __init__(self, alternatives: List[str], criteria_names: List[str],
                 benefit_criteria: List[bool], p: float = 2):
        """
        Корректная реализация Fuzzy TOPSIS

        Parameters:
        alternatives - названия альтернатив
        criteria_names - названия критериев
        benefit_criteria - список boolean, True для beneficial критериев
        p - параметр метрики Минковского (p=2 для евклидовой)
        """
        self.alternatives = alternatives
        self.criteria_names = criteria_names
        self.benefit_criteria = benefit_criteria
        self.p = p
        self.n_alternatives = len(alternatives)
        self.n_criteria = len(criteria_names)

        # DataFrames для хранения промежуточных результатов
        self.original_data = None
        self.normalized_data = None
        self.weighted_data = None
        self.distances = None
        self.results = None

    def load_data(self, criteria_data: List[List[Tuple]], criteria_weights: List[Tuple]):
        """Загрузка исходных данных"""
        self.criteria_data = criteria_data
        self.criteria_weights = [w[1] for w in criteria_weights]  # Берем ожидаемые веса

        # Создаем DataFrame с исходными данными
        data_dict = {}
        for j, criterion_name in enumerate(self.criteria_names):
            data_dict[criterion_name] = [criteria_data[j][i] for i in range(self.n_alternatives)]

        self.original_data = pd.DataFrame(data_dict, index=self.alternatives)
        self.original_data.index.name = 'Альтернативы'

        return self.original_data

    def normalize_matrix(self) -> pd.DataFrame:
        """Этап 1: Нормализация матрицы решений"""
        normalized_dict = {}

        for j, criterion_name in enumerate(self.criteria_names):
            criterion_data = self.criteria_data[j]
            normalized_values = []

            if self.benefit_criteria[j]:
                # Beneficial критерии: r_ij = x_ij / max(x_j)
                max_val = max([triplet[2] for triplet in criterion_data])  # max по оптимистичной
                for triplet in criterion_data:
                    normalized_triplet = (
                        triplet[0] / max_val,  # a/max
                        triplet[1] / max_val,  # b/max
                        triplet[2] / max_val  # c/max
                    )
                    normalized_values.append(normalized_triplet)
            else:
                # Cost критерии: r_ij = min(x_j) / x_ij
                min_val = min([triplet[0] for triplet in criterion_data])  # min по пессимистичной
                for triplet in criterion_data:
                    normalized_triplet = (
                        min_val / triplet[2],  # min/c
                        min_val / triplet[1],  # min/b
                        min_val / triplet[0]  # min/a
                    )
                    normalized_values.append(normalized_triplet)

            normalized_dict[criterion_name] = normalized_values

        self.normalized_data = pd.DataFrame(normalized_dict, index=self.alternatives)
        self.normalized_data.index.name = 'Альтернативы'

        return self.normalized_data

    def find_ideal_solutions(self) -> Tuple[Dict, Dict]:
        """Этап 2: Нахождение идеальных решений"""
        self.fpis = {}  # Fuzzy Positive Ideal Solution
        self.fnis = {}  # Fuzzy Negative Ideal Solution

        for j, criterion_name in enumerate(self.criteria_names):
            criterion_values = self.normalized_data[criterion_name]

            if self.benefit_criteria[j]:
                # Для beneficial: FPIS = максимум, FNIS = минимум
                self.fpis[criterion_name] = (
                    max([val[0] for val in criterion_values]),
                    max([val[1] for val in criterion_values]),
                    max([val[2] for val in criterion_values])
                )
                self.fnis[criterion_name] = (
                    min([val[0] for val in criterion_values]),
                    min([val[1] for val in criterion_values]),
                    min([val[2] for val in criterion_values])
                )
            else:
                # Для cost: FPIS = минимум, FNIS = максимум
                self.fpis[criterion_name] = (
                    min([val[0] for val in criterion_values]),
                    min([val[1] for val in criterion_values]),
                    min([val[2] for val in criterion_values])
                )
                self.fnis[criterion_name] = (
                    max([val[0] for val in criterion_values]),
                    max([val[1] for val in criterion_values]),
                    max([val[2] for val in criterion_values])
                )

        return self.fpis, self.fnis

    def calculate_distances(self) -> pd.DataFrame:
        """Этапы 3-4: Расчет расстояний с весами"""

        def fuzzy_distance(triplet1: Tuple, triplet2: Tuple) -> float:
            """Расстояние между двумя нечеткими числами"""
            a1, b1, c1 = triplet1
            a2, b2, c2 = triplet2
            return np.sqrt(1 / 3 * ((a1 - a2) ** 2 + (b1 - b2) ** 2 + (c1 - c2) ** 2))

        distances_data = []

        for i, alternative in enumerate(self.alternatives):
            sum_pos = 0
            sum_neg = 0

            for j, criterion_name in enumerate(self.criteria_names):
                # Получаем нормализованные значения
                norm_value = self.normalized_data.loc[alternative, criterion_name]
                fpis_value = self.fpis[criterion_name]
                fnis_value = self.fnis[criterion_name]

                # Рассчитываем расстояния (без весов)
                dist_pos = fuzzy_distance(norm_value, fpis_value)
                dist_neg = fuzzy_distance(norm_value, fnis_value)

                # Применяем веса и параметр p ПОСЛЕ расчета расстояний
                w_j = self.criteria_weights[j]
                sum_pos += (w_j ** self.p) * (dist_pos ** self.p)
                sum_neg += (w_j ** self.p) * (dist_neg ** self.p)

            # Финальное расстояние с учетом параметра p
            d_plus = sum_pos ** (1 / self.p)
            d_minus = sum_neg ** (1 / self.p)

            distances_data.append({
                'Альтернатива': alternative,
                'D_plus': d_plus,
                'D_minus': d_minus
            })

        self.distances = pd.DataFrame(distances_data)
        return self.distances

    def calculate_closeness(self) -> pd.DataFrame:
        """Этап 5: Расчет коэффициентов близости"""
        if self.distances is None:
            self.calculate_distances()

        results_data = []

        for i, row in self.distances.iterrows():
            d_plus = row['D_plus']
            d_minus = row['D_minus']

            # Коэффициент близости: C_i = D_minus / (D_plus + D_minus)
            if d_plus + d_minus == 0:
                closeness = 0
            else:
                closeness = d_minus / (d_plus + d_minus)

            results_data.append({
                'Альтернатива': row['Альтернатива'],
                'D_plus': d_plus,
                'D_minus': d_minus,
                'Коэффициент_близости': closeness
            })

        self.results = pd.DataFrame(results_data)
        # Сортируем по убыванию коэффициента близости
        self.results = self.results.sort_values('Коэффициент_близости', ascending=False)
        self.results['Ранг'] = range(1, len(self.results) + 1)

        return self.results

    def generate_report(self) -> str:
        """Генерация полного отчета с описаниями и таблицами"""

        report = []
        report.append("=" * 80)
        report.append("📊 ПОЛНЫЙ ОТЧЕТ АНАЛИЗА FUZZY TOPSIS")
        report.append("=" * 80)

        # 1. Описание методологии
        report.append("\n1. МЕТОДОЛОГИЯ АНАЛИЗА")
        report.append("-" * 40)
        methodology = """
        Метод Fuzzy TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
        используется для решения задач многокритериального принятия решений в условиях 
        неопределенности. Основные этапы метода:

        • Нормализация матрицы решений
        • Определение нечетких идеальных решений (FPIS и FNIS)
        • Расчет расстояний до идеальных решений
        • Расчет коэффициентов близости и ранжирование альтернатив

        Используется треугольные нечеткие числа вида (a, b, c), где:
        a - пессимистичная оценка, b - наиболее вероятная, c - оптимистичная оценка.
        """
        report.append(textwrap.dedent(methodology))

        # 2. Исходные данные
        report.append("\n2. ИСХОДНЫЕ ДАННЫЕ")
        report.append("-" * 40)
        report.append(f"Количество альтернатив: {self.n_alternatives}")
        report.append(f"Количество критериев: {self.n_criteria}")
        report.append(f"Параметр метрики Минковского (p): {self.p}")

        report.append("\nАльтернативы для анализа:")
        for i, alt in enumerate(self.alternatives, 1):
            report.append(f"  {i}. {alt}")

        report.append("\nКритерии оценки:")
        for i, (criterion, benefit) in enumerate(zip(self.criteria_names, self.benefit_criteria), 1):
            criterion_type = "Beneficial (больше = лучше)" if benefit else "Cost (меньше = лучше)"
            report.append(f"  {i}. {criterion} - {criterion_type}")

        # 3. Детальная таблица исходных данных
        report.append("\n3. ТАБЛИЦА ИСХОДНЫХ ДАННЫХ")
        report.append("-" * 40)

        detailed_original = []
        for alt in self.alternatives:
            row = {'Альтернатива': alt}
            for criterion in self.criteria_names:
                row[criterion] = self.original_data.loc[alt, criterion]
            detailed_original.append(row)

        df_detailed = pd.DataFrame(detailed_original)
        report.append(df_detailed.to_string(index=False))

        # 4. Веса критериев
        report.append("\n4. ВЕСА КРИТЕРИЕВ")
        report.append("-" * 40)
        weights_info = []
        for i, (criterion, weight) in enumerate(zip(self.criteria_names, self.criteria_weights), 1):
            weights_info.append({
                'Критерий': criterion,
                'Вес': f"{weight:.3f}",
                'Важность': f"{(weight / sum(self.criteria_weights)) * 100:.1f}%"
            })
        df_weights = pd.DataFrame(weights_info)
        report.append(df_weights.to_string(index=False))

        # 5. Нормализованная матрица
        report.append("\n5. НОРМАЛИЗОВАННАЯ МАТРИЦА РЕШЕНИЙ")
        report.append("-" * 40)
        report.append("Все значения приведены к безразмерному виду [0, 1]")

        detailed_normalized = []
        for alt in self.alternatives:
            row = {'Альтернатива': alt}
            for criterion in self.criteria_names:
                row[criterion] = self.normalized_data.loc[alt, criterion]
            detailed_normalized.append(row)

        df_normalized = pd.DataFrame(detailed_normalized)
        report.append(df_normalized.to_string(index=False))

        # 6. Идеальные решения
        report.append("\n6. ИДЕАЛЬНЫЕ РЕШЕНИЯ")
        report.append("-" * 40)
        ideal_solutions = []
        for criterion in self.criteria_names:
            ideal_solutions.append({
                'Критерий': criterion,
                'FPIS': self.fpis[criterion],
                'FNIS': self.fnis[criterion]
            })
        df_ideal = pd.DataFrame(ideal_solutions)
        report.append(df_ideal.to_string(index=False))

        # 7. Расстояния и коэффициенты близости
        report.append("\n7. РЕЗУЛЬТАТЫ РАСЧЕТОВ")
        report.append("-" * 40)
        report.append("D_plus - расстояние до положительного идеального решения")
        report.append("D_minus - расстояние до отрицательного идеального решения")
        report.append("Коэффициент близости = D_minus / (D_plus + D_minus)")

        results_display = self.results.copy()
        results_display['D_plus'] = results_display['D_plus'].round(6)
        results_display['D_minus'] = results_display['D_minus'].round(6)
        results_display['Коэффициент_близости'] = results_display['Коэффициент_близости'].round(6)
        report.append(results_display.to_string(index=False))

        # 8. Итоговое ранжирование
        report.append("\n8. ИТОГОВОЕ РАНЖИРОВАНИЕ АЛЬТЕРНАТИВ")
        report.append("-" * 40)

        ranking_info = []
        for _, row in self.results.iterrows():
            ranking_info.append({
                'Ранг': int(row['Ранг']),
                'Альтернатива': row['Альтернатива'],
                'Коэффициент': f"{row['Коэффициент_близости']:.4f}",
                'Статус': '🏆 ЛУЧШАЯ' if row['Ранг'] == 1 else ''
            })

        df_ranking = pd.DataFrame(ranking_info)
        report.append(df_ranking.to_string(index=False))

        # 9. Анализ и рекомендации
        report.append("\n9. АНАЛИЗ И РЕКОМЕНДАЦИИ")
        report.append("-" * 40)

        best_alt = self.results.iloc[0]
        worst_alt = self.results.iloc[-1]

        analysis = f"""
        На основе проведенного анализа методом Fuzzy TOPSIS:

        • Наилучшая альтернатива: {best_alt['Альтернатива']} 
          (коэффициент близости: {best_alt['Коэффициент_близости']:.4f})

        • Наихудшая альтернатива: {worst_alt['Альтернатива']}
          (коэффициент близости: {worst_alt['Коэффициент_близости']:.4f})

        • Размах коэффициентов: {best_alt['Коэффициент_близости'] - worst_alt['Коэффициент_близости']:.4f}

        Рекомендуется к реализации: {best_alt['Альтернатива']}
        """
        report.append(textwrap.dedent(analysis))

        # 10. Чувствительность анализа
        report.append("\n10. АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ")
        report.append("-" * 40)

        closeness_values = self.results['Коэффициент_близости'].values
        mean_closeness = np.mean(closeness_values)
        std_closeness = np.std(closeness_values)

        sensitivity = f"""
        • Средний коэффициент близости: {mean_closeness:.4f}
        • Стандартное отклонение: {std_closeness:.4f}
        • Коэффициент вариации: {(std_closeness / mean_closeness) * 100:.2f}%

        Чем выше стандартное отклонение, тем более выражено преимущество 
        лучшей альтернативы над остальными.
        """
        report.append(textwrap.dedent(sensitivity))

        report.append("\n" + "=" * 80)
        report.append("Отчет сгенерирован автоматически с использованием метода Fuzzy TOPSIS")
        report.append("=" * 80)

        return "\n".join(report)

    def solve(self) -> pd.DataFrame:
        """Полное решение методом Fuzzy TOPSIS"""
        print("🎯 ЗАПУСК МЕТОДА FUZZY TOPSIS")
        print("=" * 60)

        # Этап 1: Нормализация
        print("\n1. НОРМАЛИЗАЦИЯ МАТРИЦЫ РЕШЕНИЙ:")
        norm_df = self.normalize_matrix()
        print(norm_df)

        # Этап 2: Идеальные решения
        print("\n2. ИДЕАЛЬНЫЕ РЕШЕНИИ:")
        fpis, fnis = self.find_ideal_solutions()
        ideal_df = pd.DataFrame({'FPIS': fpis, 'FNIS': fnis})
        print(ideal_df)

        # Этапы 3-4: Расстояния
        print("\n3. РАССТОЯНИЯ ДО ИДЕАЛЬНЫХ РЕШЕНИЙ:")
        dist_df = self.calculate_distances()
        print(dist_df)

        # Этап 5: Коэффициенты близости
        print("\n4. ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        results_df = self.calculate_closeness()
        print(results_df.to_string(index=False))

        # Лучшая альтернатива
        best = results_df.iloc[0]
        print(f"\n🏆 НАИЛУЧШАЯ АЛЬТЕРНАТИВА: {best['Альтернатива']} "
              f"(коэффициент: {best['Коэффициент_близости']:.4f})")

        return results_df


# ДАННЫЕ ДЛЯ АНАЛИЗА
alternatives = ["Студ.проекты", "Бронирование", "Онлайн-курсы", "Маркетплейс"]
criteria_names = ["Сложность", "Современность", "Библиотеки", "Универсальность"]

# Типы критериев: True = beneficial (больше=лучше), False = cost (меньше=лучше)
benefit_criteria = [False, True, True, True]  # Сложность - cost, остальные - beneficial

# Исходные данные (треугольные нечеткие числа)
criteria_data = [
    # Сложность (cost)
    [(5, 4, 3), (6, 5, 4), (7, 6, 5), (8, 7, 6)],
    # Современность (beneficial)
    [(4, 5, 5), (6, 7, 8), (9, 9, 9), (8, 9, 10)],
    # Библиотеки (beneficial)
    [(8, 8, 9), (7, 7, 8), (5, 6, 6), (5, 6, 7)],
    # Универсальность (beneficial)
    [(6, 7, 7), (6, 7, 8), (9, 9, 10), (4, 7, 8)]
]

# Веса критериев
criteria_weights = [(0.266, 0.266, 0.266), (0.123, 0.123, 0.123),
                    (0.097, 0.097, 0.097), (0.514, 0.514, 0.514)]

# ЗАПУСК АНАЛИЗА
print("📊 АНАЛИЗ КОНЦЕПЦИЙ ПРОГРАММНЫХ ПРОЕКТОВ")
print("=" * 50)

# Создаем и запускаем анализ
topsis_analysis = FuzzyTOPSIS(
    alternatives=alternatives,
    criteria_names=criteria_names,
    benefit_criteria=benefit_criteria,
    p=2  # Евклидова метрика
)

# Загружаем данные
original_df = topsis_analysis.load_data(criteria_data, criteria_weights)

print("ИСХОДНЫЕ ДАННЫЕ:")
print(original_df)
print("\nВЕСА КРИТЕРИЕВ:", topsis_analysis.criteria_weights)

# Получаем результаты
final_results = topsis_analysis.solve()

# ГЕНЕРАЦИЯ ПОЛНОГО ОТЧЕТА
print("\n" + "=" * 80)
print("📈 ГЕНЕРАЦИЯ ПОЛНОГО ОТЧЕТА")
print("=" * 80)

full_report = topsis_analysis.generate_report()
print(full_report)

# Сохранение отчета в файл
with open('fuzzy_topsis_report.md', 'w', encoding='utf-8') as f:
    f.write(full_report)

print("\n💾 Отчет сохранен в файл: fuzzy_topsis_report.txt")

# ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ
print("\n" + "=" * 60)
print("📈 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ДЛЯ ОТЧЕТА")
print("=" * 60)

print("\nСВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:")
summary_df = final_results.copy()
summary_df['Коэффициент_близости'] = summary_df['Коэффициент_близости'].round(4)
summary_df['D_plus'] = summary_df['D_plus'].round(4)
summary_df['D_minus'] = summary_df['D_minus'].round(4)
print(summary_df.to_string(index=False))

print(f"\n📋 Статистика:")
print(f"Лучший коэффициент: {final_results['Коэффициент_близости'].max():.4f}")
print(f"Худший коэффициент: {final_results['Коэффициент_близости'].min():.4f}")
print(f"Размах: {final_results['Коэффициент_близости'].max() - final_results['Коэффициент_близости'].min():.4f}")
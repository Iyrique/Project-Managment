import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


class FuzzyTOPSIS:
    def __init__(self, alternatives: List[str], criteria_names: List[str],
                 benefit_criteria: List[bool], p: float = 2):
        """
        Корректная реализация Fuzzy TOPSIS

        Parameters:
        alternatives - названия концепций
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

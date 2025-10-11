    def generate_report(self) -> str:
        """Генерация полного отчета в Markdown формате"""

        report = []
        report.append("# 📊 ПОЛНЫЙ ОТЧЕТ АНАЛИЗА FUZZY TOPSIS")
        report.append("---")

        # 1. Описание методологии
        report.append("## 1. МЕТОДОЛОГИЯ АНАЛИЗА")
        methodology = """
        Метод **Fuzzy TOPSIS** (Technique for Order Preference by Similarity to Ideal Solution) 
        используется для решения задач многокритериального принятия решений в условиях неопределенности.

        **Основные этапы метода:**
        - Нормализация матрицы решений
        - Определение нечетких идеальных решений (FPIS и FNIS)
        - Расчет расстояний до идеальных решений
        - Расчет коэффициентов близости и ранжирование альтернатив

        **Треугольные нечеткие числа:** (a, b, c)
        - a - пессимистичная оценка
        - b - наиболее вероятная оценка  
        - c - оптимистичная оценка
        """
        report.append(textwrap.dedent(methodology))

        # 2. Исходные данные
        report.append("## 2. ИСХОДНЫЕ ДАННЫЕ")
        report.append(f"- **Концепции:** {self.n_alternatives}")
        report.append(f"- **Количество критериев:** {self.n_criteria}")
        report.append(f"- **Параметр метрики Минковского (p):** {self.p}")

        report.append("\n**Концепции для анализа:**")
        for i, alt in enumerate(self.alternatives, 1):
            report.append(f"  {i}. {alt}")

        report.append("\n**Критерии оценки:**")
        for i, (criterion, benefit) in enumerate(zip(self.criteria_names, self.benefit_criteria), 1):
            criterion_type = "🟢 Beneficial (больше = лучше)" if benefit else "🔴 Cost (меньше = лучше)"
            report.append(f"  {i}. **{criterion}** - {criterion_type}")

        # 3. Детальная таблица исходных данных
        report.append("## 3. ТАБЛИЦА ИСХОДНЫХ ДАННЫХ")

        # Создаем Markdown таблицу
        report.append("| Альтернатива | Сложность | Современность | Библиотеки | Универсальность |")
        report.append("|-------------|-----------|---------------|------------|----------------|")

        for alt in self.alternatives:
            row = [alt]
            for criterion in self.criteria_names:
                triplet = self.original_data.loc[alt, criterion]
                # Форматируем triplet для лучшей читаемости
                formatted_triplet = f"({triplet[0]}, {triplet[1]}, {triplet[2]})"
                row.append(formatted_triplet)
            report.append("| " + " | ".join(row) + " |")

        # 4. Веса критериев
        report.append("## 4. ВЕСА КРИТЕРИЕВ")

        report.append("| Критерий | Вес | Важность |")
        report.append("|----------|-----|----------|")
        for criterion, weight in zip(self.criteria_names, self.criteria_weights):
            importance_pct = (weight / sum(self.criteria_weights)) * 100
            report.append(f"| {criterion} | {weight:.3f} | {importance_pct:.1f}% |")

        # 5. Нормализованная матрица
        report.append("## 5. НОРМАЛИЗОВАННАЯ МАТРИЦА РЕШЕНИЙ")

        report.append("| Альтернатива | Сложность | Современность | Библиотеки | Универсальность |")
        report.append("|-------------|-----------|---------------|------------|----------------|")

        for alt in self.alternatives:
            row = [alt]
            for criterion in self.criteria_names:
                triplet = self.normalized_data.loc[alt, criterion]
                # Округляем для читаемости
                formatted_triplet = f"({triplet[0]:.3f}, {triplet[1]:.3f}, {triplet[2]:.3f})"
                row.append(formatted_triplet)
            report.append("| " + " | ".join(row) + " |")

        # 6. Идеальные решения
        report.append("## 6. ИДЕАЛЬНЫЕ РЕШЕНИЯ")

        report.append("| Критерий | FPIS | FNIS |")
        report.append("|----------|------|------|")
        for criterion in self.criteria_names:
            fpis_formatted = f"({self.fpis[criterion][0]:.3f}, {self.fpis[criterion][1]:.3f}, {self.fpis[criterion][2]:.3f})"
            fnis_formatted = f"({self.fnis[criterion][0]:.3f}, {self.fnis[criterion][1]:.3f}, {self.fnis[criterion][2]:.3f})"
            report.append(f"| {criterion} | {fpis_formatted} | {fnis_formatted} |")

        # 7. Расстояния и коэффициенты близости
        report.append("## 7. РЕЗУЛЬТАТЫ РАСЧЕТОВ")

        report.append("- **D_plus** - расстояние до положительного идеального решения")
        report.append("- **D_minus** - расстояние до отрицательного идеального решения")
        report.append("- **Коэффициент близости** = D_minus / (D_plus + D_minus)")

        report.append("")
        report.append("| Альтернатива | D_plus | D_minus | Коэффициент близости |")
        report.append("|--------------|--------|---------|----------------------|")

        for _, row in self.results.iterrows():
            report.append(
                f"| {row['Альтернатива']} | {row['D_plus']:.6f} | {row['D_minus']:.6f} | {row['Коэффициент_близости']:.6f} |")

        # 8. Итоговое ранжирование
        report.append("## 8. ИТОГОВОЕ РАНЖИРОВАНИЕ АЛЬТЕРНАТИВ")

        report.append("| Ранг | Альтернатива | Коэффициент | Статус |")
        report.append("|------|-------------|-------------|--------|")

        for _, row in self.results.iterrows():
            status = "🏆 **ЛУЧШАЯ**" if row['Ранг'] == 1 else ""
            report.append(
                f"| {int(row['Ранг'])} | {row['Альтернатива']} | {row['Коэффициент_близости']:.4f} | {status} |")

        # 9. Анализ и рекомендации
        report.append("## 9. АНАЛИЗ И РЕКОМЕНДАЦИИ")

        best_alt = self.results.iloc[0]
        worst_alt = self.results.iloc[-1]

        analysis = f"""
        **Результаты анализа методом Fuzzy TOPSIS:**

        - **Наилучшая альтернатива:** **{best_alt['Альтернатива']}**  
          (коэффициент близости: {best_alt['Коэффициент_близости']:.4f})

        - **Наихудшая альтернатива:** {worst_alt['Альтернатива']}  
          (коэффициент близости: {worst_alt['Коэффициент_близости']:.4f})

        - **Размах коэффициентов:** {best_alt['Коэффициент_близости'] - worst_alt['Коэффициент_близости']:.4f}

        **🎯 Рекомендация:** К реализации рекомендуется **{best_alt['Альтернатива']}**
        """
        report.append(textwrap.dedent(analysis))

        # 10. Анализ чувствительности
        report.append("## 10. АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ")

        closeness_values = self.results['Коэффициент_близости'].values
        mean_closeness = np.mean(closeness_values)
        std_closeness = np.std(closeness_values)

        sensitivity = f"""
        - **Средний коэффициент близости:** {mean_closeness:.4f}
        - **Стандартное отклонение:** {std_closeness:.4f}
        - **Коэффициент вариации:** {(std_closeness / mean_closeness) * 100:.2f}%

        *Чем выше стандартное отклонение, тем более выражено преимущество лучшей альтернативы над остальными.*
        """
        report.append(textwrap.dedent(sensitivity))

        report.append("---")

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
print("📈 ГЕНЕРАЦИЯ ПОЛНОГО ОТЧЕТА В MARKDOWN")
print("=" * 80)

full_report = topsis_analysis.generate_report()
print(full_report)

# Сохранение отчета в MD файл
with open('fuzzy_topsis_report.md', 'w', encoding='utf-8') as f:
    f.write(full_report)

print("\n💾 Отчет сохранен в файл: fuzzy_topsis_report.md")

# Также сохраним красивую текстовую версию для консоли
#with open('fuzzy_topsis_report.txt', 'w', encoding='utf-8') as f:
 #   # Здесь можно сохранить текстовую версию без Markdown разметки
  #  f.write(full_report.replace('**', '').replace('## ', '\n').replace('# ', '\n'))

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
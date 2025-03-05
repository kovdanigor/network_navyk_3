import networkx as nx
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler


federal_districts = {
    "Центральный": ["Москва", "Московская область", "Белгородская область", "Брянская область", "Владимирская область",
                    "Воронежская область", "Ивановская область", "Калужская область", "Костромская область",
                    "Курская область", "Липецкая область", "Орловская область", "Рязанская область",
                    "Смоленская область", "Тамбовская область", "Тверская область", "Тульская область",
                    "Ярославская область"],

    "Северо-Западный": ["Санкт-Петербург", "Ленинградская область", "Архангельская область", "Вологодская область",
                        "Калининградская область", "Республика Карелия", "Республика Коми", "Мурманская область",
                        "Ненецкий автономный округ", "Новгородская область", "Псковская область"],

    "Южный": ["Республика Адыгея", "Астраханская область", "Волгоградская область", "Республика Калмыкия",
              "Краснодарский край", "Ростовская область", "Севастополь", "Республика Крым",
              "Луганская народная республика", "Донецкая народная республика",
              "Херсонская область", "Запорожская область"],

    "Северо-Кавказский": ["Республика Дагестан", "Республика Ингушетия", "Кабардино-Балкарская республика",
                          "Карачаево-Черкесская Республика", "Республика Северная Осетия-Алания",
                          "Чеченская республика", "Ставропольский край"],

    "Приволжский": ["Республика Башкортостан", "Республика Марий Эл", "Республика Мордовия", "Республика Татарстан",
                    "Удмуртская Республика", "Чувашская Республика", "Кировская область", "Нижегородская область",
                    "Оренбургская область", "Пензенская область", "Пермский край", "Самарская область",
                    "Саратовская область", "Ульяновская область"],

    "Уральский": ["Курганская область", "Свердловская область", "Тюменская область", "Челябинская область",
                  "Ханты-Мансийский автономный округ — Югра", "Ямало-Ненецкий АО"],

    "Сибирский": ["Алтайский край", "Красноярский край", "Иркутская область", "Кемеровская область",
                  "Новосибирская область", "Омская область", "Томская область", "Республика Тыва",
                  "Республика Хакасия", "Республика Бурятия", "Забайкальский край"],

    "Дальневосточный": ["Амурская область", "Еврейская автономная область", "Камчатский край", "Магаданская область",
                        "Приморский край", "Сахалинская область", "Хабаровский край", "Чукотский автономный округ",
                        "Республика Бурятия", "Забайкальский край", "Республика Саха (Якутия)"]
}


def get_federal_district(region):
    for district, regions in federal_districts.items():
        if region in regions:
            return district
    return "Неизвестно"


def nonzero_mean(x):
    nonzero = x[x != 0]
    return nonzero.mean() if len(nonzero) > 0 else 0


def create_co_occurrence_matrix(df: pd.DataFrame, skills_field: str,
                                combination_size: int = 2) -> pd.DataFrame:
    """
    Создает матрицу co-occurrence для навыков, основываясь на столбце, где каждый элемент является списком навыков.

    :param df: DataFrame с исходными данными.
    :param skills_field: Название столбца, содержащего списки навыков.
    :param combination_size: Размер комбинации для подсчета co-occurrence (2 для пар, 3 для троек).
    :return:
        - Если combination_size = 2: возвращает квадратную матрицу (DataFrame) с подсчетом парного сосуществования навыков.
        - Если combination_size = 3: возвращает DataFrame с колонками ['Skill1', 'Skill2', 'Skill3', 'Count'],
          где каждая строка представляет уникальную тройку навыков и количество их совместного появления.
    """
    # Определяем все уникальные навыки
    all_skills = set()
    for skills in df[skills_field].dropna():
        all_skills.update(skills)
    all_skills = sorted(list(all_skills))
    skill_index = {skill: idx for idx, skill in enumerate(all_skills)}

    if combination_size:
        # Инициализируем квадратную матрицу для подсчета пар
        co_occurrence = np.zeros((len(all_skills), len(all_skills)), dtype=int)
        for skills in df[skills_field].dropna():
            unique_skills = [s for s in skills if s in skill_index]
            for pair in combinations(unique_skills, combination_size):
                i, j = skill_index[pair[0]], skill_index[pair[1]]
                co_occurrence[i, j] += 1
                co_occurrence[j, i] += 1
        co_occurrence_df = pd.DataFrame(
            co_occurrence, index=all_skills, columns=all_skills)
        return co_occurrence_df

    else:
        raise ValueError(
            "Параметр combination_size должен быть равен 2 или 3.")

#  Фильтрация матрицы по метрикам


def filter_matrix_from_graph(G: nx.Graph, centrality_type: str = 'degree_centrality',
                             top_n: int = 400, top_n_rows: int = None,
                             top_n_cols: int = None,
                             weight_attr: str = 'weight') -> pd.DataFrame:
    """
    Фильтрует граф на основе выбранной центральности, оставляя топ-N узлов.

    Для двудольного графа (у всех узлов присутствует атрибут 'bipartite'):
      - Узлы первого уровня (где data['bipartite']==1) используются как столбцы,
      - Узлы второго уровня (где data['bipartite']==2) используются как строки.
      Если не заданы top_n_rows или top_n_cols, то берется значение top_n.

    Для обычного графа возвращается квадратная матрица смежности
    только для топ-N узлов.

    :param G: входной граф (networkx.Graph).
    :param centrality_type: тип центральности для фильтрации. Поддерживаются:
     'degree', 'betweenness', 'closeness', 'eigenvector'.
    :param top_n: количество узлов для отбора в обычном графе.
    :param top_n_rows: количество строк (узлов второй доли) для двудольного графа.
    :top_n_cols: количество столбцов (узлов первой доли) для двудольного графа.
    :weight_attr: имя атрибута веса ребра (если отсутствует – считается равным 1).
    :return: DataFrame, представляющий отфильтрованную матрицу смежности.


    Примеры использования:
    Для двудольного графа:
      >>> filtered_matrix = filter_matrix_from_graph(G, centrality_type='betweenness', top_n_rows=400, top_n_cols=400)
    Для обычного графа:
      >>> filtered_matrix = filter_matrix_from_graph(G, centrality_type='degree', top_n=300)

    """

    # Выбор функции вычисления центральности
    centrality_type = centrality_type.lower()
    if centrality_type == 'degree_centrality':
        centrality = nx.degree_centrality(G)
    elif centrality_type == 'betweenness_centrality':
        centrality = nx.betweenness_centrality(G, weight=weight_attr)
    elif centrality_type == 'closeness_centrality':
        centrality = nx.closeness_centrality(G)
    elif centrality_type == 'eigenvector_centrality':
        centrality = nx.eigenvector_centrality(G, weight=weight_attr)
    else:
        raise ValueError(f"Неизвестный тип центральности: {centrality_type}. "
                         "Поддерживаются 'degree', 'betweenness', 'closeness', 'eigenvector'")

    all_have_bipartite = all('bipartite' in data for _,
                             data in G.nodes(data=True))

    if all_have_bipartite:
        # Получаем узлы для обеих долей
        first_nodes = [node for node, data in G.nodes(
            data=True) if data.get("bipartite") == 1]
        second_nodes = [node for node, data in G.nodes(
            data=True) if data.get("bipartite") == 2]

        # Если отдельно не заданы, используем top_n
        if top_n_cols is None:
            top_n_cols = top_n
        if top_n_rows is None:
            top_n_rows = top_n

        # Отбираем топ-N узлов для каждого уровня по значению центральности
        centrality_first = {node: centrality[node] for node in first_nodes}
        top_first = sorted(centrality_first, key=centrality_first.get, reverse=True)[
            :top_n_cols]

        centrality_second = {node: centrality[node] for node in second_nodes}
        top_second = sorted(centrality_second, key=centrality_second.get, reverse=True)[
            :top_n_rows]

        # Создаем DataFrame нужного размера и заполняем его нулями
        filtered_matrix = pd.DataFrame(0, index=top_second, columns=top_first)

        # Заполняем матрицу: поскольку ребро в двудольном графе соединяет узлы из разных уровней,
        # ищем ребра между выбранными узлами
        for u, v, data in G.edges(data=True):
            # Определяем вес ребра (если нет атрибута, считаем его равным 1)
            w = data.get(weight_attr, 1)
            # Ребро может быть задано в любом порядке, определяем доли
            if u in top_first and v in top_second:
                filtered_matrix.at[v, u] = w
            elif u in top_second and v in top_first:
                filtered_matrix.at[u, v] = w

        filtered_graph = create_bipartite_graph(filtered_matrix)

    else:
        # Граф не двудолен, работаем с единым списком узлов.
        # Отбираем топ-N узлов по центральности.
        nodes_sorted = sorted(
            centrality, key=centrality.get, reverse=True)[:top_n]

        # Создаем квадратную матрицу смежности с индексами и столбцами равными отобранным узлам
        filtered_matrix = pd.DataFrame(
            0, index=nodes_sorted, columns=nodes_sorted)

        # Заполняем матрицу весами ребер (если ребро существует, используем вес или 1)
        for u, v, data in G.edges(data=True):
            if u in nodes_sorted and v in nodes_sorted:
                w = data.get(weight_attr, 1)
                filtered_matrix.at[u, v] = w
                filtered_matrix.at[v, u] = w  # так как граф неориентированный

        filtered_graph = nx.from_pandas_adjacency(filtered_matrix)

    return filtered_graph

# 3.1. Создание функции для создания двумодальных матриц


def create_group_values_matrix(df: pd.DataFrame, group_field: str, value_field: str) -> pd.DataFrame:
    """
    Создает двумодальную матрицу значений, группируя по group_field и учитывая значения из value_field,
    добавляя _row и _col, если значения повторяются.

    :param df: DataFrame с исходными данными.
    :param group_field: Название колонки для группировки (столбцы матрицы).
    :param value_field: Название колонки со значениями (списками или отдельными значениями, строки матрицы).
    :return: Двумодальная матрица значений в виде DataFrame.
    """
    sample_value = df[value_field].dropna().iloc[0]  # Определяем тип значений

    if isinstance(sample_value, list):
        # Группировка и разворачивание списков значений
        role_values = (
            df.groupby(group_field)[value_field]
              .apply(lambda x: [item for sublist in x for item in sublist])
              .to_dict()
        )

        # Определяем все уникальные значения для строк матрицы
        all_values = {item for values in role_values.values()
                      for item in values}

        # Определяем пересекающиеся названия
        common_labels = all_values & set(role_values.keys())

        # Добавляем суффиксы для избежания конфликтов
        row_labels = {
            x: f"{x}_row" if x in common_labels else x for x in all_values}
        col_labels = {
            x: f"{x}_col" if x in common_labels else x for x in role_values.keys()}

        # Создаем пустую матрицу с обновленными индексами
        matrix = pd.DataFrame(0, index=row_labels.values(),
                              columns=col_labels.values())

        # Заполняем матрицу
        for group, values in role_values.items():
            for value in values:
                matrix.loc[row_labels[value], col_labels[group]] += 1
    else:
        # Используем crosstab и переименовываем совпадающие названия
        matrix = pd.crosstab(df[value_field], df[group_field])

        # Определяем пересечения и добавляем суффиксы
        common_labels = set(matrix.index) & set(matrix.columns)
        matrix.index = [
            f"{x}_row" if x in common_labels else x for x in matrix.index]
        matrix.columns = [
            f"{x}_col" if x in common_labels else x for x in matrix.columns]

    return matrix


# 3.2. Функции для нормализации матрицы и создание одномодальных и многоуровневых матриц

def normalize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализует матрицу значений в диапазоне [0, 1] с помощью MinMaxScaler.

    :param matrix: Исходная матрица (DataFrame).
    :return: Нормализованная матрица (DataFrame).


    Пример использования:
      >>> normalized_matrix = normalize_matrix(matrix)

    """
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(matrix),
                        index=matrix.index,
                        columns=matrix.columns)

# Main


def create_whole_matrix(group_matrix: pd.DataFrame, df_data: Optional[pd.DataFrame] = None,
                        use_co_occurrence: bool = False,
                        skills_field: str = 'raw_skills') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Создает полную сеть связей между навыками и профессиями.

    :param group_matrix: Матрица соответствия (строки – навыки, столбцы – группы/профессии).
    :param df_data: Исходный DataFrame с данными (используется при вычислении co-occurrence).
    :param use_co_occurrence: Флаг использования co-occurrence матрицы навыков.
    :param skills_field: Название столбца с навыками в df_data.
    :return: Кортеж из трёх матриц (row_matrix, column_matrix, whole_matrix).


    Примеры использования:
    Если параметр use_co_occurrence = True:
      >>> skills_matrix, jobs_matrix, whole_matrix = create_whole_matrix(skills_roles_matrix, df_construction, use_co_occurrence=True, skills_field='raw_skills')
    Если параметр use_co_occurrence = False:
      >>> roles_matrix, regions_matrix, whole_matrix = create_whole_matrix(roles_regions_matrix)

    """
    group_matrix = (group_matrix > 0).astype(int)
    all_row = group_matrix.index.tolist()

    if use_co_occurrence and df_data is not None:
        # Создание co-occurrence матрицы навыков
        skill_index = {skill: idx for idx, skill in enumerate(all_row)}
        co_occurrence = np.zeros((len(all_row), len(all_row)), dtype=int)

        for skills in df_data[skills_field]:
            unique_skills = [s for s in skills if s in skill_index]
            for pair in combinations(unique_skills, 2):
                i, j = skill_index[pair[0]], skill_index[pair[1]]
                co_occurrence[i, j] += 1
                co_occurrence[j, i] += 1

        row_matrix = pd.DataFrame(
            co_occurrence, index=all_row, columns=all_row)
    else:
        # Матрица схожести навыков через произведение
        row_sim = group_matrix.values.dot(group_matrix.values.T)
        row_matrix = pd.DataFrame(row_sim, index=all_row, columns=all_row)

    np.fill_diagonal(row_matrix.values, 0)
    # row_matrix = normalize_matrix(row_matrix)

    # Матрица схожести профессий (групп)
    # group_matrix = normalize_matrix(group_matrix)

    column_sim = group_matrix.T.values.dot(group_matrix.values)
    column_matrix = pd.DataFrame(
        column_sim, index=group_matrix.columns, columns=group_matrix.columns)
    np.fill_diagonal(column_matrix.values, 0)
    # column_matrix = normalize_matrix(column_matrix)

    # Создание объединённой матрицы
    row_part = row_matrix.values
    column_part = column_matrix.values
    top = np.hstack([row_part, group_matrix.values])
    bottom = np.hstack([group_matrix.T.values, column_part])
    whole_matrix = np.vstack([top, bottom])

    whole_index = all_row + list(group_matrix.columns)
    whole_matrix = pd.DataFrame(
        whole_matrix, index=whole_index, columns=whole_index)
    np.fill_diagonal(whole_matrix.values, 0)

    return whole_matrix


# Exp
# def create_whole_matrix(group_matrix: pd.DataFrame, df_data: Optional[pd.DataFrame] = None,
#                         use_co_occurrence: bool = False,
#                         skills_field: str = 'raw_skills') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     Создает полную сеть связей между навыками и профессиями/работодателями.

#     :param group_matrix: Матрица соответствия (строки – навыки, столбцы – группы/профессии/работодатели).
#     :param df_data: Исходный DataFrame с данными (используется при вычислении co-occurrence).
#     :param use_co_occurrence: Флаг использования co-occurrence матрицы навыков.
#     :param skills_field: Название столбца с навыками в df_data.
#     :return: Кортеж из трёх матриц (row_matrix, column_matrix, whole_matrix).
#     """

#     # Приведение к бинарному виду
#     group_matrix = (group_matrix > 0).astype(int)

#     # Добавление уникальных префиксов
#     group_matrix.index = ["SKILL_" + str(i) for i in group_matrix.index]
#     group_matrix.columns = ["EMPLOYER_" + str(c) for c in group_matrix.columns]

#     all_row = group_matrix.index.tolist()

#     if use_co_occurrence and df_data is not None:
#         # Создание co-occurrence матрицы навыков
#         skill_index = {skill: idx for idx, skill in enumerate(all_row)}
#         co_occurrence = np.zeros((len(all_row), len(all_row)), dtype=int)

#         for skills in df_data[skills_field]:
#             unique_skills = [
#                 f"SKILL_{s}" for s in skills if f"SKILL_{s}" in skill_index]
#             for pair in combinations(unique_skills, 2):
#                 i, j = skill_index[pair[0]], skill_index[pair[1]]
#                 co_occurrence[i, j] += 1
#                 co_occurrence[j, i] += 1

#         row_matrix = pd.DataFrame(
#             co_occurrence, index=all_row, columns=all_row)
#     else:
#         # Матрица схожести навыков через произведение
#         row_sim = group_matrix.values.dot(group_matrix.values.T)
#         row_matrix = pd.DataFrame(row_sim, index=all_row, columns=all_row)

#     np.fill_diagonal(row_matrix.values, 0)

#     # Матрица схожести работодателей (или профессий)
#     column_sim = group_matrix.T.values.dot(group_matrix.values)
#     column_matrix = pd.DataFrame(
#         column_sim, index=group_matrix.columns, columns=group_matrix.columns)
#     np.fill_diagonal(column_matrix.values, 0)

#     # Создание объединённой матрицы
#     row_part = row_matrix.values
#     column_part = column_matrix.values
#     top = np.hstack([row_part, group_matrix.values])
#     bottom = np.hstack([group_matrix.T.values, column_part])

#     # Проверка размерностей перед объединением
#     if top.shape[1] != bottom.shape[1]:
#         raise ValueError(
#             f"Несовпадение размерностей: top.shape={top.shape}, bottom.shape={bottom.shape}")

#     whole_matrix = np.vstack([top, bottom])

#     whole_index = all_row + list(group_matrix.columns)
#     whole_matrix = pd.DataFrame(
#         whole_matrix, index=whole_index, columns=whole_index)
#     np.fill_diagonal(whole_matrix.values, 0)

#     return whole_matrix

# Exp 2
# def create_bipartite_graph(matrix: pd.DataFrame) -> nx.Graph:
#     """
#     Создает двудольный граф на основе переданной матрицы.

#     :param matrix: Матрица связей (DataFrame), где строки и столбцы представляют две группы узлов.
#      По столбцам - это уровень первый (bipartite=1), а по строкам - это уровень второй (bipartite=2)
#     :return: Двудольный граф (Graph).

#     Пример использования:
#      >>> G = create_bipartite_graph(skills_roles_matrix)

#     """
#     G = nx.Graph()

#     # Извлекаем узлы для обеих групп
#     common_labels = set(matrix.index) & set(matrix.columns)

#     # Переименовываем пересекающиеся узлы
#     second_nodes = {
#         x: f"{x}_row" if x in common_labels else x for x in matrix.index}
#     first_nodes = {
#         x: f"{x}_col" if x in common_labels else x for x in matrix.columns}

#     # Добавляем узлы с указанием bipartite уровня
#     G.add_nodes_from(first_nodes.values(), bipartite=1)
#     G.add_nodes_from(second_nodes.values(), bipartite=2)

#     # Добавляем рёбра с весами
#     for row, row_name in second_nodes.items():
#         for col, col_name in first_nodes.items():
#             weight = matrix.loc[row, col]
#             if weight > 0:
#                 G.add_edge(row_name, col_name, weight=weight)

#     return G


def create_bipartite_graph(matrix: pd.DataFrame) -> nx.Graph:
    """
    Создает двудольный граф на основе переданной матрицы.

    :param matrix: Матрица связей (DataFrame), где строки и столбцы представляют две группы узлов.
     По столбцам - это уровень первый (bipartite=1), а по строкам - это уровень второй (bipartite=2)
    :return: Двудольный граф (Graph).

    Пример использования:
     >>> G = create_bipartite_graph(skills_roles_matrix)

    """
    G = nx.Graph()

    # Извлекаем узлы для обеих групп
    first_nodes = matrix.columns.tolist()
    second_nodes = matrix.index.tolist()

    # Добавляем узлы с указанием bipartite уровня
    G.add_nodes_from(first_nodes, bipartite=1)
    G.add_nodes_from(second_nodes, bipartite=2)

    # Добавляем ребра с весом
    for second in second_nodes:
        for first in first_nodes:
            weight = matrix.loc[second, first]
            if weight > 0:
                G.add_edge(first, second, weight=weight)

    return G


def generalized_jaccard(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Вычисляет обобщенный коэффициент Жаккара между двумя векторами.

    :param vector_a: Первый вектор значений.
    :param vector_b: Второй вектор значений.
    :return: Коэффициент схожести Жаккара.


    Пример использования:
     >>> similarity = generalized_jaccard(vector_a, vector_b)

    """
    min_sum = np.minimum(vector_a, vector_b).sum()
    max_sum = np.maximum(vector_a, vector_b).sum()
    return min_sum / max_sum if max_sum != 0 else 0


def recommend_similar_nodes(G: nx.Graph, target_node: str,
                            level_target: str = "first",
                            top_n: int = 5, apply_lower: bool = False) -> None:
    """
    Рекомендует схожие узлы в двудольном графе на основе обобщенного коэффициента Жаккара.

    :param G: Двудольный граф (Graph).
    :param target_node: Целевой узел для поиска схожих узлов.
    :param level_target: Уровень узла ('first' или 'second').
     По столбцам - это уровень первый (bipartite=1), а по строкам - это уровень второй (bipartite=2)
    :param top_n: Количество рекомендаций.
    :param apply_lower: Приводить ли узел ко второму регистру.


    Пример использования:
    Для первого уровня (столбца):
     >>> recommend_similar_nodes(G, "Монтажник", level_target="first")
    Для второго уровня (строки) с необходимость приведения к нижнему регистру:
     >>> recommend_similar_nodes(G, "Excel", level_target="second", apply_lower=True)

    """
    expected_level = 1 if level_target == 'first' else 2
    processed_target = target_node.lower(
    ) if apply_lower and level_target == 'second' else target_node

    # if processed_target not in G:
    #     raise ValueError(f"Узел '{processed_target}' не найден в графе.")

    # if processed_target not in G:
    #     return None

    target_neighbors = {nbr: G[processed_target][nbr]['weight']
                        for nbr in G.neighbors(processed_target)}
    recommendations = []

    for node in G.nodes:
        if node == processed_target or G.nodes[node].get('bipartite') != expected_level:
            continue

        node_neighbors = {nbr: G[node][nbr]['weight']
                          for nbr in G.neighbors(node)}
        union_keys = set(target_neighbors) | set(node_neighbors)
        vector_a = np.array([target_neighbors.get(k, 0) for k in union_keys])
        vector_b = np.array([node_neighbors.get(k, 0) for k in union_keys])

        similarity = generalized_jaccard(vector_a, vector_b)
        recommendations.append((node, similarity))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]


def neighbor_recommendations(G: nx.Graph, target_node: str,
                             level_target: str = "first",
                             top_n: int = 5,
                             apply_lower: bool = False) -> None:
    """
    Рекомендует соседние узлы (навыки или профессии) в двудольном графе.

    :param G: Двудольный граф (Graph).
    :param target_node: Целевой узел.
    :param level_target: Уровень узла ('first' или 'second').
     По столбцам - это уровень первый (bipartite=1), а по строкам - это уровень второй (bipartite=2)
    :param top_n: Количество рекомендаций.
    :param apply_lower: Приводить ли узел к нижнему регистру.


    Пример использования:
    Для первого уровня (столбца):
     >>> neighbor_recommendations(G, "Монтажник")
    Для второго уровня (строки) с необходимость приведения к нижнему регистру:
     >>> neighbor_recommendations(G, "Монтаж металлоконструкций", level_target="second", apply_lower=True)

    """
    target_node = target_node.lower() if apply_lower else target_node
    recommendations = []

    if level_target == 'first':
        recommendations = [(nbr, G[target_node][nbr]['weight']) for nbr in G.neighbors(
            target_node) if G.nodes[nbr].get('bipartite') == 2]
    elif level_target == 'second':
        recommendations = [(nbr, G[nbr][target_node]['weight']) for nbr in G.neighbors(
            target_node) if G.nodes[nbr].get('bipartite') == 1]

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]


def parse_skills(s):
    if pd.isna(s):
        return []
    return [skill.strip() for skill in s.split(';') if skill.strip()]


def filter_graph(graph, threshold):
    """
    Функция принимает граф networkx, фильтрует его по заданному порогу веса ребер,
    удаляя ребра с весом ниже порога, а затем удаляет узлы без ребер (изолированные узлы).

    Параметры:
    - graph: networkx.Graph (или другой тип графа), где ребра имеют атрибут 'weight'
    - threshold: числовой порог для фильтрации ребер.

    Возвращает:
    - networkx.Graph: отфильтрованный граф.
    """
    # Копируем граф, чтобы не изменять исходный
    filtered_graph = graph.copy()

    # Проходим по всем ребрам (используем list для безопасной итерации)
    for u, v, data in list(filtered_graph.edges(data=True)):
        # если атрибут отсутствует, считаем вес равным 0
        weight = data.get('weight', 0)
        if weight < threshold:
            filtered_graph.remove_edge(u, v)

    # Удаляем узлы без ребер (изолированные)
    isolated_nodes = [node for node in filtered_graph.nodes()
                      if filtered_graph.degree(node) == 0]
    filtered_graph.remove_nodes_from(isolated_nodes)

    return filtered_graph

from pathlib import Path
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from qdrant_client import models, QdrantClient
from uuid import uuid4


def create_qdrant_collection(collection_name: str, vector_size: int = 384, distance_metric=models.Distance.COSINE) -> QdrantClient:
    """
    Создает коллекцию в Qdrant с заданными параметрами.

    :param collection_name: имя коллекции
    :param vector_size: размер векторов
    :param distance_metric: метрика расстояния (по умолчанию COSINE)
    :return: экземпляр QdrantClient
    """
    qdrant_client = QdrantClient(":memory:")
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=distance_metric
        )
    )
    return qdrant_client


def create_vector_store(qdrant_client: QdrantClient, collection_name: str, embeddings, documents: list) -> QdrantVectorStore:
    """
    Создает векторное хранилище и добавляет документы с уникальными ID.

    :param qdrant_client: клиент Qdrant
    :param collection_name: имя коллекции
    :param embeddings: объект или функция эмбеддингов
    :param documents: список документов для добавления
    :return: экземпляр QdrantVectorStore
    """
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    # Генерируем уникальные идентификаторы для каждого документа
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    return vector_store


def create_retrievers(vector_store: QdrantVectorStore, documents: list) -> EnsembleRetriever:
    """
    Создает два ретривера:
      1. Ретривер на основе similarity поиска
      2. BM25 ретривер

    :param vector_store: объект векторного хранилища
    :param documents: список документов для BM25 ретривера
    :param k: количество возвращаемых документов (по умолчанию 10)
    :param score_threshold: порог оценки (если требуется)
    :return: кортеж (similarity_retriever, bm25_retriever)
    """
    # Ретривер по эмбеддингам
    similarity_retriever = vector_store.as_retriever(
        search_type="similarity",
        k=10,
        score_threshold=None,
    )
    # BM25 ретривер (эмбеддинги ему не нужны)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 10

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, similarity_retriever],
        weights=[0.8, 0.2],
    )
    return ensemble


def format_docs(docs) -> str:
    """
    Формирует строку из списка документов.

    :param docs: список документов с атрибутами page_content и metadata
    :return: объединенная строка с информацией из документов
    """
    data = "\n\n".join(
        f"{doc.page_content}\n{doc.metadata.get('Название региона', '')}\nОпыт работы: {doc.metadata.get('Опыт работы', '')}"
        for doc in docs
    )
    return data


def delete_qdrant_collection(qdrant_client: QdrantClient, collection_name: str):
    """
    Удаляет коллекцию из Qdrant.

    :param qdrant_client: экземпляр клиента Qdrant
    :param collection_name: имя коллекции для удаления
    """
    qdrant_client.delete_collection(collection_name=collection_name)


# ------ Переменнные -------
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2")

template = """
Ты профессиональный ассистент по анализу вакансий, навыков и работодателей.
Принципы твоей работы:
1. Отвечаешь на основе контекста с важными дополнениями из своих знаний по вопросу.
2. Если в контексте нет ответа на вопрос, то ты должен написать: 
"Извините данные отсутствуют, но я могу Вам порекомендовать следующее:" и немного дополни сам
3. Ты должен отвечать грамотно, вежливо и профессионально.
4. Нужно дополнять контекст важной информацией. Например, при перечислении навыков кратко описать их суть.
5. Пример шаблона ответа на вопрос: "Какие навыки нужны аналитику данных с опытом работы 1-3 года?"
Ответ:
"Для позиции аналитика данных с опытом 1-3 года ключевые навыки, основанные на вакансиях из контекста, можно разделить на несколько категорий:

---

### **Технические навыки:**
1. **Работа с данными:**
   - **SQL** (основной язык для запросов к реляционным БД).
   - **Oracle Database** (используется в банковской сфере, например, в АБСОЛЮТ БАНК).
   - **NoSQL** (для неструктурированных данных, упомянут в некоторых вакансиях Data Scientist).

2. **Инструменты анализа:**
   - **Tableau** (визуализация данных, требуется в Emerging Travel Group).
   - **Google Analytics** (веб-аналитика, упомянута в Онлайнтурс).
   - **Microsoft Excel** (базовый анализ и отчетность).

3. **Программирование:**
   - **Python** (для автоматизации задач и анализа, встречается в вакансиях Data Scientist, но полезен и для аналитиков).
   - **HTML/CSS, JavaScript** (для веб-аналитики и взаимодействия с фронтендом, например, в Онлайнтурс).

---

### **Отраслевые знания:**
- **Банковская сфера:**  
  - Работа с банковскими продуктами для юрлиц/физлиц (АБСОЛЮТ БАНК, Онлайнтурс).  
  - Знание расчетно-кассового обслуживания и нормативно-правовых актов (РОСБАНК).  
- **Маркетинг и SEO:**  
  - Поисковая оптимизация (SEO), Key Collector, анализ конкурентов (Онлайнтурс).  
  - Управление брендом и основы маркетинга (Вайтлист).  

---

### **Мягкие навыки:**
- **Коммуникация:**  
  - Взаимодействие с клиентами, формирование ТЗ (АБСОЛЮТ БАНК, Онлайнтурс).  
  - Консультирование и поддержка пользователей (Emerging Travel Group).  
- **Обучаемость:**  
  - Адаптивность к новым инструментам и процессам (АБСОЛЮТ БАНК).  
- **Работа в команде:**  
  - Упоминается в большинстве вакансий, включая Emerging Travel Group и Онлайнтурс.  

---

### **Дополнительные требования:**
- **Управление продуктом:**  
  - Разработка документации, управление задачами в Asana (Emerging Travel Group, МГТС).  
- **Работа с CRM-системами:**  
  - Требуется в QIC для аналитики в контексте продаж.  

---

### **Рекомендации (если выходить за рамки контекста):**
- Изучите **Power BI** как альтернативу Tableau.  
- Освойте основы **статистики** (A/B-тестирование, гипотезы).  
- Практикуйтесь в написании **SQL-запросов** на платформах вроде LeetCode.  
- Развивайте навыки **презентации данных** (например, через курсы по Data Storytelling).  

Примеры из вакансий показывают, что требования зависят от отрасли: в банках акцент на нормативы и продукты, в IT-компаниях — на технические инструменты и аналитику в продукте".

5. Ты должен отвечать на вопросы на основе следующего контекста:
{context}

Question: {question}
"""
# Создаём промпт

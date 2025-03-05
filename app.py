import pickle
from pathlib import Path
from shutil import copyfile

import networkx as nx
from ipysigma import Sigma
from shinyswatch import theme
from shiny import reactive, req
from shiny.express import input, ui, render
from shinywidgets import render_widget, render_plotly
import pandas as pd
import netfunction
import plotly.express as px
from faicons import icon_svg
import plotly.graph_objects as go

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

from rag_chat import create_qdrant_collection, create_vector_store, create_retrievers, format_docs, template

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Определяем базовую директорию относительно текущего файла
here = Path(__file__).parent
# Создаем поддиректорию для временных файлов, если её нет
temp_dir = here / "temp"
temp_dir.mkdir(exist_ok=True)

# Путь к файлу векторного хранилища будет строиться относительно temp_dir
vector_store_file = temp_dir / "vector_store.pkl"

# Настройки страницы
ui.page_opts(
    title=ui.div(
        icon_svg("vector-square"),      # Иконка сети из faicons
        " Network Dashboard",
        style="display: flex; align-items: center;"
    ),
    fillable=True,
    id="page",
    theme=theme.journal
)

# Sidebar: Обработка данных и универсальные фильтры для графов
with ui.sidebar(width=350):
    ui.HTML("<h5> ⚙️Обработка данных</h5>")
    ui.hr()
    with ui.card(full_screen=False):
        ui.input_file("file", "Загрузить данные:", accept=".xlsx", width=200,
                      button_label='Обзор', placeholder='Файл отсутствует')
    ui.hr()


@reactive.calc
def df():
    f = req(input.file())
    return pd.read_excel(f[0]['datapath'])


@reactive.calc
def processed_data():
    data = df()
    data['Обработанные навыки'] = data['Ключевые навыки'].apply(
        netfunction.parse_skills)
    data = data.dropna(subset=['Работодатель', 'Ключевые навыки'])
    data.reset_index(inplace=True, drop=True)
    data['Дата публикации'] = pd.to_datetime(data['Дата публикации'])
    data["Федеральный округ"] = data["Название региона"].apply(
        netfunction.get_federal_district)
    data['Данные'] = data.apply(lambda row:
                                f"{row['Работодатель']} ищет "
                                f"{row['Название специальности']} с {row['Ключевые навыки']}.", axis=1)
    return data


ui.nav_spacer()
with ui.nav_panel("Данные", icon=icon_svg("table")):
    with ui.card(full_screen=True):
        ui.card_header("📖 Загруженные данные")

        @render.data_frame
        def table():
            return render.DataTable(processed_data(), filters=True, height='550px')

# Обработка данных и создание векторного хранилища при нажатии кнопки "Обработать и векторизовать данные"
sample_set = set()  # Храним уже обработанные размеры выборки


@reactive.event(input.process_data)
def build_vector_store():
    data = processed_data()  # Получаем обработанный DataFrame
    if data.empty:
        # Удаляем все файлы хранилища, если данных нет
        for file in temp_dir.glob("vector_store_*.pkl"):
            file.unlink()
        return

    sample_size = input.sample_size()
    # Уникальное имя для каждого sample_size относительно temp_dir
    vector_store_file = temp_dir / f"vector_store_{sample_size}.pkl"

    # Если выборка с таким sample_size уже есть, загружаем существующее хранилище
    if vector_store_file.exists():
        with open(vector_store_file, "rb") as f:
            ensemble = pickle.load(f)
        print(
            f"Загружено существующее векторное хранилище для sample_size={sample_size}.")
        return ensemble

    sample_set.add(sample_size)
    # Формируем выборку данных
    data_sample = data[['Название региона', 'Данные', 'Опыт работы']].sample(
        sample_size, random_state=1)
    print("Размер выборки:", data_sample.shape)

    loader = DataFrameLoader(data_sample, page_content_column="Данные")
    docs = loader.load()
    print("Документов загружено:", len(docs))

    text_splitter = RecursiveCharacterTextSplitter()
    split_docs = text_splitter.split_documents(docs)
    print("Фрагментов:", len(split_docs))

    collection_name = 'navyk'
    client = create_qdrant_collection(collection_name=collection_name)
    vs = create_vector_store(client, collection_name, embeddings, split_docs)
    ensemble = create_retrievers(vs, split_docs)
    print(f"Векторное хранилище создано для sample_size={sample_size}.")

    # Сохраняем векторное хранилище в отдельный файл относительно temp_dir
    with open(vector_store_file, "wb") as f:
        pickle.dump(ensemble, f)
    print(f"Векторное хранилище сохранено в {vector_store_file}")

    return ensemble


@reactive.effect
def update_models():
    if input.base_url() == "https://bothub.chat/api/v2/openai/v1":
        models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini",
                  "o3-mini", "o1-mini"]
        ui.update_selectize("chat_model", choices=models)
    elif input.base_url() == "https://openrouter.ai/api/v1":
        models = ["google/gemini-2.0-flash-thinking-exp:free",
                  "deepseek/deepseek-chat:free",
                  "deepseek/deepseek-r1:free"]
        ui.update_selectize("chat_model", choices=models)


prompt = ChatPromptTemplate.from_template(template)

with ui.nav_panel("Чат-бот", icon=icon_svg('robot')):
    with ui.layout_columns(col_widths=(4, 8)):
        # Левая колонка: Фильтры для чат-бота
        with ui.card(full_screen=False):
            ui.card_header("🔎 Фильтры для чат-бота")
            ui.input_password("chat_token", "API-Токен сервиса:",
                              width='400px', placeholder="Введите токен")
            ui.input_selectize("chat_model", "Языковая модель:",
                               choices=[], width='400px')
            ui.input_selectize("base_url", "Базовый URL-адрес сервиса:",
                               choices=["https://bothub.chat/api/v2/openai/v1",
                                        "https://api.deepseek.com", "https://openrouter.ai/api/v1"],
                               selected='https://openrouter.ai/api/v1', width='400px')
            ui.input_slider("temp", "Температура:", min=0,
                            max=1.5, value=0, step=0.1, width='400px')
            ui.hr()
            ui.input_slider("sample_size", "Размер выборки для фильтрации данных:",
                            min=100, max=2000, value=400, step=100, width='400px')
            ui.input_action_button(
                "process_data", "Обработать и векторизовать данные", width="400px")

        # Правая колонка: Чат-бот
        with ui.card(full_screen=True):
            ui.card_header("🤖 Чат-бот")
            welcome = ui.markdown("Hi!")
            chat = ui.Chat(id="chat", messages=[welcome])
            chat.ui(placeholder='Введите запрос...', width='min(850px, 100%)')

            @chat.on_user_submit
            async def process_chat():
                user_message = chat.user_input()
                if user_message == "Очистить чат":
                    await chat.clear_messages()
                    await chat.append_message_stream('Чат очищен ✅')
                    return

                # Загружаем векторное хранилище из файла, путь строим относительно temp_dir
                try:
                    build_vector_store()
                    vector_store_file = temp_dir / \
                        f"vector_store_{input.sample_size()}.pkl"
                    if not vector_store_file.exists():
                        await chat.append_message("Данные не обработаны. Пожалуйста, нажмите кнопку 'Обработать и векторизовать данные'.")
                        return
                except Exception as e:
                    await chat.append_message(f'Данные не загружены, пожалуйста, загрузите: {e}')
                    return

                try:
                    with open(vector_store_file, "rb") as f:
                        ensemble = pickle.load(f)
                except Exception as e:
                    await chat.append_message(f"Ошибка загрузки векторного хранилища: {e}")
                    return

                model = input.chat_model()
                temperature = input.temp()
                base_url = input.base_url()
                api_key = input.chat_token() or None

                try:
                    llm = ChatOpenAI(model_name=model,
                                     temperature=temperature,
                                     max_tokens=6000,
                                     base_url=base_url,
                                     openai_api_key=api_key)
                    llm_chain = (
                        {"context": ensemble | format_docs,
                         "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    response = llm_chain.invoke(user_message)
                    await chat.append_message_stream(response)
                except Exception as e:
                    await chat.append_message(f'Извините, произошла ошибка: {str(e)}')

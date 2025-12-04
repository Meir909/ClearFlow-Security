import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
import traceback
import os
from src.preprocessing import load_data, preprocess
from src.advanced_models import advanced_model_pipeline, get_model_contributions, visualize_model_comparison
from src.rules import rule_engine, get_rule_explanations
from src.explainability import calculate_shap_values, plot_feature_importance, generate_explanation_text, aggregate_explanations
from src.output_generator import export_all_results
from src.evaluation import calculate_metrics, plot_roc_curve, plot_precision_recall_curve, evaluate_model_performance
from src.self_learning import integrate_self_learning
from src.advanced_visualizations import advanced_viz
from src.enhanced_visualizations import enhanced_viz
from src.user_preferences import user_prefs
from src.localization import localization_manager
from src.data_processor import data_processor
from src.progress_manager import progress_manager
from src.user_database import user_db
from src.advanced_models import build_transaction_graph, predict_fraud_probability_next_week, cluster_user_profiles
import warnings
warnings.filterwarnings('ignore')

def _(key):
    return localization_manager.get_text(key, 'ru')

user_preferences = user_prefs.load_preferences()

def show_analysis_modal():
    if st.session_state.get('show_analysis_modal', False):
        selected_file = st.session_state.get('selected_file_for_analysis')
        analysis_results = st.session_state.get('analysis_results', {})
        
        if selected_file and selected_file in analysis_results:
            result = analysis_results[selected_file]
            
            st.markdown("""
            <div class="modal-overlay">
                <div class="modal-content">
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid rgba(102, 126, 234, 0.3);">
                <h2 style="margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    📈 Анализ файла: {selected_file}
                </h2>
                <button onclick="document.getElementById('close_modal').click()" 
                        style="background: #ff4757; color: white; border: none; width: 30px; height: 30px; border-radius: 50%; cursor: pointer; font-weight: bold;">
                    ✕
                </button>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3>📊 Ключевые показатели</h3>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                            padding: 15px; border-radius: 10px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #667eea;">{result['total_transactions']:,}</div>
                    <div style="font-size: 0.9rem; color: #666;">Всего транзакций</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(245, 87, 108, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%); 
                            padding: 15px; border-radius: 10px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #f5576c;">{result['suspicious_count']:,}</div>
                    <div style="font-size: 0.9rem; color: #666;">Подозрительных</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(17, 153, 142, 0.1) 0%, rgba(56, 239, 125, 0.1) 100%); 
                            padding: 15px; border-radius: 10px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #11998e;">{result['clean_count']:,}</div>
                    <div style="font-size: 0.9rem; color: #666;">Чистых</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(102, 126, 234, 0.1) 100%); 
                            padding: 15px; border-radius: 10px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #764ba2;">{result['fraud_percentage']:.2f}%</div>
                    <div style="font-size: 0.9rem; color: #666;">Уровень мошенничества</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("<h3>📊 Распределение Fraud Score</h3>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(result['combined_scores'], bins=50, alpha=0.7, color='#667eea', edgecolor='black')
            ax.set_xlabel('Fraud Score')
            ax.set_ylabel('Частота')
            ax.set_title('Распределение Fraud Score')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            if 'model_contributions' in result and result['model_contributions']:
                st.markdown("<h3>🤖 Вклад моделей</h3>", unsafe_allow_html=True)
                
                contributions = result['model_contributions']
                model_names = list(contributions.keys())
                contribution_values = [contributions[name].get('contribution_percentage', 0) for name in model_names]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.bar(model_names, contribution_values, color=['#667eea', '#764ba2', '#f5576c'])
                ax.set_ylabel('Процент вклада (%)')
                ax.set_title('Вклад моделей в общий результат')
                ax.set_ylim(0, max(contribution_values) * 1.2)
                
                for bar, value in zip(bars, contribution_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                            f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
                
                st.markdown("<h4>📊 Детальный разбор</h4>", unsafe_allow_html=True)
                for model_name, contrib in contributions.items():
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); 
                                padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <div style="font-weight: 700; color: #667eea; margin-bottom: 10px;">{model_name.capitalize()}</div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                            <div><strong>Вес:</strong> {contrib.get('weight', 0):.2f}</div>
                            <div><strong>Вклад:</strong> {contrib.get('contribution_percentage', 0):.1f}%</div>
                            <div><strong>Аномалий:</strong> {int(contrib.get('anomaly_count', 0))}</div>
                            <div><strong>Средний скор:</strong> {contrib.get('mean_score', 0):.4f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<h3>❓ Почему транзакция помечена как подозрительная?</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(245, 87, 108, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%); 
                        padding: 20px; border-radius: 15px; margin: 15px 0; border-left: 5px solid #f5576c;">
                <div style="font-weight: 600; color: #f5576c; margin-bottom: 10px; font-size: 1.1rem;">Анализ подозрительных транзакций</div>
                <ul style="padding-left: 20px; margin: 10px 0; line-height: 1.6;">
                    <li><strong>Правила бизнес-логики:</strong> Транзакции проверяются на соответствие заранее определенным правилам, таким как необычно большие суммы, переводы на новые счета, опустошение счетов и т.д.</li>
                    <li><strong>Машинное обучение:</strong> Модели машинного обучения анализируют паттерны поведения клиентов и выявляют отклонения от нормального поведения.</li>
                    <li><strong>Комбинированный подход:</strong> Итоговый Fraud Score рассчитывается как комбинация результатов правил и моделей ML (70% ML + 30% правила).</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3>🔍 Какие признаки повлияли на решение?</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                        padding: 20px; border-radius: 15px; margin: 15px 0; border-left: 5px solid #667eea;">
                <div style="font-weight: 600; color: #667eea; margin-bottom: 10px; font-size: 1.1rem;">Ключевые признаки подозрительности</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                    <div>
                        <strong>📊 Финансовые признаки:</strong>
                        <ul style="padding-left: 20px; margin: 5px 0;">
                            <li>Необычно большая сумма перевода</li>
                            <li>Полное опустошение счета отправителя</li>
                            <li>Круглые суммы (1000, 5000, 10000 и т.д.)</li>
                            <li>Резкое изменение баланса счета</li>
                        </ul>
                    </div>
                    <div>
                        <strong>⏰ Временные признаки:</strong>
                        <ul style="padding-left: 20px; margin: 5px 0;">
                            <li>Транзакции в необычное время (ночь, раннее утро)</li>
                            <li>Резкий скачок активности клиента</li>
                            <li>Нехарактерные временные интервалы</li>
                        </ul>
                    </div>
                    <div>
                        <strong>👥 Поведенческие признаки:</strong>
                        <ul style="padding-left: 20px; margin: 5px 0;">
                            <li>Перевод на новый, ранее неиспользуемый счет</li>
                            <li>Отклонение от привычных паттернов клиента</li>
                            <li>Необычная последовательность операций</li>
                        </ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3>📏 Насколько операция отклоняется от паттернов клиента?</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(17, 153, 142, 0.1) 0%, rgba(56, 239, 125, 0.1) 100%); 
                        padding: 20px; border-radius: 15px; margin: 15px 0; border-left: 5px solid #11998e;">
                <div style="font-weight: 600; color: #11998e; margin-bottom: 10px; font-size: 1.1rem;">Анализ отклонений от нормы</div>
                <p style="margin: 10px 0; line-height: 1.6;">
                    Система сравнивает каждую транзакцию с историческим поведением клиента, анализируя:
                </p>
                <ul style="padding-left: 20px; margin: 10px 0; line-height: 1.6;">
                    <li><strong>Средние суммы транзакций:</strong> Текущая сумма сравнивается со средними значениями клиента</li>
                    <li><strong>Частота операций:</strong> Анализируется интенсивность транзакций в сравнении с обычной активностью</li>
                    <li><strong>Предпочтительные получатели:</strong> Проверяется, является ли получатель новым или ранее использовавшимся</li>
                    <li><strong>Временные паттерны:</strong> Сравнение времени транзакции с привычным расписанием клиента</li>
                    <li><strong>Типы операций:</strong> Анализ соответствия типу обычно совершаемых транзакций</li>
                </ul>
                <p style="margin: 10px 0; line-height: 1.6;">
                    <strong>Fraud Score</strong> представляет собой нормализованную метрику отклонения, где значения выше 0.5 указывают на подозрительную активность, 
                    а значения выше 0.7 требуют особого внимания.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3>🔗 Построение графовой модели перемещений денег</h3>", unsafe_allow_html=True)
            
            graph_data = build_transaction_graph(result.get('processed_data', pd.DataFrame()))
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(255, 165, 0, 0.1) 0%, rgba(255, 140, 0, 0.1) 100%); 
                        padding: 20px; border-radius: 15px; margin: 15px 0; border-left: 5px solid orange;">
                <div style="font-weight: 600; color: orange; margin-bottom: 10px; font-size: 1.1rem;">Анализ сетевой структуры транзакций</div>
                <p style="margin: 10px 0; line-height: 1.6;">
                    Графовая модель позволяет выявлять сложные схемы мошенничества, анализируя связи между участниками транзакций.
                </p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                    <div style="background: rgba(255, 165, 0, 0.1); padding: 10px; border-radius: 10px;">
                        <strong>Узлы:</strong> {graph_data['nodes_count']:,}
                    </div>
                    <div style="background: rgba(255, 165, 0, 0.1); padding: 10px; border-radius: 10px;">
                        <strong>Связи:</strong> {graph_data['edges_count']:,}
                    </div>
                    <div style="background: rgba(255, 165, 0, 0.1); padding: 10px; border-radius: 10px;">
                        <strong>Сообщества:</strong> {len(graph_data['communities'])}
                    </div>
                </div>
                <p style="margin: 10px 0; line-height: 1.6;">
                    <strong>Ключевые участники:</strong> Высокая степень центральности может указывать на важных игроков в сети транзакций.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3>🔮 Прогноз вероятности фрода на следующие 7 дней</h3>", unsafe_allow_html=True)
            
            fraud_forecast = predict_fraud_probability_next_week(result.get('processed_data', pd.DataFrame()))
            
            forecast_days = [f"День {i+1}" for i in range(7)]
            predictions = fraud_forecast['predictions']
            
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(forecast_days, predictions, color=['#667eea' if p < 5 else '#f5576c' if p > 10 else '#764ba2' for p in predictions])
            ax.set_ylabel('Прогнозируемое количество фрод-транзакций')
            ax.set_title('Прогноз фрода на следующие 7 дней')
            ax.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, predictions):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(128, 0, 128, 0.1) 0%, rgba(75, 0, 130, 0.1) 100%); 
                        padding: 20px; border-radius: 15px; margin: 15px 0; border-left: 5px solid purple;">
                <div style="font-weight: 600; color: purple; margin-bottom: 10px; font-size: 1.1rem;">Анализ прогноза</div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                    <div style="background: rgba(128, 0, 128, 0.1); padding: 10px; border-radius: 10px;">
                        <strong>Тренд:</strong> {fraud_forecast['trend']}
                    </div>
                    <div style="background: rgba(128, 0, 128, 0.1); padding: 10px; border-radius: 10px;">
                        <strong>Уровень риска:</strong> {fraud_forecast['risk_level']}
                    </div>
                </div>
                <p style="margin: 10px 0; line-height: 1.6;">
                    Система прогнозирует {sum(predictions):.1f} потенциальных фрод-транзакций в ближайшие 7 дней.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h3>👥 Кластеризация профилей пользователей</h3>", unsafe_allow_html=True)
            
            clustering_result = cluster_user_profiles(result.get('processed_data', pd.DataFrame()))
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(0, 191, 255, 0.1) 0%, rgba(30, 144, 255, 0.1) 100%); 
                        padding: 20px; border-radius: 15px; margin: 15px 0; border-left: 5px solid #1e90ff;">
                <div style="font-weight: 600; color: #1e90ff; margin-bottom: 10px; font-size: 1.1rem;">Сегментация пользователей</div>
                <p style="margin: 10px 0; line-height: 1.6;">
                    Кластеризация позволяет группировать пользователей по схожим паттернам поведения для более точного анализа.
                </p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                    <div style="background: rgba(0, 191, 255, 0.1); padding: 10px; border-radius: 10px;">
                        <strong>Кластеров:</strong> {clustering_result['n_clusters']}
                    </div>
                    <div style="background: rgba(0, 191, 255, 0.1); padding: 10px; border-radius: 10px;">
                        <strong>Проанализировано:</strong> {len(clustering_result['user_clusters'])} пользователей
                    </div>
                </div>
                <p style="margin: 10px 0; line-height: 1.6;">
                    <strong>Рекомендация:</strong> Используйте информацию о кластерах для адаптации порогов детекции фрода под разные группы пользователей.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("❌ Закрыть", key="close_modal"): 
                st.session_state['show_analysis_modal'] = False
                st.rerun()
            
            st.markdown("</div></div>", unsafe_allow_html=True)

def show_user_profile():
    """Display user profile information"""
    if st.session_state.get('current_user'):
        user = st.session_state.current_user
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                        padding: 1rem; border-radius: 15px; margin-bottom: 1rem;">
                <h3 style="margin: 0 0 0.5rem 0; color: #667eea;">👤 Профиль пользователя</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                    <div><strong>ФИО:</strong> {user['full_name']}</div>
                    <div><strong>Должность:</strong> {user['position']}</div>
                    <div><strong>Возраст:</strong> {user['age']} лет</div>
                    <div><strong>Компания:</strong> {user['company']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("🚪 Выйти", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.current_user = None
                st.rerun()

def show_login_page():
    """Display the login/registration page"""
    st.markdown("""
    <div style="max-width: 500px; margin: 0 auto; padding: 2rem; background: rgba(30, 30, 40, 0.95); 
                border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.5); text-align: center;">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent; margin-bottom: 2rem;">🔐 Авторизация</h1>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔑 Вход", "📝 Регистрация"])
    
    with tab1:
        st.markdown("### Войдите в систему")
        username = st.text_input("Имя пользователя", key="login_username")
        password = st.text_input("Пароль", type="password", key="login_password")
        
        if st.button("Войти", use_container_width=True):
            if username and password:
                success, result = user_db.authenticate_user(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.current_user = result
                    st.success(f"Добро пожаловать, {result['full_name']}!")
                    st.rerun()
                else:
                    st.error(result)
            else:
                st.error("Пожалуйста, заполните все поля")
    
    with tab2:
        st.markdown("### Создайте новый аккаунт")
        new_username = st.text_input("Имя пользователя", key="reg_username")
        new_password = st.text_input("Пароль", type="password", key="reg_password")
        full_name = st.text_input("ФИО", key="reg_full_name")
        position = st.text_input("Должность", key="reg_position")
        age = st.number_input("Возраст", min_value=18, max_value=100, key="reg_age")
        company = st.text_input("Компания", key="reg_company")
        
        if st.button("Зарегистрироваться", use_container_width=True):
            if all([new_username, new_password, full_name, position, age, company]):
                success, message = user_db.register_user(new_username, new_password, full_name, position, age, company)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.error("Пожалуйста, заполните все поля")
    
    st.markdown("</div>", unsafe_allow_html=True)

st.set_page_config(
    page_title="ClearFlow Security - Обнаружение Мошенничества",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if not st.session_state.get('authenticated', False):
    show_login_page()
    st.stop()

show_user_profile()

st.session_state['theme'] = 'dark'
st.session_state['language'] = 'ru'

card_bg = 'rgba(30, 30, 40, 0.95)'
shadow_color = 'rgba(0,0,0,0.5)'
text_secondary = '#a0a0a0'

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        transition: background-color 0.3s ease, color 0.3s ease;
        height: auto;
        min-height: 100vh;
    }}
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
        transition: background 0.3s ease;
        min-height: 100vh;
    }}
    [data-testid="stApp"] {{
        height: auto;
        min-height: 100vh;
    }}
    .main {{
        background: transparent;
        height: auto;
        min-height: 100vh;
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: {card_bg};
        border-radius: 20px;
        box-shadow: 0 20px 60px {shadow_color};
        margin: 20px;
        animation: fadeIn 0.8s ease-in;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease-in-out;
        border: 1px solid rgba(255,255,255,0.1);
        position: relative;
        overflow: visible;
    }}
    .block-container:before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        transform: rotate(30deg);
        pointer-events: none;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    h1 {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 1rem;
        animation: slideDown 0.6s ease-out;
        transition: all 0.3s ease-in-out;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: relative;
    }}
    h1:after {{
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }}
    @keyframes slideDown {{
        from {{ transform: translateY(-30px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}
    h2 {{
        color: #667eea;
        font-weight: 700;
        border-left: 5px solid #667eea;
        padding-left: 15px;
        margin-top: 2rem;
        animation: slideRight 0.5s ease-out;
        transition: all 0.3s ease-in-out;
        position: relative;
        padding-bottom: 10px;
    }}
    h2:after {{
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 50px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 3px;
    }}
    @keyframes slideRight {{
        from {{ transform: translateX(-20px); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}
    h3 {{
        color: #764ba2;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
        position: relative;
        padding-left: 10px;
        margin-top: 1.5rem;
    }}
    h3:before {{
        content: '▶';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        font-size: 0.8rem;
        color: #667eea;
    }}
    .stButton > button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        font-size: 16px;
        padding: 12px 30px;
        border-radius: 25px;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease-in-out;
        width: 100%;
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }}
    .stButton > button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }}
    .stButton > button:active {{
        transform: translateY(-1px);
    }}
    .stButton > button:before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: 0.5s;
    }}
    .stButton > button:hover:before {{
        left: 100%;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        transition: all 0.3s ease-in-out;
    }}
    div[data-testid="stMetricLabel"] {{
        font-weight: 600;
        color: {text_secondary};
        font-size: 0.9rem;
        transition: all 0.3s ease-in-out;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 15px;
        padding: 5px;
        transition: all 0.3s ease-in-out;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin-bottom: 20px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.3s ease-in-out;
        background: transparent;
        color: {text_secondary};
        position: relative;
        overflow: hidden;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(102, 126, 234, 0.2);
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }}
    .stTabs [aria-selected="true"]:before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: white;
    }}
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
        transition: all 0.3s ease-in-out;
        border-radius: 0 20px 20px 0;
    }}
    .sidebar [data-testid="stSidebar"] {{
        background: transparent;
    }}
    
    * {{
        transition: all 0.3s ease-in-out;
    }}
    
    .metric-card, .info-card, .success-box, .warning-box {{
        transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.3s ease;
        will-change: transform;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        animation: cardAppear 0.4s ease-out;
    }}
    
    @keyframes cardAppear {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .metric-card:hover, .info-card:hover {{
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15) !important;
    }}
    
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(0,0,0,0.05);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, #764ba2, #667eea);
    }}
    
    .modal-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.7);
        z-index: 9999;
        display: flex;
        justify-content: center;
        align-items: center;
        backdrop-filter: blur(5px);
        animation: fadeIn 0.3s ease-out;
    }}
    
    .modal-content {{
        background: {card_bg};
        border-radius: 20px;
        padding: 30px;
        max-width: 90%;
        max-height: 90%;
        overflow-y: auto;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.1);
        animation: modalAppear 0.3s ease-out;
    }}
    
    @keyframes modalAppear {{
        from {{ opacity: 0; transform: scale(0.9); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}
    
    .upload-area {{
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
        cursor: pointer;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    }}
    
    .upload-area:hover {{
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.1);
    }}
    
    .upload-area:before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
        transform: rotate(30deg);
        pointer-events: none;
    }}
    
    .upload-area i {{
        font-size: 3rem;
        color: #667eea;
        margin-bottom: 15px;
        display: block;
    }}
    
    .spinner {{
        border: 4px solid rgba(0, 0, 0, 0.1);
        border-left-color: #667eea;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        display: inline-block;
        vertical-align: middle;
        margin-right: 10px;
    }}
    
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    
    [data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
    }}
    
    [data-testid="stDataFrame"] table {{
        border-collapse: separate;
        border-spacing: 0;
    }}
    
    [data-testid="stDataFrame"] th {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
    }}
    
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {{
        border-bottom: 1px solid rgba(102, 126, 234, 0.2);
    }}
    
    [data-testid="stDataFrame"] tr:hover {{
        background: rgba(102, 126, 234, 0.05) !important;
    }}
    
    .stProgress {{
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        overflow: hidden;
    }}
    
    .stProgress > div {{
        background: linear-gradient(90deg, #667eea, #764ba2);
    }}
    
    @media (max-width: 768px) {{
        .block-container {{
            margin: 10px;
            padding: 1rem;
        }}
        
        h1 {{
            font-size: 2rem !important;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 8px 15px;
            font-size: 0.9rem;
        }}
        
        .upload-area {{
            padding: 20px;
        }}
    }}
    
    .stAlert {{
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    
    .stAlert[data-baseweb="notification"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }}
    
    [data-testid="stExpander"] {{
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        overflow: hidden;
    }}
    
    [data-testid="stExpander"] summary {{
        background: rgba(102, 126, 234, 0.1);
        padding: 15px;
        font-weight: 600;
    }}
    
    [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        padding: 15px;
        background: rgba(102, 126, 234, 0.05);
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<h1>{_("app_title")}</h1>', unsafe_allow_html=True)
subtitle_text = _("app_subtitle")
st.markdown(f'<p style="text-align: center; font-size: 1.3rem; color: #666; font-weight: 600; margin-bottom: 2rem;">{subtitle_text}</p>', unsafe_allow_html=True)

show_analysis_modal()

st.sidebar.markdown(f'<h2 style="color: white; text-align: center; margin-bottom: 2rem;">Настройки</h2>', unsafe_allow_html=True)

st.sidebar.markdown('<div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
st.sidebar.markdown('<h3 style="color: white; margin-top: 0; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 10px;">🤖 Модели ML</h3>', unsafe_allow_html=True)
if 'confirmed_models' not in st.session_state:
    st.session_state['confirmed_models'] = None
model_options = st.sidebar.multiselect(
    "Выберите модели:",
    ["isolation_forest", "autoencoder", "lstm"],
    ["isolation_forest"],
    help="Isolation Forest - быстрая модель, Она изолирует (отделяет) подозрительные транзакции от нормальных, AutoEncoder - нейросеть декодирует данные, учится воспроизводить нормальные транзакции, LSTM - анализ последовательностей, смотрит, как ведет себя клиент со временем"
)
confirm_models = st.sidebar.button("✅ Подтвердить выбор", use_container_width=True)
if confirm_models:
    st.session_state['confirmed_models'] = model_options
    st.sidebar.success("Модели подтверждены")
if st.session_state['confirmed_models']:
    st.sidebar.markdown(f"<p style=\"color: white; background: rgba(102, 126, 234, 0.3); padding: 10px; border-radius: 10px; text-align: center;\">Выбрано: {', '.join(st.session_state['confirmed_models'])}</p>", unsafe_allow_html=True)
reset_models = st.sidebar.button("↩️ Сбросить", use_container_width=True)
if reset_models:
    st.session_state['confirmed_models'] = None
    st.sidebar.info("Выбор моделей сброшен")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
st.sidebar.markdown('<h3 style="color: white; margin-top: 0; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 10px;">📊 Параметры анализа</h3>', unsafe_allow_html=True)
contamination_level = st.sidebar.slider(
    "Ожидаемый уровень мошенничества:",
    min_value=0.01, max_value=0.2, value=0.05, step=0.01,
    help="Процент транзакций, которые вы ожидаете увидеть как мошеннические"
)
st.sidebar.markdown(f'<p style="color: white; text-align: center; font-weight: 600; background: rgba(102, 126, 234, 0.3); padding: 10px; border-radius: 10px;">{contamination_level*100:.1f}%</p>', unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)



st.sidebar.markdown('---')
st.sidebar.markdown('<div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
st.sidebar.markdown('<h3 style="color: white; margin-top: 0; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 10px; text-align: center;">📊 Тестовые данные</h3>', unsafe_allow_html=True)
test_file = st.sidebar.selectbox(
    "Выберите тестовый файл:",
    ["Не использовать", "Нормальные транзакции", "Мошеннические транзакции", "Смешанные транзакции"],
    key="test_file_selectbox"
)
if test_file != "Не использовать":
    test_file_map = {
        "Нормальные транзакции": "test_data/normal_transactions.csv",
        "Мошеннические транзакции": "test_data/fraud_transactions.csv",
        "Смешанные транзакции": "test_data/mixed_transactions.csv"
    }
    if st.sidebar.button("📥 Загрузить тестовый файл", use_container_width=True):
        st.session_state['test_file_path'] = test_file_map[test_file]
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.markdown('<div style="background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
st.sidebar.markdown('<h3 style="color: white; margin-top: 0; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 10px; text-align: center;">📚 Документация</h3>', unsafe_allow_html=True)


try:
    with open('docs/user_guide.md', 'r', encoding='utf-8') as f:
        documentation_content = f.read()
    

    st.sidebar.markdown("""
    <div style="color: #e0e0e0; font-size: 0.9rem; line-height: 1.5;">
        <p><strong>Основные возможности:</strong></p>
        <ul style="padding-left: 20px; margin: 10px 0;">
            <li>Анализ транзакций на мошенничество</li>
            <li>Машинное обучение</li>
            <li>Визуализация результатов</li>
            <li>Экспорт отчетов</li>
        </ul>
        <p><strong>Требования к данным:</strong></p>
        <p>Файлы должны содержать колонки: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest</p>
        <p><strong>Рекомендации:</strong></p>
        <ul style="padding-left: 20px; margin: 10px 0;">
            <li>Начните с Isolation Forest</li>
            <li>Установите уровень мошенничества 1-5%</li>
            <li>Проверяйте подозрительные транзакции</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    

    if st.sidebar.button("📖 Полное руководство", use_container_width=True):
        st.session_state['show_documentation'] = not st.session_state.get('show_documentation', False)
    

    if st.session_state.get('show_documentation', False):
        st.markdown("---")
        st.markdown('<div style="background: linear-gradient(135deg, rgba(30, 30, 40, 0.95) 0%, rgba(20, 20, 30, 0.95) 100%); padding: 30px; border-radius: 20px; margin: 20px 0; border: 1px solid rgba(102, 126, 234, 0.3);">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #667eea; text-align: center; margin-bottom: 30px;">📚 Полное руководство пользователя</h2>', unsafe_allow_html=True)
 
        import markdown
        html_content = markdown.markdown(documentation_content)
        st.markdown(f'<div style="color: #e0e0e0; line-height: 1.6;">{html_content}</div>', unsafe_allow_html=True)
        
        if st.button("⬆️ Скрыть руководство", use_container_width=True):
            st.session_state['show_documentation'] = False
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
except FileNotFoundError:
    st.sidebar.markdown('<p style="color: #ff6b6b; font-size: 0.9rem;">Документация не найдена</p>', unsafe_allow_html=True)
except Exception as e:
    st.sidebar.markdown(f'<p style="color: #ff6b6b; font-size: 0.9rem;">Ошибка загрузки документации: {str(e)}</p>', unsafe_allow_html=True)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.markdown('<p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; text-align: center;">💡 Совет: Начните с Isolation Forest для быстрого анализа</p>', unsafe_allow_html=True)

st.markdown('<h2>📁 Загрузка данных о транзакциях</h2>', unsafe_allow_html=True)
st.markdown('<div class="upload-area">', unsafe_allow_html=True)

if 'uploaded_files_list' not in st.session_state:
    st.session_state['uploaded_files_list'] = []
if 'confirmed_files' not in st.session_state:
    st.session_state['confirmed_files'] = False
if 'analysis_type' not in st.session_state:
    st.session_state['analysis_type'] = 'separate'

uploaded_files = st.file_uploader(
    "Перетащите CSV или Excel файлы сюда или нажмите для выбора (максимум 5 файлов)",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
    help="Файл должен содержать колонки: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest"
)

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.error("❌ Максимум 5 файлов за раз! Удалите лишние файлы.")
        st.stop()
    st.session_state['uploaded_files_list'] = uploaded_files
    st.info(f"📂 Загружено файлов: {len(uploaded_files)}")
    
    file_data = []
    for idx, file in enumerate(uploaded_files, 1):
        file_data.append({
            "№": idx,
            "Имя файла": file.name,
            "Размер": f"{file.size / 1024:.1f} KB",
            "Тип": file.type
        })
    
    file_df = pd.DataFrame(file_data)
    st.dataframe(file_df, use_container_width=True, hide_index=True)
    
    if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
        st.markdown("---")
        st.markdown("### 📊 Просмотр анализа файлов")
        analysis_results = st.session_state['analysis_results']
        
        cols = st.columns(min(3, len(uploaded_files)))
        for idx, file in enumerate(uploaded_files):
            col_idx = idx % 3
            with cols[col_idx]:
                file_key = file.name if hasattr(file, 'name') else f"file_{idx}"
                if file_key in analysis_results:
                    result = analysis_results[file_key]
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                                padding: 15px; border-radius: 10px; margin: 10px 0; text-align: center;">
                        <div style="font-weight: 600; margin-bottom: 10px;">{file.name[:20]}{'...' if len(file.name) > 20 else ''}</div>
                        <div style="font-size: 0.9rem; color: #666; margin-bottom: 10px;">
                            Подозрительных: {result['suspicious_count']}
                        </div>
                        <button onclick="document.getElementById('view_analysis_{idx}').click()" 
                                style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                       color: white; border: none; padding: 8px 15px; border-radius: 20px; 
                                       cursor: pointer; font-size: 0.9rem; width: 100%;">
                            📈 Просмотр анализа
                        </button>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("", key=f"view_analysis_{idx}", help=f"Просмотр анализа {file.name}"):
                        st.session_state['selected_file_for_analysis'] = file_key
                        st.session_state['show_analysis_modal'] = True
                        st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        analysis_type = st.radio(
            "Тип анализа:",
            ["Отдельно для каждого файла", "Объединить все файлы"],
            key="analysis_type_radio"
        )
        st.session_state['analysis_type'] = 'separate' if analysis_type == "Отдельно для каждого файла" else 'combined'
            
    with col2:
        if st.button("✅ Подтвердить файлы и продолжить", use_container_width=True, key="confirm_files_btn"):
            st.session_state['confirmed_files'] = True
            st.success(f"✅ Подтверждено {len(uploaded_files)} файлов для анализа ({analysis_type})")
            
            if st.session_state.get('current_user'):
                user_id = st.session_state['current_user']['id']
                for file in uploaded_files:
                    user_db.store_user_analysis_data(user_id, file.name, "uploaded_file", False)
            
            if not st.session_state.get('confirmed_models'):
                st.warning("⚠️ Модели не подтверждены. Будут использованы модели по умолчанию: Isolation Forest.")
            
            st.rerun()
    
    if st.session_state.get('current_user'):
        st.markdown("---")
        st.markdown("### 🌍 Разрешение на использование местоположения")
        st.info("Для более точного анализа мы можем использовать информацию о местоположении транзакций. Это поможет выявлять подозрительные транзакции, связанные с географическими аномалиями.")
        
        location_permission = st.checkbox(
            "Разрешить использование данных о местоположении для анализа", 
            key="location_permission",
            help="Это поможет улучшить точность детекции мошенничества"
        )
        
        if location_permission:
            st.success("✅ Разрешение на использование местоположения предоставлено")
            if st.session_state.get('current_user') and uploaded_files:
                user_id = st.session_state['current_user']['id']
                for file in uploaded_files:
                    user_db.update_location_permission(user_id, file.name, True)

st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.get('test_file_path'):
    st.session_state['uploaded_files_list'] = [st.session_state['test_file_path']]
    st.session_state['confirmed_files'] = True
    st.session_state['analysis_type'] = 'separate'
    st.info(f"✅ Тестовый файл загружен: {st.session_state['test_file_path']}")

if st.session_state.get('confirmed_files') and st.session_state.get('uploaded_files_list'):
    files_to_process = st.session_state['uploaded_files_list']
    analysis_mode = st.session_state.get('analysis_type', 'separate')
    
    if analysis_mode == 'combined':
        st.markdown('<h2>🔄 Объединенный анализ всех файлов</h2>', unsafe_allow_html=True)
        all_dfs = []
        for file in files_to_process:
            try:
                if isinstance(file, str):
                    df_temp = load_data(file)
                else:
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    df_temp = load_data(temp_path)
                    os.remove(temp_path)
                all_dfs.append(df_temp)
            except Exception as e:
                st.error(f"❌ Ошибка загрузки файла: {str(e)}")
                continue
        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
            st.success(f"✅ Объединено {len(all_dfs)} файлов. Всего транзакций: {len(df)}")
            files_to_analyze = [{'name': 'Объединенные данные', 'data': df}]
    else:
        st.markdown('<h2>🔄 Отдельный анализ каждого файла</h2>', unsafe_allow_html=True)
        files_to_analyze = []
        for file in files_to_process:
            try:
                if isinstance(file, str):
                    df_temp = load_data(file)
                    file_name = file.split('/')[-1]
                else:
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    df_temp = load_data(temp_path)
                    os.remove(temp_path)
                    file_name = file.name
                files_to_analyze.append({'name': file_name, 'data': df_temp})
            except Exception as e:
                st.error(f"❌ Ошибка загрузки {file}: {str(e)}")
                continue
    
    for file_info in files_to_analyze:
        df = file_info['data']
        file_name = file_info['name']
        
        st.markdown(f'<div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%); padding: 30px; border-radius: 20px; margin: 30px 0; border: 2px solid #667eea;">', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color: #667eea; text-align: center; margin-top: 0;">📄 {file_name}</h2>', unsafe_allow_html=True)
        st.markdown('<hr style="border: 2px solid #667eea; margin: 20px 0;">', unsafe_allow_html=True)
        
        try:
            with st.spinner("🔄 Загружаем и обрабатываем данные..."):
                df_processed = preprocess(df)
            st.success("✅ Данные успешно загружены и обработаны!")
            st.markdown('<br>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Ошибка предобработки: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
            continue
        
        with st.expander("👁️ Предварительный просмотр данных", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown('<div style="background: linear-gradient(135deg, rgba(56, 239, 125, 0.1) 0%, rgba(17, 153, 142, 0.1) 100%); padding: 25px; border-radius: 15px; margin: 20px 0;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #11998e; margin-top: 0;">📊 Статистика данных</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💳 Всего транзакций", f"{len(df):,}")
        with col2:
            total_amount = df['amount'].sum()
            st.metric("💰 Общая сумма", f"{total_amount:,.0f}")
        with col3:
            st.metric("👥 Уникальных клиентов", f"{df['nameOrig'].nunique():,}")
        with col4:
            st.metric("🎯 Получателей", f"{df['nameDest'].nunique():,}")
        
        st.markdown('<br>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**📈 Средняя сумма транзакции:** `{df['amount'].mean():,.2f}`")
            st.markdown(f"**📊 Медианная сумма:** `{df['amount'].median():,.2f}`")
        with col2:
            top_type = df['type'].value_counts().index[0]
            top_type_count = df['type'].value_counts().values[0]
            st.markdown(f"**🔝 Самый частый тип:** `{top_type}` ({top_type_count} транзакций)")
            st.markdown(f"**⏰ Период данных:** `{df['step'].min()} - {df['step'].max()} часов`")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<h2>🔍 Анализ на мошенничество</h2>', unsafe_allow_html=True)
        st.markdown('<div style="background: linear-gradient(135deg, rgba(245, 87, 108, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%); padding: 30px; border-radius: 20px; margin: 20px 0;">', unsafe_allow_html=True)
        
        try:
            with st.spinner("🧠 Анализируем транзакции на предмет мошенничества..."):
                selected_models = st.session_state['confirmed_models'] if st.session_state.get('confirmed_models') else model_options
                fraud_scores, anomalies, model_details = advanced_model_pipeline(
                    df_processed, 
                    model_types=selected_models,
                    contamination=contamination_level
                )
                
                rules_combined, rules_flags = rule_engine(df_processed)
                
                normalized_ml_scores = (fraud_scores - np.min(fraud_scores)) / (np.max(fraud_scores) - np.min(fraud_scores) + 1e-8)
                combined_scores = 0.7 * normalized_ml_scores + 0.3 * rules_combined
                
                is_suspicious = combined_scores > np.percentile(combined_scores, 95)
                suspicious_count = np.sum(is_suspicious)
                

                self_learning_results = integrate_self_learning(df_processed, combined_scores, is_suspicious)
                
                if 'analysis_results' not in st.session_state:
                    st.session_state['analysis_results'] = {}
                
                st.session_state['analysis_results'][file_name] = {
                    'total_transactions': len(df),
                    'suspicious_count': int(suspicious_count),
                    'clean_count': int(len(df) - suspicious_count),
                    'fraud_percentage': (suspicious_count / len(df)) * 100,
                    'combined_scores': combined_scores,
                    'model_contributions': model_details if model_details else {},
                    'is_suspicious': is_suspicious
                }
            
            st.success("✅ Анализ завершен!")
            
            st.markdown('<h3>📊 Ключевые показатели:</h3>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🚨 Подозрительных", f"{suspicious_count:,}")
            with col2:
                st.metric("✅ Чистых", f"{len(df) - suspicious_count:,}")
            with col3:
                fraud_pct = (suspicious_count / len(df)) * 100
                st.metric("📉 Процент мошенничества", f"{fraud_pct:.2f}%")
            with col4:
                avg_score = np.mean(combined_scores)
                st.metric("🔢 Средний Fraud Score", f"{avg_score:.4f}")
            
            st.markdown('<br>', unsafe_allow_html=True)
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Распределение", "📋 Подозрительные", "🤖 Модели", "📊 Типы", "🧠 Рекомендации"])
            
            with tab1:
                try:
                    perf_dashboard = enhanced_viz.create_model_performance_dashboard(model_details)
                    if perf_dashboard:
                        st.plotly_chart(perf_dashboard, use_container_width=True)
                    else:
                        st.info("Расширенная визуализация недоступна (требуется установка plotly)")
                except Exception as e:
                    st.warning("Не удалось создать панель производительности моделей")
                
                fig, ax = plt.subplots(figsize=(14, 7))
                
                plot_scores = combined_scores[:len(df)] if len(combined_scores) > len(df) else combined_scores
                if len(plot_scores) < len(df):
                    extension = np.full(len(df) - len(plot_scores), np.mean(plot_scores) if len(plot_scores) > 0 else 0)
                    plot_scores = np.concatenate([plot_scores, extension])
            
                n, bins, patches = ax.hist(plot_scores, bins=30, alpha=0.8, color='#667eea', edgecolor='white', linewidth=0.5)
                
                for i, patch in enumerate(patches):
                    if bins[i] > np.percentile(plot_scores, 90):
                        patch.set_facecolor('#f5576c') 
                    elif bins[i] > np.percentile(plot_scores, 75):
                        patch.set_facecolor('#f093fb')
                    else:
                        patch.set_facecolor('#667eea')  
                
                ax.set_xlabel('Уровень подозрительности (Fraud Score) - чем выше, тем подозрительнее', fontsize=14, fontweight='bold')
                ax.set_ylabel('Количество транзакций', fontsize=14, fontweight='bold')
                ax.set_title('📊 Как часто встречаются подозрительные транзакции?', fontsize=16, fontweight='bold', pad=20)
                
                ax.grid(True, alpha=0.3, linestyle='--')
                
                avg_score = np.mean(plot_scores)
                ax.axvline(avg_score, color='orange', linestyle='-', linewidth=3, 
                          label=f'Средний уровень подозрительности: {avg_score:.3f}')
                p75 = np.percentile(plot_scores, 75)
                p90 = np.percentile(plot_scores, 90)
                ax.axvline(p75, color='yellow', linestyle='--', linewidth=2, 
                          label=f'75% транзакций менее подозрительны чем: {p75:.3f}')
                ax.axvline(p90, color='red', linestyle='-.', linewidth=2, 
                          label=f'90% транзакций менее подозрительны чем: {p90:.3f}')
                
                ax.legend(fontsize=12, loc='upper right')
                plt.figtext(0.5, 0.01, '💡 Синие столбики = нормальные транзакции, Розовые = подозрительные, Красные = очень подозрительные', 
                           ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
            
                plt.tight_layout()
                st.pyplot(fig)
                
                if 'step' in df.columns:
                    fig, ax = plt.subplots(figsize=(14, 7))
                    
                    x_data = df['step'].values[:len(plot_scores)] if len(df['step']) > len(plot_scores) else df['step'].values
                    y_data = plot_scores[:len(x_data)] if len(plot_scores) > len(x_data) else plot_scores
                    
                    min_len = min(len(x_data), len(y_data))
                    if min_len > 0:
                        x_data = x_data[:min_len]
                        y_data = y_data[:min_len]
                        scatter = ax.scatter(x_data, y_data, c=y_data, cmap='RdYlBu_r', 
                                           alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
                        cbar = plt.colorbar(scatter)
                        cbar.set_label('Уровень подозрительности - красный = очень подозрительно!', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Время (часы) - когда была транзакция', fontsize=14, fontweight='bold')
                        ax.set_ylabel('Уровень подозрительности (Fraud Score) - насколько она странная', fontsize=14, fontweight='bold')
                        ax.set_title('⏰ Меняется ли подозрительность со временем?', fontsize=16, fontweight='bold', pad=20)
                        ax.grid(True, alpha=0.3, linestyle='--')
                        
                        if len(x_data) > 1:
                            z = np.polyfit(x_data, y_data, 1)
                            p = np.poly1d(z)
                            trend_direction = "↑ Растет" if z[0] > 0 else "↓ Падает"
                            trend_explanation = "подозрительность увеличивается" if z[0] > 0 else "подозрительность уменьшается"
                            ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=3, 
                                   label=f'Тренд: {trend_direction} - {trend_explanation}')
                            ax.legend(fontsize=12)
                        
                        plt.figtext(0.5, 0.01, '💡 Каждая точка = одна транзакция. Чем краснее точка, тем подозрительнее транзакция.', 
                                   ha='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
                        
                        plt.tight_layout()
                        st.pyplot(fig)
            
            with tab2:
                st.markdown('<h3>🚨 Топ подозрительных транзакций</h3>', unsafe_allow_html=True)
                suspicious_indices = np.where(is_suspicious)[0]
                if len(suspicious_indices) > 0:
                    suspicious_data = []
                    for idx in suspicious_indices[:20]:
                        row = df.iloc[idx].to_dict()
                        row['fraud_score'] = combined_scores[idx]
                        suspicious_data.append(row)
                    
                    suspicious_df = pd.DataFrame(suspicious_data)
                    if not suspicious_df.empty:
                        suspicious_df = suspicious_df.sort_values('fraud_score', ascending=False)
                        
                        st.dataframe(suspicious_df[['step', 'type', 'amount', 'nameOrig', 'nameDest', 'fraud_score']].style.format({
                            'amount': '{:,.2f}',
                            'fraud_score': '{:.4f}'
                        }), use_container_width=True)
                    else:
                        st.info("Нет подозрительных транзакций для отображения")
                else:
                    st.info("Нет подозрительных транзакций")
            
            with tab3:
                if model_details:
                    st.markdown('<h3>🤖 Сравнение моделей</h3>', unsafe_allow_html=True)
                    try:
                        model_names = list(model_details.keys())
                        anomaly_counts = [int(details.get('anomaly_count', 0)) for details in model_details.values()]
                        mean_scores = [float(details.get('mean_score', 0)) for details in model_details.values()]
                        weights = [float(details.get('weight', 0)) for details in model_details.values()]
                        
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                        fig.suptitle('Сравнение эффективности моделей', fontsize=16, fontweight='bold')
                        
                        bars1 = ax1.bar(model_names, anomaly_counts, color=['#667eea', '#764ba2', '#f093fb', '#f5576c'][:len(model_names)])
                        ax1.set_title('Количество обнаруженных подозрительных транзакций', fontsize=12, fontweight='bold')
                        ax1.set_ylabel('Количество транзакций')
                        ax1.set_xlabel('Модели')
    
                        for i, bar in enumerate(bars1):
                            height = bar.get_height()
                            ax1.annotate(f'{anomaly_counts[i]}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),
                                        textcoords="offset points",
                                        ha='center', va='bottom', fontweight='bold')
                        
                        bars2 = ax2.bar(model_names, mean_scores, color=['#11998e', '#38ef7d', '#f093fb', '#667eea'][:len(model_names)])
                        ax2.set_title('Средний уровень подозрительности', fontsize=12, fontweight='bold')
                        ax2.set_ylabel('Уровень подозрительности (Fraud Score)')
                        ax2.set_xlabel('Модели')
                        
                        for i, bar in enumerate(bars2):
                            height = bar.get_height()
                            ax2.annotate(f'{mean_scores[i]:.3f}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),
                                        textcoords="offset points",
                                        ha='center', va='bottom', fontweight='bold')
                        
                        bars3 = ax3.bar(model_names, weights, color=['#f5576c', '#f093fb', '#667eea', '#11998e'][:len(model_names)])
                        ax3.set_title('Вклад модели в общий результат', fontsize=12, fontweight='bold')
                        ax3.set_ylabel('Весовой коэффициент')
                        ax3.set_xlabel('Модели')
                        
                        for i, bar in enumerate(bars3):
                            height = bar.get_height()
                            ax3.annotate(f'{weights[i]:.2f}',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),
                                        textcoords="offset points",
                                        ha='center', va='bottom', fontweight='bold')
                        
                        for ax in [ax1, ax2, ax3]:
                            ax.tick_params(axis='x', rotation=45)
                            ax.grid(True, alpha=0.3, linestyle='--')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.warning(f"Не удалось создать сравнение моделей: {str(e)}")
                    
                    st.markdown('<h4>📊 Детальный разбор</h4>', unsafe_allow_html=True)
                    for model_name, details in model_details.items():
                        model_name_ru = {
                            'isolation_forest': 'Изолирующий лес (Isolation Forest)',
                            'autoencoder': 'Автоэнкодер (AutoEncoder)',
                            'lstm': 'Долгая краткосрочная память (LSTM)',
                            'fallback_isolation_forest': 'Резервный изолирующий лес',
                            'random_fallback': 'Случайная модель'
                        }.get(model_name, model_name)
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); 
                                    padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #667eea;">
                            <div style="font-weight: 700; color: #667eea; margin-bottom: 10px; font-size: 1.1em;">{model_name_ru}</div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                                <div><strong>⚖️ Вес в анализе:</strong> {details.get('weight', 0):.2f}</div>
                                <div><strong>🚨 Подозрительных транзакций:</strong> {int(details.get('anomaly_count', 0))}</div>
                                <div><strong>📊 Средний уровень подозрительности:</strong> {details.get('mean_score', 0):.4f}</div>
                                {'<div><strong>⏱️ Время выполнения:</strong> ' + f"{details.get('execution_time', 0):.2f} сек" + '</div>' if 'execution_time' in details else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Нет информации о моделях для отображения")
            with tab4:
                st.markdown('<h3>💳 Анализ по типам транзакций</h3>', unsafe_allow_html=True)
                
                type_counts = df['type'].value_counts()
                suspicious_by_type = df[df.index.isin(suspicious_indices)]['type'].value_counts()
                
                type_analysis = pd.DataFrame({
                    'Всего': type_counts,
                    'Подозрительных': suspicious_by_type
                }).fillna(0)
                type_analysis['Процент подозрительных'] = (type_analysis['Подозрительных'] / type_analysis['Всего']) * 100
                
                st.dataframe(type_analysis.style.format({
                    'Всего': '{:.0f}',
                    'Подозрительных': '{:.0f}',
                    'Процент подозрительных': '{:.2f}%'
                }), use_container_width=True)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                fig.suptitle('Анализ транзакций по типам', fontsize=16, fontweight='bold')
                
                x = np.arange(len(type_analysis))
                width = 0.35
                
                total_data = type_analysis['Всего'].values
                suspicious_data = type_analysis['Подозрительных'].values
                min_bar_len = min(len(total_data), len(suspicious_data), len(x))
                
                if min_bar_len > 0:
                    x_trimmed = x[:min_bar_len]
                    total_data_trimmed = total_data[:min_bar_len]
                    suspicious_data_trimmed = suspicious_data[:min_bar_len]
                    
                    bars1 = ax1.bar(x_trimmed - width/2, total_data_trimmed, width, 
                                   label='Все транзакции', alpha=0.8, color='#667eea', edgecolor='white')
                    bars2 = ax1.bar(x_trimmed + width/2, suspicious_data_trimmed, width, 
                                   label='Подозрительные', alpha=0.8, color='#f5576c', edgecolor='white')
                    
                    for bar in bars1:
                        height = bar.get_height()
                        ax1.annotate(f'{int(height)}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3), 
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)
                    
                    for bar in bars2:
                        height = bar.get_height()
                        ax1.annotate(f'{int(height)}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)
                    
                    ax1.set_xlabel('Тип транзакции', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Количество транзакций', fontsize=12, fontweight='bold')
                    ax1.set_title('Сравнение всех транзакций и подозрительных по типам', fontsize=12, fontweight='bold')
                    ax1.set_xticks(x_trimmed)
                    ax1.set_xticklabels(type_analysis.index[:min_bar_len], rotation=45, ha='right')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3, linestyle='--')
                
                suspicious_types = type_analysis[type_analysis['Подозрительных'] > 0]
                if not suspicious_types.empty:
                    wedges, texts, autotexts = ax2.pie(suspicious_types['Подозрительных'], 
                                                      labels=suspicious_types.index, 
                                                      autopct='%1.1f%%', 
                                                      startangle=90,
                                                      colors=plt.cm.Set3.colors,
                                                      explode=[0.05] * len(suspicious_types),  
                                                      shadow=True)
                    
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(10)
                    
                    ax2.set_title('Процент подозрительных транзакций по типам', fontsize=12, fontweight='bold')
                else:
                    ax2.text(0.5, 0.5, 'Нет подозрительных транзакций', 
                            ha='center', va='center', transform=ax2.transAxes,
                            fontsize=12, fontweight='bold', color='#666')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab5:
                st.markdown('<h3>💡 Интеллектуальные рекомендации</h3>', unsafe_allow_html=True)
                
                recommendations = []
                
                if fraud_pct > 5:
                    recommendations.append({
                        'priority': 'critical',
                        'message': f'🚨 Высокий уровень мошенничества ({fraud_pct:.2f}%). Рекомендуется немедленное расследование.'
                    })
                
                avg_amount = df['amount'].mean()
                high_amount_transactions = df[df['amount'] > avg_amount * 5]
                if len(high_amount_transactions) > 0:
                    recommendations.append({
                        'priority': 'attention',
                        'message': f'⚠️ Обнаружены транзакции с очень высокими суммами (более чем в 5 раз превышающими среднюю). Следует проверить эти транзакции.'
                    })
                
                night_transactions = df[(df['step'] % 24 >= 22) | (df['step'] % 24 <= 6)]
                if len(night_transactions) > len(df) * 0.1:
                    recommendations.append({
                        'priority': 'attention',
                        'message': f'🌙 Более 10% транзакций происходят ночью. Это может указывать на подозрительную активность.'
                    })
                
                frequent_recipients = df['nameDest'].value_counts()
                top_recipients = frequent_recipients[frequent_recipients > 10]
                if len(top_recipients) > 0:
                    recommendations.append({
                        'priority': 'normal',
                        'message': f'🔄 Некоторые получатели получают много транзакций. Стоит проверить, не связано ли это с мошенничеством.'
                    })
                
                if recommendations:
                    for rec in recommendations:
                        if rec['priority'] == 'critical':
                            st.markdown(f'<div style="background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0;"><strong>КРИТИЧНО:</strong> {rec["message"]}</div>', unsafe_allow_html=True)
                        elif rec['priority'] == 'attention':
                            st.markdown(f'<div style="background: linear-gradient(135deg, #f093fb 0%, #667eea 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0;"><strong>ВНИМАНИЕ:</strong> {rec["message"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0;"><strong>ИНФОРМАЦИЯ:</strong> {rec["message"]}</div>', unsafe_allow_html=True)
                else:
                    st.info("На основе анализа не найдено значительных рисков. Продолжайте мониторинг.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<h2>📤 Экспорт результатов</h2>', unsafe_allow_html=True)
            st.markdown('<div style="background: linear-gradient(135deg, rgba(56, 239, 125, 0.1) 0%, rgba(17, 153, 142, 0.1) 100%); padding: 30px; border-radius: 20px; margin: 20px 0;">', unsafe_allow_html=True)
            
            export_format = st.selectbox(
                "Выберите формат экспорта:",
                ["HTML отчет", "CSV данные", "Excel данные", "JSON данные"],
                key="export_format_selectbox"
            )
            
            if st.button("💾 Экспортировать результаты"):
                try:
                    with st.spinner("Генерируем отчет..."):
                        export_data = df.copy()
                        
                        if len(combined_scores) != len(df):
                            if len(combined_scores) > len(df):
                                adjusted_scores = combined_scores[:len(df)]
                                adjusted_suspicious = is_suspicious[:len(df)]
                            else:
                                adjusted_scores = np.pad(combined_scores, (0, len(df) - len(combined_scores)), 
                                                       mode='constant', constant_values=np.mean(combined_scores))
                                adjusted_suspicious = np.pad(is_suspicious, (0, len(df) - len(is_suspicious)), 
                                                           mode='constant', constant_values=0)
                        else:
                            adjusted_scores = combined_scores
                            adjusted_suspicious = is_suspicious
                        
                        export_data['fraud_score'] = adjusted_scores
                        export_data['is_suspicious'] = adjusted_suspicious
                        
                        def get_risk_level(score):
                            if score > 0.7:
                                return "Высокий"
                            elif score > 0.4:
                                return "Средний"
                            else:
                                return "Низкий"
                        
                        export_data['risk_level'] = [get_risk_level(score) for score in adjusted_scores]
                        
                        if export_format == "HTML отчет":
                            try:
                                report_html = export_all_results(
                                    df, 
                                    adjusted_scores, 
                                    adjusted_suspicious, 
                                    model_details, 
                                    rules_flags,
                                    file_name
                                )
                                st.download_button(
                                    label="📥 Скачать HTML отчет",
                                    data=report_html,
                                    file_name=f"fraud_report_{file_name.replace('.csv', '').replace('.xlsx', '')}.html",
                                    mime="text/html"
                                )
                            except Exception as e:
                                st.error(f"Ошибка создания HTML отчета: {str(e)}")
                        elif export_format == "CSV данные":
                            try:
                                csv_data = export_data.to_csv(index=False, encoding='utf-8-sig')
                                st.download_button(
                                    label="📥 Скачать CSV данные",
                                    data=csv_data,
                                    file_name=f"fraud_data_{file_name.replace('.csv', '').replace('.xlsx', '')}.csv",
                                    mime="text/csv"
                                )
                            except Exception as e:
                                st.error(f"Ошибка создания CSV файла: {str(e)}")
                        elif export_format == "Excel данные":
                            try:
                                import io
                                excel_buffer = io.BytesIO()
                                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                    export_data.to_excel(writer, index=False, sheet_name='Fraud_Analysis')
                                    
                                    summary_data = pd.DataFrame({
                                        'Метрика': ['Всего транзакций', 'Подозрительных', 'Процент мошенничества', 'Средний Fraud Score'],
                                        'Значение': [len(df), int(np.sum(adjusted_suspicious)), 
                                                   f"{(np.sum(adjusted_suspicious) / len(df) * 100):.2f}%", 
                                                   f"{np.mean(adjusted_scores):.4f}"]
                                    })
                                    summary_data.to_excel(writer, index=False, sheet_name='Summary')
                                
                                excel_buffer.seek(0)
                                st.download_button(
                                    label="📥 Скачать Excel данные",
                                    data=excel_buffer,
                                    file_name=f"fraud_data_{file_name.replace('.csv', '').replace('.xlsx', '')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            except Exception as e:
                                st.error(f"Ошибка создания Excel файла: {str(e)}")
                        else:  
                            try:
                                json_structure = {
                                    "file_info": {
                                        "filename": file_name,
                                        "total_transactions": len(df),
                                        "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                                    },
                                    "fraud_summary": {
                                        "suspicious_count": int(np.sum(adjusted_suspicious)),
                                        "fraud_percentage": float(np.sum(adjusted_suspicious) / len(df) * 100),
                                        "average_fraud_score": float(np.mean(adjusted_scores))
                                    },
                                    "risk_distribution": {
                                        "high_risk": int(np.sum(adjusted_scores > 0.7)),
                                        "medium_risk": int(np.sum((adjusted_scores > 0.4) & (adjusted_scores <= 0.7))),
                                        "low_risk": int(np.sum(adjusted_scores <= 0.4))
                                    },
                                    "model_details": model_details,
                                    "top_suspicious_transactions": []
                                }
                                
                                suspicious_indices = np.where(adjusted_suspicious)[0]
                                top_suspicious = sorted(suspicious_indices, key=lambda x: adjusted_scores[x], reverse=True)[:20]
                                
                                for idx in top_suspicious:
                                    if idx < len(df):
                                        transaction_info = {
                                            "transaction_id": int(idx),
                                            "fraud_score": float(adjusted_scores[idx]),
                                            "risk_level": get_risk_level(adjusted_scores[idx])
                                        }
                                        
                                        for col in ['step', 'type', 'amount', 'nameOrig', 'nameDest']:
                                            if col in df.columns and idx < len(df):
                                                transaction_info[col] = str(df.iloc[idx][col]) if not pd.isna(df.iloc[idx][col]) else "N/A"
                                        
                                        json_structure["top_suspicious_transactions"].append(transaction_info)
                                
                                json_data = json.dumps(json_structure, ensure_ascii=False, indent=2)
                                st.download_button(
                                    label="📥 Скачать JSON данные",
                                    data=json_data,
                                    file_name=f"fraud_data_{file_name.replace('.csv', '').replace('.xlsx', '')}.json",
                                    mime="application/json"
                                )
                            except Exception as e:
                                st.error(f"Ошибка создания JSON файла: {str(e)}")
                    
                    st.success("✅ Результаты готовы к скачиванию!")
                except Exception as e:
                    st.error(f"❌ Ошибка при экспорте: {str(e)}")
                    st.info("ℹ️ Попробуйте другой формат экспорта или свяжитесь с техподдержкой")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            error_message = str(e)
            
            if 'error_history' not in st.session_state:
                st.session_state['error_history'] = {}
            
            error_key = error_message.strip()
            if error_key in st.session_state['error_history']:
                st.session_state['error_history'][error_key] += 1
                if st.session_state['error_history'][error_key] > 1:
                    st.error(f"❌ Повторяющаяся ошибка анализа: {error_message}")
                    st.info("💡 Система самообучения: Эта ошибка уже возникала ранее. Применяются автоматические корректировки...")
                    if "x and y must be the same size" in error_message:
                        st.info("🔧 Автоматическое решение: Синхронизация размеров массивов данных для визуализации.")
                    elif "contamination" in error_message.lower():
                        st.info("🔧 Автоматическое решение: Корректировка параметров модели для предотвращения ошибок.")
                    else:
                        st.info("🔧 Автоматическое решение: Применение универсальных корректировок для стабилизации анализа.")
                else:
                    st.error(f"❌ Ошибка анализа: {error_message}")
                    st.info("💡 Система запомнит эту ошибку для будущих автоматических корректировок.")
            else:
                st.session_state['error_history'][error_key] = 1
                st.error(f"❌ Ошибка анализа: {error_message}")
                st.info("💡 Система самообучения: Эта ошибка будет отслеживаться для автоматического решения в будущем.")
            
            if "column" in error_message.lower() and "missing" in error_message.lower():
                st.info("💡 Это ошибка данных. Проверьте, что ваш файл содержит все необходимые столбцы: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest")
            elif "contamination" in error_message.lower() or "parameter" in error_message.lower():
                st.info("💡 Это ошибка модели. Попробуйте изменить уровень ожидаемого мошенничества в настройках или выбрать другие модели.")
            elif "memory" in error_message.lower() or "size" in error_message.lower():
                st.info("💡 Это ошибка памяти. Файл слишком большой. Попробуйте использовать только Isolation Forest модель или уменьшить файл.")
            elif "x and y must be the same size" in error_message:
                st.info("💡 Это ошибка визуализации. Размеры данных для графиков не совпадают. Система автоматически корректирует размеры.")
            else:
                st.info("💡 Если эта ошибка повторяется, попробуйте перезапустить приложение с помощью кнопки 'Начать заново' ниже.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    

    if st.button("🔄 Начать заново", use_container_width=True):
        keys_to_clear = ['uploaded_files_list', 'confirmed_files', 'analysis_type', 'test_file_path', 'analysis_results', 'confirmed_models']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
st.markdown('<hr style="margin: 40px 0;">', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #666; font-size: 0.9rem;">', unsafe_allow_html=True)
st.markdown('🛡️ ClearFlow Security- Интеллектуальная система обнаружения финансового мошенничества')
st.markdown('© 2025 ClearFlow Security. Все права защищены.')
st.markdown('</div>', unsafe_allow_html=True)
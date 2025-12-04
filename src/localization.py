import json
import os

class LocalizationManager:
    
    def __init__(self, default_language="ru"):
        self.default_language = default_language
        self.translations = self._load_translations()
    
    def _load_translations(self):
        translations = {
            "ru": {
                "app_title": "🛡️ ClearFlow Security",
                "app_subtitle": "Интеллектуальная система обнаружения финансового мошенничества",
                "settings": "⚙️ Настройки",
                "theme": "🎨 Тема",
                "language": "🌍 Язык",
                "dark_theme": "🌑 Темная",
                "light_theme": "☀️ Светлая",
                "russian": "🇷🇺 Русский",
                "english": "🇬🇧 English",
                
                "upload_title": "📁 Загрузка данных о транзакциях",
                "upload_instruction": "Перетащите CSV или Excel файлы сюда или нажмите для выбора (максимум 5 файлов)",
                "file_help": "Файл должен содержать колонки: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest",
                
                "analysis_title": "🔄 Анализ файлов",
                "separate_analysis": "Отдельно для каждого файла",
                "combined_analysis": "Объединить все файлы",
                "confirm_files": "✅ Подтвердить файлы и продолжить",
                
                "test_data": "📊 Тестовые данные",
                "test_file_select": "Выберите тестовый файл:",
                "no_test_file": "Не использовать",
                "normal_transactions": "Нормальные транзакции",
                "fraud_transactions": "Мошеннические транзакции",
                "mixed_transactions": "Смешанные транзакции",
                "load_test_file": "📥 Загрузить тестовый файл",
                
                "model_selection": "🤖 Выберите ML-модели:",
                "model_help": "Isolation Forest - быстрая модель, AutoEncoder - нейросеть декодирует данные, LSTM - анализ последовательностей",
                "confirm_models": "✅ Подтвердить выбор моделей",
                "models_confirmed": "Модели подтверждены",
                "reset_confirmation": "↩️ Сбросить подтверждение",
                "models_reset": "Выбор моделей сброшен",
                "selected_models": "Выбрано: ",
                
                "contamination_level": "📊 Ожидаемый уровень мошенничества:",
                "contamination_help": "Процент транзакций, которые вы ожидаете увидеть как мошеннические",
                
                "advice": "💡 Совет: Начните с Isolation Forest для быстрого анализа",
                
                "overview_tab": "📊 Обзор",
                "score_dist_tab": "📈 Распределение скоров",
                "time_analysis_tab": "⏰ Временной анализ",
                "feature_importance_tab": "🎯 Важность признаков",
                "model_eval_tab": "🔬 Оценка модели",
                "heatmap_tab": "🔥 Тепловая карта",
                "amount_dist_tab": "💰 Распределение сумм",
                
                "top_risky_clients_tab": "👥 Топ рисковых клиентов",
                "timeline_tab": "⏰ Временная линия",
                "type_stats_tab": "📊 Статистика по типам",
                "recommendations_tab": "🎯 Рекомендации",
                
                "export_title": "💾 Экспорт результатов",
                "export_all": "📦 Экспортировать все результаты",
                "download_suspicious": "📥 Скачать подозрительные транзакции (CSV)",
                
                "instructions_title": "📖 Инструкция по использованию FraudDetect 2.0",
                "how_to_use": "🚀 Как использовать систему:",
                "usage_step1": "1. **📁 Загрузите CSV файл** с данными о финансовых транзакциях",
                "usage_step2": "2. **⚙️ Настройте параметры** обнаружения в боковой панели",
                "usage_step3": "3. **⏳ Дождитесь** завершения анализа",
                "usage_step4": "4. **📊 Изучите результаты** в интерактивной панели",
                "usage_step5": "5. **💾 Экспортируйте** результаты при необходимости",
                
                "csv_format": "📋 Формат CSV файла:",
                "csv_step": "`step` | Временной шаг транзакции",
                "csv_type": "`type` | Тип транзакции (PAYMENT, TRANSFER и т.д.)",
                "csv_amount": "`amount` | Сумма транзакции",
                "csv_nameOrig": "`nameOrig` | ID клиента-отправителя",
                "csv_oldbalanceOrg": "`oldbalanceOrg` | Баланс до транзакции (отправитель)",
                "csv_newbalanceOrig": "`newbalanceOrig` | Баланс после транзакции (отправитель)",
                "csv_nameDest": "`nameDest` | ID получателя",
                "csv_oldbalanceDest": "`oldbalanceDest` | Баланс до транзакции (получатель)",
                "csv_newbalanceDest": "`newbalanceDest` | Баланс после транзакции (получатель)",
                "csv_isFraud": "`isFraud` (опционально) | Метки для оценки (1 - мошенничество, 0 - норма)",
                
                "models_title": "🤖 Используемые модели:",
                "isolation_forest_desc": "**Isolation Forest** 🌲 - Обнаруживает аномалии на основе принципа изоляции",
                "autoencoder_desc": "**AutoEncoder** 🧠 - Нейросеть, которая выявляет ошибки реконструкции",
                "lstm_desc": "**LSTM AutoEncoder** ⏰ - Анализ последовательностей для временных паттернов",
                
                "interpretation_title": "🎯 Интерпретация результатов:",
                "fraud_score_desc": "- **Fraud Score** (0-1): Чем выше значение, тем более подозрительна транзакция",
                "explanations_desc": "- **Объяснения**: Показывают конкретные причины, почему транзакция помечена",
                "combined_approach_desc": "- **Комбинированный подход**: 70% ML-модели + 30% правила = максимальная точность",
                
                "fraud_rules_title": "💡 Правила обнаружения мошенничества:",
                "rule_large_amounts": "⚠️ **Необычно большие суммы** - относительно истории клиента",
                "rule_new_accounts": "🆕 **Транзакции на новые счета** - первый перевод получателю",
                "rule_account_drain": "💸 **Опустошение баланса** - снятие >90% средств",
                "rule_unusual_time": "🌙 **Необычное время** - операции с 1 до 5 утра",
                "rule_activity_spike": "📈 **Всплеск активности** - резкое увеличение частоты транзакций",
                "rule_round_amounts": "💰 **Круглые суммы** - 1000, 5000, 10000 (часто связано с мошенничеством)",
                
                "buttons_and_actions": ".Buttons and Actions",
                "start_analysis": "🚀 Запустить анализ",
                "confirm_choice": "✅ Подтвердить выбор",
                "reset": "↩️ Сбросить",
                "export": "💾 Экспорт",
                "download": "📥 Скачать",
                
                "status_messages": "Status Messages",
                "files_uploaded": "📂 Загружено файлов: ",
                "analysis_complete": "🎉 Анализ завершен! Результаты готовы",
                "data_loaded": "✅ Данные успешно загружены и обработаны!",
                "processing_data": "🔄 Загружаем и обрабатываем данные...",
                "analyzing_transactions": "🔍 Анализируем транзакции с помощью ИИ...",
                "exporting_results": "🔄 Экспортируем результаты...",
                "results_exported": "✅ Результаты успешно экспортированы!",
                
                "error_messages": "Error Messages",
                "error_processing": "❌ Ошибка предобработки: ",
                "error_analysis": "❌ Ошибка при анализе данных: ",
                "error_export": "❌ Ошибка при экспорте: ",
                "error_data_types": "❌ Ошибка типов данных: ",
                "max_files_error": "❌ Максимум 5 файлов за раз! Удалите лишние файлы.",
                
                "tips": "Tips",
                "tip_try_isolation_forest": "💡 Попробуйте:\n\n• Использовать только Isolation Forest\n• Проверьте формат данных\n• Используйте один из тестовых файлов",
                "tip_reduce_file_size": "💡 Попробуйте:\n\n• Использовать только Isolation Forest\n• Уменьшите размер файла\n• Перезапустите приложение",
                
                "risk_levels": "Risk Levels",
                "critical": "🔴 Критическое",
                "attention": "🟡 Внимание",
                "normal": "🟢 Нормально",
                "high_risk": "⚠️ Высокий риск",
                "night_activity": "🌙 Ночная активность",
                "monitoring": "📊 Мониторинг",
                "protection": "🛡️ Защита",
                
                "metric_labels": "Metric Labels",
                "total_transactions": "💳 Всего транзакций",
                "total_amount": "💰 Общая сумма",
                "unique_customers": "👥 Уникальных клиентов",
                "recipients": "🎯 Получателей",
                "avg_transaction": "📈 Средняя сумма транзакции",
                "median_transaction": "📊 Медианная сумма",
                "most_frequent_type": "🔝 Самый частый тип",
                "data_period": "⏰ Период данных",
                "analyzed_total": "📊 Всего проанализировано",
                "suspicious_count": "⚠️ Подозрительных",
                "clean_count": "✅ Чистых",
                "fraud_level": "📉 Уровень мошенничества",
                
                "table_headers": "Table Headers",
                "transaction_id": "🆔 ID транзакции",
                "fraud_score": "🔍 Fraud Score",
                "is_suspicious": "🚩 Подозрительная",
                "explanation": "📝 Объяснение",
                "step": "⏱️ Шаг",
                "type": "🏷️ Тип",
                "amount": "💰 Сумма",
                "sender": "👤 Отправитель",
                "recipient": "🎯 Получатель",
                
                "self_learning": "Self-learning",
                "self_learning_status": "🤖 Статус самообучения:",
                "patterns_learned": "Изучено паттернов",
                "rules_adapted": "Адаптаций правил",
                "detection_effectiveness": "Эффективность обнаружения",
                
                "customer_risk_analysis": "Customer Risk Analysis",
                "top_risky_clients": "👥 Топ-10 клиентов с высоким риском",
                "customer": "Клиент",
                "risk_score": "Риск-скор",
                "suspicious_transactions": "Подозрительных транзакций",
                
                "transaction_analysis": "Transaction Analysis",
                "transaction_details": "Детали транзакции",
                "sender_balance_before": "Баланс отправителя до",
                "sender_balance_after": "Баланс отправителя после",
                "recipient_balance_before": "Баланс получателя до",
                "recipient_balance_after": "Баланс получателя после",
                
                "export_options": "Export Options",
                "export_format": "Выберите формат экспорта:",
                "html_report": "HTML отчет",
                "csv_data": "CSV данные",
                "excel_data": "Excel данные",
                "json_data": "JSON данные",
                
                "model_weights": "Model Weights",
                "isolation_forest_weight": "Isolation Forest вес",
                "autoencoder_weight": "AutoEncoder вес",
                "lstm_weight": "LSTM вес",
                
                "performance_metrics": "Performance Metrics",
                "accuracy": "Точность",
                "precision": "Точность положительных",
                "recall": "Полнота",
                "f1_score": "F1-мера",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                
                "fraud_patterns": "Fraud Patterns",
                "pattern_detection": "Обнаружение паттернов",
                "new_patterns_found": "Найдено новых паттернов",
                "pattern_quality": "Качество паттернов",
                
                "system_status": "System Status",
                "system_ready": "Система готова к работе",
                "processing": "Обработка...",
                "completed": "Завершено",
                
                "navigation": "Navigation",
                "home": "🏠 Главная",
                "dashboard": "📊 Панель",
                "reports": "📋 Отчеты",
                "settings_nav": "⚙️ Настройки",
                
                "confirmation": "Confirmation",
                "are_you_sure": "Вы уверены?",
                "confirm_action": "Подтвердить действие",
                "cancel": "Отмена",
                
                "file_operations": "File Operations",
                "browse_files": "Обзор файлов",
                "remove_file": "Удалить файл",
                "clear_all": "Очистить все",
                
                "data_preview": "Data Preview",
                "preview_data": "Предпросмотр данных",
                "show_more": "Показать больше",
                "show_less": "Показать меньше",
                
                "advanced_settings": "Advanced Settings",
                "advanced_options": "Расширенные настройки",
                "expert_mode": "Экспертный режим",
                "debug_mode": "Режим отладки",
                
                "help_support": "Help & Support",
                "documentation": "Документация",
                "faq": "Часто задаваемые вопросы",
                "contact_support": "Связаться с поддержкой",
                
                "version_info": "Version Info",
                "version": "Версия",
                "build": "Сборка",
                "release_date": "Дата выпуска",
                
                "legal": "Legal",
                "terms_of_service": "Условия использования",
                "privacy_policy": "Политика конфиденциальности",
                "license": "Лицензия",
                
                "credits": "Credits",
                "developed_by": "Разработано",
                "powered_by": "На базе",
                "special_thanks": "Особая благодарность",
                
                "feedback": "Feedback",
                "rate_app": "Оценить приложение",
                "send_feedback": "Отправить отзыв",
                "report_bug": "Сообщить об ошибке",
                
                "social_media": "Social Media",
                "follow_us": "Подписаться",
                "share": "Поделиться",
                
                "additional_features": "Additional Features",
                "real_time_monitoring": "Мониторинг в реальном времени",
                "automated_alerts": "Автоматические оповещения",
                "custom_rules": "Пользовательские правила",
                
                "security": "Security",
                "data_encryption": "Шифрование данных",
                "secure_connection": "Безопасное соединение",
                "access_control": "Контроль доступа",
                
                "performance": "Performance",
                "fast_processing": "Быстрая обработка",
                "low_latency": "Низкая задержка",
                "high_throughput": "Высокая пропускная способность",
                
                "compatibility": "Compatibility",
                "cross_platform": "Кроссплатформенность",
                "mobile_friendly": "Мобильная версия",
                "browser_support": "Поддержка браузеров",
                
                "customization": "Customization",
                "themes": "Темы",
                "layouts": "Макеты",
                "widgets": "Виджеты",
                
                "integration": "Integration",
                "api_access": "Доступ к API",
                "third_party": "Сторонние интеграции",
                "plugins": "Плагины",
                
                "backup_restore": "Backup & Restore",
                "backup_data": "Резервное копирование",
                "restore_data": "Восстановление данных",
                "sync_across_devices": "Синхронизация между устройствами",
                
                "analytics": "Analytics",
                "trends": "Тренды",
                "insights": "Инсайты",
                "predictions": "Прогнозы",
                
                "notifications": "Notifications",
                "email_alerts": "Email оповещения",
                "sms_alerts": "SMS оповещения",
                "push_notifications": "Push-уведомления",
                
                "collaboration": "Collaboration",
                "team_access": "Доступ для команды",
                "sharing_options": "Опции совместного использования",
                "permissions": "Права доступа",
                
                "training": "Training",
                "tutorials": "Обучающие материалы",
                "webinars": "Вебинары",
                "certification": "Сертификация",
                
                "support": "Support",
                "live_chat": "Живой чат",
                "phone_support": "Телефонная поддержка",
                "ticket_system": "Система тикетов",
                
                "updates": "Updates",
                "check_for_updates": "Проверить обновления",
                "auto_update": "Автообновление",
                "release_notes": "Примечания к выпуску",
                
                "account": "Account",
                "profile": "Профиль",
                "preferences": "Предпочтения",
                "logout": "Выйти",
                
                "billing": "Billing",
                "subscription": "Подписка",
                "payment_methods": "Способы оплаты",
                "invoices": "Счета",
                
                "miscellaneous": "Miscellaneous",
                "about": "О нас",
                "careers": "Карьера",
                "press": "Пресса",
                
                "end_of_translations": "END_OF_TRANSLATIONS"
            },
            "en": {
                "app_title": "🛡️ ClearFlow Security",
                "app_subtitle": "Intelligent Financial Fraud Detection System",
                "settings": "⚙️ Settings",
                "theme": "🎨 Theme",
                "language": "🌍 Language",
                "dark_theme": "🌑 Dark",
                "light_theme": "☀️ Light",
                "russian": "🇷🇺 Русский",
                "english": "🇬🇧 English",
                
                "upload_title": "📁 Upload Transaction Data",
                "upload_instruction": "Drag and drop CSV or Excel files here or click to select (maximum 5 files)",
                "file_help": "File should contain columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest",
                
                "analysis_title": "🔄 File Analysis",
                "separate_analysis": "Separately for each file",
                "combined_analysis": "Combine all files",
                "confirm_files": "✅ Confirm files and continue",
                
                "test_data": "📊 Test Data",
                "test_file_select": "Select test file:",
                "no_test_file": "Do not use",
                "normal_transactions": "Normal transactions",
                "fraud_transactions": "Fraudulent transactions",
                "mixed_transactions": "Mixed transactions",
                "load_test_file": "📥 Load test file",
                
                "model_selection": "🤖 Select ML Models:",
                "model_help": "Isolation Forest - fast model, AutoEncoder - neural network decodes data, LSTM - sequence analysis",
                "confirm_models": "✅ Confirm model selection",
                "models_confirmed": "Models confirmed",
                "reset_confirmation": "↩️ Reset confirmation",
                "models_reset": "Model selection reset",
                "selected_models": "Selected: ",
                
                "contamination_level": "📊 Expected fraud level:",
                "contamination_help": "Percentage of transactions you expect to see as fraudulent",
                
                "advice": "💡 Tip: Start with Isolation Forest for quick analysis",
                
                "overview_tab": "📊 Overview",
                "score_dist_tab": "📈 Score Distribution",
                "time_analysis_tab": "⏰ Time Analysis",
                "feature_importance_tab": "🎯 Feature Importance",
                "model_eval_tab": "🔬 Model Evaluation",
                "heatmap_tab": "🔥 Heatmap",
                "amount_dist_tab": "💰 Amount Distribution",
                
                "top_risky_clients_tab": "👥 Top Risky Clients",
                "timeline_tab": "⏰ Timeline",
                "type_stats_tab": "📊 Type Statistics",
                "recommendations_tab": "🎯 Recommendations",
                
                "export_title": "💾 Export Results",
                "export_all": "📦 Export all results",
                "download_suspicious": "📥 Download suspicious transactions (CSV)",
                
                "instructions_title": "📖 FraudDetect 2.0 User Guide",
                "how_to_use": "🚀 How to use the system:",
                "usage_step1": "1. **📁 Upload a CSV file** with financial transaction data",
                "usage_step2": "2. **⚙️ Configure detection parameters** in the sidebar",
                "usage_step3": "3. **⏳ Wait** for analysis completion",
                "usage_step4": "4. **📊 Review results** in the interactive dashboard",
                "usage_step5": "5. **💾 Export** results if needed",
                
                "csv_format": "📋 CSV File Format:",
                "csv_step": "`step` | Transaction time step",
                "csv_type": "`type` | Transaction type (PAYMENT, TRANSFER, etc.)",
                "csv_amount": "`amount` | Transaction amount",
                "csv_nameOrig": "`nameOrig` | Sender client ID",
                "csv_oldbalanceOrg": "`oldbalanceOrg` | Balance before transaction (sender)",
                "csv_newbalanceOrig": "`newbalanceOrig` | Balance after transaction (sender)",
                "csv_nameDest": "`nameDest` | Recipient ID",
                "csv_oldbalanceDest": "`oldbalanceDest` | Balance before transaction (recipient)",
                "csv_newbalanceDest": "`newbalanceDest` | Balance after transaction (recipient)",
                "csv_isFraud": "`isFraud` (optional) | Labels for evaluation (1 - fraud, 0 - normal)",
                
                "models_title": "🤖 Used Models:",
                "isolation_forest_desc": "**Isolation Forest** 🌲 - Detects anomalies based on isolation principle",
                "autoencoder_desc": "**AutoEncoder** 🧠 - Neural network that identifies reconstruction errors",
                "lstm_desc": "**LSTM AutoEncoder** ⏰ - Sequence analysis for temporal patterns",
                
                "interpretation_title": "🎯 Result Interpretation:",
                "fraud_score_desc": "- **Fraud Score** (0-1): Higher values indicate more suspicious transactions",
                "explanations_desc": "- **Explanations**: Show specific reasons why a transaction was flagged",
                "combined_approach_desc": "- **Combined approach**: 70% ML models + 30% rules = maximum accuracy",
                
                "fraud_rules_title": "💡 Fraud Detection Rules:",
                "rule_large_amounts": "⚠️ **Unusually large amounts** - relative to client history",
                "rule_new_accounts": "🆕 **Transactions to new accounts** - first transfer to recipient",
                "rule_account_drain": "💸 **Balance depletion** - withdrawal >90% of funds",
                "rule_unusual_time": "🌙 **Unusual time** - operations from 1 to 5 AM",
                "rule_activity_spike": "📈 **Activity spike** - sudden increase in transaction frequency",
                "rule_round_amounts": "💰 **Round amounts** - 1000, 5000, 10000 (often associated with fraud)",
                
                "buttons_and_actions": "Buttons and Actions",
                "start_analysis": "🚀 Start Analysis",
                "confirm_choice": "✅ Confirm Choice",
                "reset": "↩️ Reset",
                "export": "💾 Export",
                "download": "📥 Download",
                
                "status_messages": "Status Messages",
                "files_uploaded": "📂 Files uploaded: ",
                "analysis_complete": "🎉 Analysis complete! Results are ready",
                "data_loaded": "✅ Data successfully loaded and processed!",
                "processing_data": "🔄 Loading and processing data...",
                "analyzing_transactions": "🔍 Analyzing transactions with AI...",
                "exporting_results": "🔄 Exporting results...",
                "results_exported": "✅ Results successfully exported!",
                
                "error_messages": "Error Messages",
                "error_processing": "❌ Preprocessing error: ",
                "error_analysis": "❌ Data analysis error: ",
                "error_export": "❌ Export error: ",
                "error_data_types": "❌ Data type error: ",
                "max_files_error": "❌ Maximum 5 files at once! Remove extra files.",
                
                "tips": "Tips",
                "tip_try_isolation_forest": "💡 Try:\n\n• Using only Isolation Forest\n• Check data format\n• Use one of the test files",
                "tip_reduce_file_size": "💡 Try:\n\n• Using only Isolation Forest\n• Reduce file size\n• Restart the application",
                
                "risk_levels": "Risk Levels",
                "critical": "🔴 Critical",
                "attention": "🟡 Attention",
                "normal": "🟢 Normal",
                "high_risk": "⚠️ High Risk",
                "night_activity": "🌙 Night Activity",
                "monitoring": "📊 Monitoring",
                "protection": "🛡️ Protection",
                
                "metric_labels": "Metric Labels",
                "total_transactions": "💳 Total Transactions",
                "total_amount": "💰 Total Amount",
                "unique_customers": "👥 Unique Customers",
                "recipients": "🎯 Recipients",
                "avg_transaction": "📈 Average Transaction Amount",
                "median_transaction": "📊 Median Amount",
                "most_frequent_type": "🔝 Most Frequent Type",
                "data_period": "⏰ Data Period",
                "analyzed_total": "📊 Total Analyzed",
                "suspicious_count": "⚠️ Suspicious",
                "clean_count": "✅ Clean",
                "fraud_level": "📉 Fraud Level",
                
                "table_headers": "Table Headers",
                "transaction_id": "🆔 Transaction ID",
                "fraud_score": "🔍 Fraud Score",
                "is_suspicious": "🚩 Suspicious",
                "explanation": "📝 Explanation",
                "step": "⏱️ Step",
                "type": "🏷️ Type",
                "amount": "💰 Amount",
                "sender": "👤 Sender",
                "recipient": "🎯 Recipient",
                
                "self_learning": "Self-learning",
                "self_learning_status": "🤖 Self-Learning Status:",
                "patterns_learned": "Patterns Learned",
                "rules_adapted": "Rule Adaptations",
                "detection_effectiveness": "Detection Effectiveness",
                
                "customer_risk_analysis": "Customer Risk Analysis",
                "top_risky_clients": "👥 Top 10 High-Risk Clients",
                "customer": "Customer",
                "risk_score": "Risk Score",
                "suspicious_transactions": "Suspicious Transactions",
                
                "transaction_analysis": "Transaction Analysis",
                "transaction_details": "Transaction Details",
                "sender_balance_before": "Sender Balance Before",
                "sender_balance_after": "Sender Balance After",
                "recipient_balance_before": "Recipient Balance Before",
                "recipient_balance_after": "Recipient Balance After",
                
                "export_options": "Export Options",
                "export_format": "Select export format:",
                "html_report": "HTML Report",
                "csv_data": "CSV Data",
                "excel_data": "Excel Data",
                "json_data": "JSON Data",
                
                "model_weights": "Model Weights",
                "isolation_forest_weight": "Isolation Forest weight",
                "autoencoder_weight": "AutoEncoder weight",
                "lstm_weight": "LSTM weight",
                
                "performance_metrics": "Performance Metrics",
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1-Score",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                
                "fraud_patterns": "Fraud Patterns",
                "pattern_detection": "Pattern Detection",
                "new_patterns_found": "New Patterns Found",
                "pattern_quality": "Pattern Quality",
                
                "system_status": "System Status",
                "system_ready": "System ready for operation",
                "processing": "Processing...",
                "completed": "Completed",
                
                "navigation": "Navigation",
                "home": "🏠 Home",
                "dashboard": "📊 Dashboard",
                "reports": "📋 Reports",
                "settings_nav": "⚙️ Settings",
                
                "confirmation": "Confirmation",
                "are_you_sure": "Are you sure?",
                "confirm_action": "Confirm action",
                "cancel": "Cancel",
                
                "file_operations": "File Operations",
                "browse_files": "Browse files",
                "remove_file": "Remove file",
                "clear_all": "Clear all",
                
                "data_preview": "Data Preview",
                "preview_data": "Preview data",
                "show_more": "Show more",
                "show_less": "Show less",
                
                "advanced_settings": "Advanced Settings",
                "advanced_options": "Advanced options",
                "expert_mode": "Expert mode",
                "debug_mode": "Debug mode",
                
                "help_support": "Help & Support",
                "documentation": "Documentation",
                "faq": "Frequently Asked Questions",
                "contact_support": "Contact support",
                
                "version_info": "Version Info",
                "version": "Version",
                "build": "Build",
                "release_date": "Release date",
                
                "legal": "Legal",
                "terms_of_service": "Terms of Service",
                "privacy_policy": "Privacy Policy",
                "license": "License",
                
                "credits": "Credits",
                "developed_by": "Developed by",
                "powered_by": "Powered by",
                "special_thanks": "Special thanks",
                
                "feedback": "Feedback",
                "rate_app": "Rate app",
                "send_feedback": "Send feedback",
                "report_bug": "Report bug",
                
                "social_media": "Social Media",
                "follow_us": "Follow us",
                "share": "Share",
                
                "additional_features": "Additional Features",
                "real_time_monitoring": "Real-time monitoring",
                "automated_alerts": "Automated alerts",
                "custom_rules": "Custom rules",
                
                "security": "Security",
                "data_encryption": "Data encryption",
                "secure_connection": "Secure connection",
                "access_control": "Access control",
                
                "performance": "Performance",
                "fast_processing": "Fast processing",
                "low_latency": "Low latency",
                "high_throughput": "High throughput",
                
                "compatibility": "Compatibility",
                "cross_platform": "Cross-platform",
                "mobile_friendly": "Mobile friendly",
                "browser_support": "Browser support",
                
                "customization": "Customization",
                "themes": "Themes",
                "layouts": "Layouts",
                "widgets": "Widgets",
                
                "integration": "Integration",
                "api_access": "API access",
                "third_party": "Third-party integrations",
                "plugins": "Plugins",
                
                "backup_restore": "Backup & Restore",
                "backup_data": "Backup data",
                "restore_data": "Restore data",
                "sync_across_devices": "Sync across devices",
                
                "analytics": "Analytics",
                "trends": "Trends",
                "insights": "Insights",
                "predictions": "Predictions",
                
                "notifications": "Notifications",
                "email_alerts": "Email alerts",
                "sms_alerts": "SMS alerts",
                "push_notifications": "Push notifications",
                
                "collaboration": "Collaboration",
                "team_access": "Team access",
                "sharing_options": "Sharing options",
                "permissions": "Permissions",
                
                "training": "Training",
                "tutorials": "Tutorials",
                "webinars": "Webinars",
                "certification": "Certification",
                
                "support": "Support",
                "live_chat": "Live chat",
                "phone_support": "Phone support",
                "ticket_system": "Ticket system",
                
                "updates": "Updates",
                "check_for_updates": "Check for updates",
                "auto_update": "Auto-update",
                "release_notes": "Release notes",
                
                "account": "Account",
                "profile": "Profile",
                "preferences": "Preferences",
                "logout": "Logout",
                
                "billing": "Billing",
                "subscription": "Subscription",
                "payment_methods": "Payment methods",
                "invoices": "Invoices",
                
                "miscellaneous": "Miscellaneous",
                "about": "About",
                "careers": "Careers",
                "press": "Press",
                
                "end_of_translations": "END_OF_TRANSLATIONS"
            }
        }
        return translations
    
    def get_text(self, key, language=None):
        if language is None:
            language = self.default_language
        
        if language in self.translations and key in self.translations[language]:
            return self.translations[language][key]
        elif self.default_language in self.translations and key in self.translations[self.default_language]:
            return self.translations[self.default_language][key]
        else:
            return key

localization_manager = LocalizationManager()
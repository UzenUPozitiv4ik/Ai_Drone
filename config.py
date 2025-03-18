"""
Файл конфигурации для системы управления БПЛА
"""

# Настройки подключения к БПЛА
DRONE_CONNECTION = {
    "connection_string": "udp:127.0.0.1:14550",  # Строка подключения (для симулятора или реального устройства)
    "baud_rate": 57600,                          # Скорость передачи данных при последовательном подключении
    "wait_ready": True,                          # Ожидать готовность дрона перед выполнением команд
    "timeout": 30                                # Таймаут подключения в секундах
}

# Настройки GPS
GPS_SETTINGS = {
    "default_altitude": 30,                      # Высота полета по умолчанию (в метрах)
    "default_speed": 5,                          # Скорость полета по умолчанию (м/с)
    "waypoint_radius": 5,                        # Радиус достижения точки маршрута (в метрах)
    "home_radius": 10,                           # Радиус определения домашней точки (в метрах)
    "return_altitude": 40                        # Высота для возврата на базу (в метрах)
}

# Настройки обхода препятствий
OBSTACLE_AVOIDANCE = {
    "enabled": True,                             # Включить/выключить обход препятствий
    "min_distance": 3,                           # Минимальное расстояние до препятствия (в метрах)
    "safe_distance": 5,                          # Безопасное расстояние для маневра (в метрах)
    "scan_angle": 90,                            # Угол сканирования (в градусах)
    "avoidance_strategy": "stop_and_redirect"    # Стратегия обхода: "stop_and_redirect" или "dynamic_path"
}

# Настройки камеры и компьютерного зрения
VISION_SETTINGS = {
    "camera_index": 0,                           # Индекс камеры (0 для основной)
    "resolution": (640, 480),                    # Разрешение видео (ширина, высота)
    "fps": 30,                                   # Частота кадров
    "object_detection": {
        "enabled": True,                         # Включить/выключить обнаружение объектов
        "model_path": "models/detection_model.pt", # Путь к модели обнаружения
        "confidence_threshold": 0.6,             # Порог уверенности для обнаружения
        "classes_of_interest": [                 # Классы объектов, которые нужно обнаруживать
            "person", "car", "truck", "bicycle"
        ]
    },
    "save_images": True,                         # Сохранять ли изображения
    "save_path": "data/images/"                  # Путь для сохранения изображений
}

# Настройки телеметрии
TELEMETRY_SETTINGS = {
    "enabled": True,                             # Включить/выключить передачу телеметрии
    "broker": "localhost",                       # MQTT брокер
    "port": 1883,                                # Порт MQTT
    "topic_prefix": "drone/telemetry/",          # Префикс темы
    "update_frequency": 1.0,                     # Частота обновления данных (Гц)
    "log_to_file": True,                         # Логировать телеметрию в файл
    "log_path": "logs/telemetry/",               # Путь для логов телеметрии
    "include_data": [                            # Типы данных для включения в телеметрию
        "location", "attitude", "velocity", 
        "battery", "system_status", "armed"
    ]
}

# Настройки логирования
LOGGING = {
    "level": "INFO",                             # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    "log_to_file": True,                         # Логировать в файл
    "log_path": "logs/system.log",               # Путь к файлу лога
    "max_log_size": 10 * 1024 * 1024,            # Максимальный размер лог-файла (10 МБ)
    "backup_count": 5                            # Количество файлов ротации логов
}

# Настройки безопасности
SAFETY_SETTINGS = {
    "geofence_enabled": True,                    # Включить/выключить виртуальный забор
    "max_distance": 500,                         # Максимальная дистанция от точки взлета (в метрах)
    "max_altitude": 120,                         # Максимальная высота (в метрах)
    "voltage_threshold": 10.5,                   # Минимальное напряжение батареи для RTL (Return To Launch)
    "failsafe_enabled": True,                    # Включить/выключить режим failsafe
    "return_home_on_low_battery": True,          # Возвращаться домой при низком заряде
    "auto_land_on_critical_battery": True        # Автоматическая посадка при критическом заряде
} 
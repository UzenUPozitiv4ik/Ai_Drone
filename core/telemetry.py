"""
Модуль для сбора и передачи телеметрии БПЛА
"""

import time
import json
import logging
import os
import threading
from typing import Dict, Any, Optional, List, Callable
import paho.mqtt.client as mqtt
import sys
import socket
from datetime import datetime

# Добавляем корневую директорию проекта в path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TELEMETRY_SETTINGS
from core.drone_controller import DroneController

class TelemetryManager:
    """
    Класс для управления сбором и передачей телеметрии БПЛА.
    Поддерживает передачу данных через MQTT и/или сохранение в файл.
    """
    
    def __init__(self, drone_controller: DroneController):
        """
        Инициализирует менеджер телеметрии.
        
        Args:
            drone_controller: Контроллер БПЛА
        """
        self.logger = logging.getLogger(__name__)
        self.drone = drone_controller
        
        # Настройки телеметрии
        self.enabled = TELEMETRY_SETTINGS["enabled"]
        self.broker = TELEMETRY_SETTINGS["broker"]
        self.port = TELEMETRY_SETTINGS["port"]
        self.topic_prefix = TELEMETRY_SETTINGS["topic_prefix"]
        self.update_frequency = TELEMETRY_SETTINGS["update_frequency"]
        self.log_to_file = TELEMETRY_SETTINGS["log_to_file"]
        self.log_path = TELEMETRY_SETTINGS["log_path"]
        self.include_data = TELEMETRY_SETTINGS["include_data"]
        
        # Инициализация MQTT-клиента
        self.mqtt_client = None
        self.mqtt_connected = False
        
        # Поток для периодической передачи телеметрии
        self.telemetry_thread = None
        self.running = False
        
        # Обработчики для различных типов данных телеметрии
        self.telemetry_handlers = {}
        
        # ID сессии телеметрии
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Файл для логирования телеметрии
        self.log_file = None
        self._init_log_file()
        
    def _init_log_file(self) -> None:
        """Инициализирует файл для логирования телеметрии"""
        if self.log_to_file:
            try:
                # Создаем директорию для логов, если она не существует
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                
                # Формируем имя файла логов с датой и временем
                log_filename = f"{self.log_path}/telemetry_{self.session_id}.csv"
                
                # Открываем файл для записи
                self.log_file = open(log_filename, 'w')
                
                # Записываем заголовок CSV
                headers = ["timestamp", "armed", "mode"]
                
                for data_type in self.include_data:
                    if data_type == "location":
                        headers.extend(["lat", "lon", "alt"])
                    elif data_type == "attitude":
                        headers.extend(["roll", "pitch", "yaw"])
                    elif data_type == "velocity":
                        headers.extend(["vx", "vy", "vz"])
                    elif data_type == "battery":
                        headers.extend(["voltage", "current", "level"])
                    else:
                        headers.append(data_type)
                        
                self.log_file.write(",".join(headers) + "\n")
                self.log_file.flush()
                
                self.logger.info(f"Файл логирования телеметрии создан: {log_filename}")
                
            except Exception as e:
                self.logger.error(f"Ошибка при инициализации файла логирования телеметрии: {str(e)}")
                self.log_file = None
                
    def _log_telemetry_to_file(self, telemetry: Dict[str, Any]) -> None:
        """
        Записывает данные телеметрии в файл.
        
        Args:
            telemetry: Словарь с данными телеметрии
        """
        if not self.log_to_file or not self.log_file:
            return
            
        try:
            # Формируем строку с данными в формате CSV
            values = [str(telemetry["timestamp"]), str(telemetry["armed"]), telemetry["mode"]]
            
            for data_type in self.include_data:
                if data_type == "location" and "location" in telemetry:
                    values.extend([
                        str(telemetry["location"].get("lat", "")),
                        str(telemetry["location"].get("lon", "")),
                        str(telemetry["location"].get("alt", ""))
                    ])
                elif data_type == "attitude" and "attitude" in telemetry:
                    values.extend([
                        str(telemetry["attitude"].get("roll", "")),
                        str(telemetry["attitude"].get("pitch", "")),
                        str(telemetry["attitude"].get("yaw", ""))
                    ])
                elif data_type == "velocity" and "velocity" in telemetry:
                    values.extend([
                        str(telemetry["velocity"].get("vx", "")),
                        str(telemetry["velocity"].get("vy", "")),
                        str(telemetry["velocity"].get("vz", ""))
                    ])
                elif data_type == "battery" and "battery" in telemetry:
                    values.extend([
                        str(telemetry["battery"].get("voltage", "")),
                        str(telemetry["battery"].get("current", "")),
                        str(telemetry["battery"].get("level", ""))
                    ])
                elif data_type in telemetry:
                    values.append(str(telemetry[data_type]))
                else:
                    values.append("")
                    
            self.log_file.write(",".join(values) + "\n")
            self.log_file.flush()
            
        except Exception as e:
            self.logger.error(f"Ошибка при записи телеметрии в файл: {str(e)}")
            
    def start(self) -> bool:
        """
        Запускает сбор и передачу телеметрии.
        
        Returns:
            bool: True если телеметрия успешно запущена
        """
        if not self.enabled:
            self.logger.info("Телеметрия отключена в настройках")
            return False
            
        if self.running:
            self.logger.info("Телеметрия уже запущена")
            return True
            
        self.logger.info("Запуск системы телеметрии...")
        
        # Подключаемся к MQTT-брокеру
        if self._connect_mqtt():
            self.logger.info(f"Подключено к MQTT-брокеру: {self.broker}:{self.port}")
        else:
            self.logger.warning("Не удалось подключиться к MQTT-брокеру, телеметрия будет только логироваться")
            
        # Запускаем поток для периодической передачи телеметрии
        self.running = True
        self.telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self.telemetry_thread.start()
        
        self.logger.info("Система телеметрии успешно запущена")
        return True
        
    def stop(self) -> None:
        """Останавливает сбор и передачу телеметрии"""
        if not self.running:
            return
            
        self.logger.info("Остановка системы телеметрии...")
        
        # Останавливаем поток телеметрии
        self.running = False
        if self.telemetry_thread:
            self.telemetry_thread.join(timeout=2.0)
            
        # Отключаемся от MQTT-брокера
        if self.mqtt_client and self.mqtt_connected:
            self.mqtt_client.disconnect()
            self.mqtt_connected = False
            
        # Закрываем файл логирования
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            
        self.logger.info("Система телеметрии остановлена")
        
    def _connect_mqtt(self) -> bool:
        """
        Подключается к MQTT-брокеру.
        
        Returns:
            bool: True если подключение успешно
        """
        try:
            # Создаем клиента MQTT
            self.mqtt_client = mqtt.Client(client_id=f"drone_telemetry_{self.session_id}")
            
            # Настраиваем обработчики событий
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            # Устанавливаем таймаут для подключения
            self.mqtt_client.connect_async(self.broker, self.port, keepalive=60)
            
            # Запускаем фоновый поток для работы с MQTT
            self.mqtt_client.loop_start()
            
            # Ожидаем подключения
            wait_time = 0
            while not self.mqtt_connected and wait_time < 5:
                time.sleep(0.5)
                wait_time += 0.5
                
            return self.mqtt_connected
            
        except Exception as e:
            self.logger.error(f"Ошибка при подключении к MQTT-брокеру: {str(e)}")
            return False
            
    def _on_mqtt_connect(self, client, userdata, flags, rc) -> None:
        """Обработчик успешного подключения к MQTT-брокеру"""
        if rc == 0:
            self.mqtt_connected = True
            self.logger.info("Успешное подключение к MQTT-брокеру")
            
            # Публикуем сообщение о подключении
            self.publish_message("status", {
                "status": "connected",
                "session_id": self.session_id,
                "timestamp": time.time(),
                "drone_id": socket.gethostname()
            })
        else:
            self.logger.error(f"Ошибка подключения к MQTT-брокеру, код: {rc}")
            
    def _on_mqtt_disconnect(self, client, userdata, rc) -> None:
        """Обработчик отключения от MQTT-брокера"""
        self.mqtt_connected = False
        if rc != 0:
            self.logger.warning(f"Неожиданное отключение от MQTT-брокера, код: {rc}")
        else:
            self.logger.info("Отключение от MQTT-брокера")
            
    def publish_message(self, topic: str, message: Dict[str, Any]) -> bool:
        """
        Публикует сообщение по указанной теме.
        
        Args:
            topic: Тема для публикации (будет добавлена к префиксу)
            message: Сообщение для публикации (будет преобразовано в JSON)
            
        Returns:
            bool: True если сообщение успешно опубликовано
        """
        if not self.mqtt_client or not self.mqtt_connected:
            return False
            
        try:
            # Формируем полную тему
            full_topic = f"{self.topic_prefix}{topic}"
            
            # Преобразуем сообщение в JSON
            json_message = json.dumps(message)
            
            # Публикуем сообщение
            self.mqtt_client.publish(full_topic, json_message, qos=0, retain=False)
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при публикации сообщения: {str(e)}")
            return False
            
    def _telemetry_loop(self) -> None:
        """Фоновый поток для периодической передачи телеметрии"""
        self.logger.info(f"Запущен поток телеметрии с частотой {self.update_frequency} Гц")
        
        # Вычисляем интервал между обновлениями в секундах
        update_interval = 1.0 / self.update_frequency
        
        while self.running:
            start_time = time.time()
            
            # Собираем данные телеметрии
            telemetry = self.collect_telemetry()
            
            # Публикуем данные через MQTT
            if self.mqtt_client and self.mqtt_connected:
                self.publish_message("data", telemetry)
                
            # Записываем данные в файл
            if self.log_to_file:
                self._log_telemetry_to_file(telemetry)
                
            # Вычисляем оставшееся время до следующего обновления
            elapsed_time = time.time() - start_time
            sleep_time = max(0, update_interval - elapsed_time)
            
            # Спим до следующего обновления
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def collect_telemetry(self) -> Dict[str, Any]:
        """
        Собирает данные телеметрии с дрона.
        
        Returns:
            Dict[str, Any]: Словарь с данными телеметрии
        """
        # Получаем базовую телеметрию от контроллера дрона
        telemetry = self.drone.get_telemetry()
        
        # Добавляем собственные данные телеметрии
        telemetry["session_id"] = self.session_id
        telemetry["system_time"] = datetime.now().isoformat()
        
        # Вызываем обработчики для добавления дополнительных данных
        for data_type, handler in self.telemetry_handlers.items():
            try:
                telemetry[data_type] = handler()
            except Exception as e:
                self.logger.error(f"Ошибка в обработчике телеметрии {data_type}: {str(e)}")
                
        return telemetry
        
    def register_telemetry_handler(self, data_type: str, handler: Callable[[], Any]) -> None:
        """
        Регистрирует обработчик для сбора дополнительных данных телеметрии.
        
        Args:
            data_type: Тип данных, которые собирает обработчик
            handler: Функция-обработчик, которая возвращает данные телеметрии
        """
        self.telemetry_handlers[data_type] = handler
        self.logger.info(f"Зарегистрирован обработчик телеметрии для типа данных '{data_type}'")
        
    def unregister_telemetry_handler(self, data_type: str) -> None:
        """
        Удаляет обработчик для указанного типа данных.
        
        Args:
            data_type: Тип данных, для которого нужно удалить обработчик
        """
        if data_type in self.telemetry_handlers:
            del self.telemetry_handlers[data_type]
            self.logger.info(f"Удален обработчик телеметрии для типа данных '{data_type}'")

# Класс для работы с телеметрическими данными
class TelemetryAnalyzer:
    """
    Класс для анализа телеметрических данных.
    Может использоваться для выявления аномалий, предсказания поведения
    и визуализации данных телеметрии.
    """
    
    def __init__(self, log_path: Optional[str] = None):
        """
        Инициализирует анализатор телеметрии.
        
        Args:
            log_path: Путь к файлу логов телеметрии. Если None, используется из настроек.
        """
        self.logger = logging.getLogger(__name__)
        self.log_path = log_path or TELEMETRY_SETTINGS["log_path"]
        
        # Загруженные данные телеметрии
        self.telemetry_data = []
        
    def load_telemetry_from_file(self, filename: Optional[str] = None) -> bool:
        """
        Загружает данные телеметрии из файла.
        
        Args:
            filename: Имя файла с данными телеметрии. Если None, используется последний файл.
            
        Returns:
            bool: True если данные успешно загружены
        """
        try:
            # Если имя файла не указано, пытаемся найти последний файл логов
            if filename is None:
                if not os.path.exists(self.log_path):
                    self.logger.error(f"Директория логов {self.log_path} не существует")
                    return False
                    
                # Находим самый новый файл логов
                files = [f for f in os.listdir(self.log_path) if f.startswith("telemetry_") and f.endswith(".csv")]
                if not files:
                    self.logger.error("Файлы логов телеметрии не найдены")
                    return False
                    
                filename = os.path.join(self.log_path, max(files, key=lambda f: os.path.getmtime(os.path.join(self.log_path, f))))
            else:
                # Используем указанное имя файла
                if not os.path.exists(filename):
                    self.logger.error(f"Файл {filename} не существует")
                    return False
                    
            # Загружаем данные из файла
            self.logger.info(f"Загрузка данных телеметрии из файла {filename}")
            
            # Читаем файл и разбираем CSV
            with open(filename, 'r') as file:
                # Читаем заголовок
                headers = file.readline().strip().split(',')
                
                # Читаем данные
                self.telemetry_data = []
                for line in file:
                    values = line.strip().split(',')
                    data = {headers[i]: values[i] for i in range(min(len(headers), len(values)))}
                    self.telemetry_data.append(data)
                    
            self.logger.info(f"Загружено {len(self.telemetry_data)} записей телеметрии")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных телеметрии: {str(e)}")
            return False
            
    def get_average_values(self, field: str) -> Dict[str, float]:
        """
        Вычисляет средние значения указанного поля телеметрии.
        
        Args:
            field: Название поля телеметрии (например, "altitude", "velocity", и т.д.)
            
        Returns:
            Dict[str, float]: Словарь со средними значениями
        """
        if not self.telemetry_data:
            return {}
            
        try:
            # Собираем все значения поля
            values = []
            for data in self.telemetry_data:
                if field in data:
                    try:
                        value = float(data[field])
                        values.append(value)
                    except (ValueError, TypeError):
                        pass
                        
            if not values:
                return {}
                
            # Вычисляем статистику
            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            
            return {
                "average": avg,
                "min": min_val,
                "max": max_val,
                "count": len(values)
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при вычислении средних значений: {str(e)}")
            return {}
            
    def detect_anomalies(self, field: str, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Обнаруживает аномалии в указанном поле телеметрии.
        
        Args:
            field: Название поля телеметрии
            threshold: Порог для обнаружения аномалий (в стандартных отклонениях)
            
        Returns:
            List[Dict[str, Any]]: Список обнаруженных аномалий
        """
        if not self.telemetry_data:
            return []
            
        try:
            # Собираем все значения поля
            values = []
            for data in self.telemetry_data:
                if field in data:
                    try:
                        value = float(data[field])
                        values.append((data, value))
                    except (ValueError, TypeError):
                        pass
                        
            if not values:
                return []
                
            # Вычисляем среднее и стандартное отклонение
            vals = [v[1] for v in values]
            avg = sum(vals) / len(vals)
            std_dev = (sum((v - avg) ** 2 for v in vals) / len(vals)) ** 0.5
            
            # Находим аномалии
            anomalies = []
            for data, value in values:
                z_score = abs(value - avg) / std_dev if std_dev > 0 else 0
                if z_score > threshold:
                    anomalies.append({
                        "timestamp": data.get("timestamp", "unknown"),
                        "value": value,
                        "z_score": z_score,
                        "data": data
                    })
                    
            return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Ошибка при обнаружении аномалий: {str(e)}")
            return []
            
    def export_to_json(self, filename: str) -> bool:
        """
        Экспортирует данные телеметрии в JSON-файл.
        
        Args:
            filename: Имя файла для экспорта
            
        Returns:
            bool: True если данные успешно экспортированы
        """
        if not self.telemetry_data:
            self.logger.warning("Нет данных для экспорта")
            return False
            
        try:
            with open(filename, 'w') as file:
                json.dump(self.telemetry_data, file, indent=2)
                
            self.logger.info(f"Данные телеметрии экспортированы в файл {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при экспорте данных телеметрии: {str(e)}")
            return False 
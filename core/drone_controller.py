"""
Модуль для базового управления БПЛА
"""

import time
import math
import logging
from typing import Tuple, List, Dict, Any, Optional
from dronekit import connect, VehicleMode, Vehicle, LocationGlobalRelative, LocationGlobal
import sys
import os

# Добавляем корневую директорию проекта в path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DRONE_CONNECTION, GPS_SETTINGS, SAFETY_SETTINGS

class DroneController:
    """
    Базовый класс для управления БПЛА с использованием DroneKit.
    Предоставляет основные функции управления для полета, навигации и мониторинга.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Инициализирует контроллер дрона.
        
        Args:
            connection_string: Строка подключения к дрону. Если None, используется из конфигурации.
        """
        self.logger = logging.getLogger(__name__)
        self.conn_string = connection_string or DRONE_CONNECTION["connection_string"]
        self.baud_rate = DRONE_CONNECTION["baud_rate"]
        self.wait_ready = DRONE_CONNECTION["wait_ready"]
        self.timeout = DRONE_CONNECTION["timeout"]
        self.vehicle = None
        self.home_location = None
        self.is_armed = False
        self.mode = None
        
        # Флаги для отслеживания состояния
        self.mission_in_progress = False
        self.return_to_home_in_progress = False
        self.obstacle_avoiding = False
        
    def connect(self) -> bool:
        """
        Подключается к БПЛА.
        
        Returns:
            bool: True если подключение успешно, иначе False
        """
        try:
            self.logger.info(f"Подключение к БПЛА через {self.conn_string}")
            self.vehicle = connect(
                self.conn_string, 
                baud=self.baud_rate,
                wait_ready=self.wait_ready,
                timeout=self.timeout
            )
            
            if self.vehicle:
                self.logger.info("Подключение к БПЛА успешно установлено")
                self.mode = self.vehicle.mode.name
                self.is_armed = self.vehicle.armed
                
                # Ждем получения домашней позиции
                self._wait_for_home_location()
                return True
            else:
                self.logger.error("Ошибка подключения: vehicle объект не создан")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка подключения к БПЛА: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Отключается от БПЛА."""
        if self.vehicle:
            self.logger.info("Отключение от БПЛА")
            self.vehicle.close()
            self.vehicle = None
    
    def arm(self) -> bool:
        """
        Активирует моторы БПЛА.
        
        Returns:
            bool: True если активация успешна, иначе False
        """
        if not self.vehicle:
            self.logger.error("Ошибка активации: нет подключения к БПЛА")
            return False
            
        if self.vehicle.armed:
            self.logger.info("БПЛА уже активирован")
            return True
            
        # Устанавливаем режим GUIDED для активации
        self.set_mode("GUIDED")
        
        self.logger.info("Активация БПЛА...")
        self.vehicle.armed = True
        
        # Ждем активации
        timeout = 10
        start_time = time.time()
        while not self.vehicle.armed:
            if time.time() - start_time > timeout:
                self.logger.error("Тайм-аут при ожидании активации БПЛА")
                return False
            time.sleep(0.5)
            
        self.is_armed = True
        self.logger.info("БПЛА успешно активирован")
        return True
    
    def disarm(self) -> bool:
        """
        Деактивирует моторы БПЛА.
        
        Returns:
            bool: True если деактивация успешна, иначе False
        """
        if not self.vehicle:
            self.logger.error("Ошибка деактивации: нет подключения к БПЛА")
            return False
            
        if not self.vehicle.armed:
            self.logger.info("БПЛА уже деактивирован")
            return True
            
        self.logger.info("Деактивация БПЛА...")
        self.vehicle.armed = False
        
        # Ждем деактивации
        timeout = 10
        start_time = time.time()
        while self.vehicle.armed:
            if time.time() - start_time > timeout:
                self.logger.error("Тайм-аут при ожидании деактивации БПЛА")
                return False
            time.sleep(0.5)
            
        self.is_armed = False
        self.logger.info("БПЛА успешно деактивирован")
        return True
    
    def takeoff(self, target_altitude: Optional[float] = None) -> bool:
        """
        Взлет БПЛА на указанную высоту.
        
        Args:
            target_altitude: Целевая высота в метрах. Если None, использует высоту по умолчанию.
            
        Returns:
            bool: True если взлет успешен, иначе False
        """
        if not self.vehicle:
            self.logger.error("Ошибка взлета: нет подключения к БПЛА")
            return False
            
        if not self.vehicle.armed:
            self.logger.info("БПЛА не активирован, выполняем активацию")
            if not self.arm():
                return False
        
        # Устанавливаем режим GUIDED для взлета
        self.set_mode("GUIDED")
        
        # Определяем высоту взлета
        altitude = target_altitude if target_altitude is not None else GPS_SETTINGS["default_altitude"]
        
        self.logger.info(f"Взлет на высоту {altitude} м")
        self.vehicle.simple_takeoff(altitude)
        
        # Ждем достижения высоты
        self._wait_for_altitude(altitude)
        
        self.logger.info(f"Взлет завершен, достигнута высота {self.vehicle.location.global_relative_frame.alt} м")
        return True
    
    def land(self) -> bool:
        """
        Посадка БПЛА.
        
        Returns:
            bool: True если посадка успешно инициирована
        """
        if not self.vehicle:
            self.logger.error("Ошибка посадки: нет подключения к БПЛА")
            return False
            
        self.logger.info("Начинаем посадку БПЛА")
        self.set_mode("LAND")
        
        return True
    
    def return_to_home(self) -> bool:
        """
        Инициирует возврат БПЛА на точку взлета.
        
        Returns:
            bool: True если возврат успешно инициирован
        """
        if not self.vehicle:
            self.logger.error("Ошибка возврата: нет подключения к БПЛА")
            return False
            
        self.logger.info("Начинаем возврат БПЛА на базу")
        
        # Если задана высота для возврата, поднимаемся на неё перед возвратом
        return_altitude = GPS_SETTINGS["return_altitude"]
        if self.vehicle.location.global_relative_frame.alt < return_altitude:
            self.logger.info(f"Поднимаемся на безопасную высоту {return_altitude} м перед возвратом")
            self.set_mode("GUIDED")
            self.goto(
                self.vehicle.location.global_relative_frame.lat,
                self.vehicle.location.global_relative_frame.lon,
                return_altitude
            )
            self._wait_for_altitude(return_altitude)
        
        # Устанавливаем режим RTL (Return To Launch)
        self.set_mode("RTL")
        self.return_to_home_in_progress = True
        
        return True
    
    def set_mode(self, mode_name: str) -> bool:
        """
        Устанавливает режим полета БПЛА.
        
        Args:
            mode_name: Название режима полета ("GUIDED", "AUTO", "RTL", "LAND", и т.д.)
            
        Returns:
            bool: True если режим успешно установлен
        """
        if not self.vehicle:
            self.logger.error(f"Ошибка установки режима {mode_name}: нет подключения к БПЛА")
            return False
            
        try:
            self.logger.info(f"Устанавливаем режим полета {mode_name}")
            self.vehicle.mode = VehicleMode(mode_name)
            
            # Ждем установки режима
            timeout = 5
            start_time = time.time()
            while self.vehicle.mode.name != mode_name:
                if time.time() - start_time > timeout:
                    self.logger.error(f"Тайм-аут при установке режима {mode_name}")
                    return False
                time.sleep(0.5)
                
            self.mode = mode_name
            self.logger.info(f"Режим {mode_name} успешно установлен")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при установке режима {mode_name}: {str(e)}")
            return False
    
    def goto(self, latitude: float, longitude: float, altitude: Optional[float] = None) -> bool:
        """
        Отправляет БПЛА в указанную точку с заданной высотой.
        
        Args:
            latitude: Широта точки назначения
            longitude: Долгота точки назначения
            altitude: Высота полета (в метрах). Если None, сохраняет текущую высоту.
            
        Returns:
            bool: True если команда успешно отправлена
        """
        if not self.vehicle:
            self.logger.error("Ошибка перемещения: нет подключения к БПЛА")
            return False
            
        # Устанавливаем режим GUIDED для перемещения
        self.set_mode("GUIDED")
        
        # Если высота не указана, используем текущую
        if altitude is None:
            altitude = self.vehicle.location.global_relative_frame.alt
        
        self.logger.info(f"Отправляем БПЛА в точку: lat={latitude}, lon={longitude}, alt={altitude}")
        
        # Создаем целевую локацию и отправляем команду
        target_location = LocationGlobalRelative(latitude, longitude, altitude)
        self.vehicle.simple_goto(target_location)
        
        return True
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Получает текущую телеметрию дрона.
        
        Returns:
            Dict: Словарь с данными телеметрии
        """
        if not self.vehicle:
            self.logger.error("Ошибка получения телеметрии: нет подключения к БПЛА")
            return {}
            
        telemetry = {
            "timestamp": time.time(),
            "armed": self.vehicle.armed,
            "mode": self.vehicle.mode.name,
            "location": {
                "lat": self.vehicle.location.global_relative_frame.lat if self.vehicle.location.global_relative_frame else None,
                "lon": self.vehicle.location.global_relative_frame.lon if self.vehicle.location.global_relative_frame else None,
                "alt": self.vehicle.location.global_relative_frame.alt if self.vehicle.location.global_relative_frame else None
            },
            "attitude": {
                "roll": self.vehicle.attitude.roll if self.vehicle.attitude else None,
                "pitch": self.vehicle.attitude.pitch if self.vehicle.attitude else None,
                "yaw": self.vehicle.attitude.yaw if self.vehicle.attitude else None
            },
            "velocity": {
                "vx": self.vehicle.velocity[0] if self.vehicle.velocity else None,
                "vy": self.vehicle.velocity[1] if self.vehicle.velocity else None,
                "vz": self.vehicle.velocity[2] if self.vehicle.velocity else None
            },
            "battery": {
                "voltage": self.vehicle.battery.voltage if self.vehicle.battery else None,
                "current": self.vehicle.battery.current if self.vehicle.battery else None,
                "level": self.vehicle.battery.level if self.vehicle.battery and hasattr(self.vehicle.battery, 'level') else None
            },
            "system_status": self.vehicle.system_status.state if self.vehicle.system_status else None,
            "airspeed": self.vehicle.airspeed if hasattr(self.vehicle, 'airspeed') else None,
            "groundspeed": self.vehicle.groundspeed if hasattr(self.vehicle, 'groundspeed') else None,
            "heading": self.vehicle.heading if hasattr(self.vehicle, 'heading') else None,
            "last_heartbeat": self.vehicle.last_heartbeat if hasattr(self.vehicle, 'last_heartbeat') else None
        }
        
        return telemetry
    
    def get_home_location(self) -> Tuple[float, float, float]:
        """
        Получает координаты домашней точки (точки взлета).
        
        Returns:
            Tuple[float, float, float]: Координаты (lat, lon, alt) домашней точки или (None, None, None)
        """
        if not self.vehicle or not self.home_location:
            return (None, None, None)
            
        return (
            self.home_location.lat,
            self.home_location.lon,
            self.home_location.alt
        )
    
    def check_battery(self) -> bool:
        """
        Проверяет состояние батареи и инициирует возврат при низком заряде.
        
        Returns:
            bool: True если батарея в порядке, False если низкий заряд
        """
        if not self.vehicle or not self.vehicle.battery:
            return True
            
        voltage = self.vehicle.battery.voltage
        threshold = SAFETY_SETTINGS["voltage_threshold"]
        
        if voltage is not None and voltage < threshold:
            self.logger.warning(f"Низкий заряд батареи: {voltage}V ниже порога {threshold}V")
            
            if SAFETY_SETTINGS["return_home_on_low_battery"]:
                self.logger.info("Инициируем возврат на базу из-за низкого заряда батареи")
                self.return_to_home()
                
            return False
            
        return True
    
    def check_geofence(self) -> bool:
        """
        Проверяет, не вышел ли дрон за границы геозоны.
        
        Returns:
            bool: True если дрон в границах, False если вышел за границы
        """
        if not self.vehicle or not self.home_location or not SAFETY_SETTINGS["geofence_enabled"]:
            return True
            
        current_location = self.vehicle.location.global_frame
        if not current_location:
            return True
            
        # Вычисляем расстояние до домашней точки
        distance = self._calculate_distance(
            current_location.lat, current_location.lon,
            self.home_location.lat, self.home_location.lon
        )
        
        max_distance = SAFETY_SETTINGS["max_distance"]
        max_altitude = SAFETY_SETTINGS["max_altitude"]
        
        if distance > max_distance:
            self.logger.warning(f"Дрон вышел за границы геозоны: {distance}м > {max_distance}м")
            self.return_to_home()
            return False
            
        if current_location.alt > max_altitude:
            self.logger.warning(f"Дрон превысил максимальную высоту: {current_location.alt}м > {max_altitude}м")
            self.return_to_home()
            return False
            
        return True
    
    def _wait_for_altitude(self, target_altitude: float, accuracy: float = 0.5, timeout: float = 60) -> bool:
        """
        Ожидает достижения заданной высоты.
        
        Args:
            target_altitude: Целевая высота в метрах
            accuracy: Точность достижения высоты в метрах
            timeout: Тайм-аут в секундах
            
        Returns:
            bool: True если высота достигнута, False если произошел тайм-аут
        """
        if not self.vehicle:
            return False
            
        self.logger.info(f"Ожидаем достижения высоты {target_altitude} м")
        
        start_time = time.time()
        while True:
            current_altitude = self.vehicle.location.global_relative_frame.alt
            
            # Выводим текущую высоту каждые 2 секунды
            if int(time.time()) % 2 == 0:
                self.logger.info(f"Текущая высота: {current_altitude:.1f} м")
                
            # Проверяем достижение целевой высоты
            if abs(current_altitude - target_altitude) <= accuracy:
                self.logger.info(f"Достигнута целевая высота: {current_altitude:.1f} м")
                return True
                
            # Проверяем тайм-аут
            if time.time() - start_time > timeout:
                self.logger.warning(f"Тайм-аут при ожидании высоты {target_altitude} м")
                return False
                
            time.sleep(0.5)
    
    def _wait_for_home_location(self) -> bool:
        """
        Ожидает получения домашней локации от дрона.
        
        Returns:
            bool: True если домашняя локация получена, иначе False
        """
        if not self.vehicle:
            return False
            
        self.logger.info("Ожидаем получения домашней локации")
        
        # Если домашняя локация уже доступна
        if self.vehicle.home_location:
            self.home_location = self.vehicle.home_location
            self.logger.info(f"Домашняя локация: lat={self.home_location.lat}, lon={self.home_location.lon}, alt={self.home_location.alt}")
            return True
            
        # Если нет, запрашиваем её
        self.vehicle.commands.download()
        self.vehicle.commands.wait_ready()
        
        if self.vehicle.home_location:
            self.home_location = self.vehicle.home_location
            self.logger.info(f"Домашняя локация: lat={self.home_location.lat}, lon={self.home_location.lon}, alt={self.home_location.alt}")
            return True
            
        # Если всё еще нет, используем текущую локацию как домашнюю
        current_location = self.vehicle.location.global_frame
        if current_location:
            self.home_location = current_location
            self.logger.info(f"Используем текущую локацию как домашнюю: lat={self.home_location.lat}, lon={self.home_location.lon}, alt={self.home_location.alt}")
            return True
            
        self.logger.warning("Не удалось получить домашнюю локацию")
        return False
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Вычисляет расстояние между двумя точками по координатам (формула гаверсинуса).
        
        Args:
            lat1, lon1: Координаты первой точки
            lat2, lon2: Координаты второй точки
            
        Returns:
            float: Расстояние в метрах
        """
        # Радиус Земли в метрах
        earth_radius = 6371000
        
        # Переводим координаты в радианы
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Разница координат
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Формула гаверсинуса
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = earth_radius * c
        
        return distance 
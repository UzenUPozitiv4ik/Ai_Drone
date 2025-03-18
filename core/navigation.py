"""
Модуль для GPS-навигации и построения маршрутов для БПЛА
"""

import time
import math
import logging
from typing import List, Tuple, Dict, Any, Optional
from dronekit import Command, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil
import sys
import os

# Добавляем корневую директорию проекта в path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GPS_SETTINGS
from core.drone_controller import DroneController

class Waypoint:
    """Класс, представляющий точку маршрута"""
    
    def __init__(self, lat: float, lon: float, alt: Optional[float] = None, action: str = "GOTO"):
        """
        Инициализирует точку маршрута.
        
        Args:
            lat: Широта
            lon: Долгота
            alt: Высота (если None, будет использоваться текущая высота)
            action: Действие в точке ("GOTO", "LOITER", "TAKEOFF", "LAND", "RTL")
        """
        self.lat = lat
        self.lon = lon
        self.alt = alt if alt is not None else GPS_SETTINGS["default_altitude"]
        self.action = action
        self.reached = False
        
    def __str__(self) -> str:
        """Возвращает строковое представление точки маршрута"""
        return f"Waypoint(lat={self.lat:.6f}, lon={self.lon:.6f}, alt={self.alt}m, action={self.action})"
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует точку маршрута в словарь"""
        return {
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "action": self.action,
            "reached": self.reached
        }

class Mission:
    """Класс, представляющий миссию (маршрут) БПЛА"""
    
    def __init__(self, name: str = "New Mission"):
        """
        Инициализирует миссию.
        
        Args:
            name: Название миссии
        """
        self.name = name
        self.waypoints: List[Waypoint] = []
        self.current_waypoint_index = -1
        self.completed = False
        
    def add_waypoint(self, waypoint: Waypoint) -> None:
        """
        Добавляет точку маршрута в миссию.
        
        Args:
            waypoint: Точка маршрута
        """
        self.waypoints.append(waypoint)
        
    def add_waypoint_latlon(self, lat: float, lon: float, alt: Optional[float] = None, action: str = "GOTO") -> None:
        """
        Создает и добавляет точку маршрута по координатам.
        
        Args:
            lat: Широта
            lon: Долгота
            alt: Высота (если None, будет использоваться текущая высота)
            action: Действие в точке
        """
        self.add_waypoint(Waypoint(lat, lon, alt, action))
        
    def clear(self) -> None:
        """Очищает все точки маршрута"""
        self.waypoints = []
        self.current_waypoint_index = -1
        self.completed = False
        
    def get_current_waypoint(self) -> Optional[Waypoint]:
        """
        Возвращает текущую точку маршрута.
        
        Returns:
            Waypoint: Текущая точка маршрута или None, если нет текущей точки
        """
        if self.current_waypoint_index < 0 or self.current_waypoint_index >= len(self.waypoints):
            return None
        return self.waypoints[self.current_waypoint_index]
        
    def get_next_waypoint(self) -> Optional[Waypoint]:
        """
        Возвращает следующую точку маршрута.
        
        Returns:
            Waypoint: Следующая точка маршрута или None, если нет следующей точки
        """
        next_index = self.current_waypoint_index + 1
        if next_index >= len(self.waypoints):
            return None
        return self.waypoints[next_index]
        
    def mark_current_waypoint_reached(self) -> None:
        """Отмечает текущую точку маршрута как достигнутую"""
        if self.current_waypoint_index >= 0 and self.current_waypoint_index < len(self.waypoints):
            self.waypoints[self.current_waypoint_index].reached = True
            
    def advance_to_next_waypoint(self) -> Optional[Waypoint]:
        """
        Переходит к следующей точке маршрута.
        
        Returns:
            Waypoint: Следующая точка маршрута или None, если больше нет точек
        """
        self.mark_current_waypoint_reached()
        self.current_waypoint_index += 1
        
        if self.current_waypoint_index >= len(self.waypoints):
            self.completed = True
            self.current_waypoint_index = len(self.waypoints) - 1 if len(self.waypoints) > 0 else -1
            return None
            
        return self.get_current_waypoint()
        
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует миссию в словарь"""
        return {
            "name": self.name,
            "waypoints": [wp.to_dict() for wp in self.waypoints],
            "current_waypoint_index": self.current_waypoint_index,
            "completed": self.completed
        }

class NavigationController:
    """
    Контроллер навигации для БПЛА.
    Отвечает за планирование и выполнение маршрутов, отслеживание текущего местоположения.
    """
    
    def __init__(self, drone_controller: DroneController):
        """
        Инициализирует контроллер навигации.
        
        Args:
            drone_controller: Контроллер БПЛА
        """
        self.logger = logging.getLogger(__name__)
        self.drone = drone_controller
        self.current_mission = Mission()
        self.mission_running = False
        self.waypoint_radius = GPS_SETTINGS["waypoint_radius"]
        self.home_radius = GPS_SETTINGS["home_radius"]
        self.default_altitude = GPS_SETTINGS["default_altitude"]
        self.speed = GPS_SETTINGS["default_speed"]
        
    def set_speed(self, speed: float) -> None:
        """
        Устанавливает скорость движения БПЛА.
        
        Args:
            speed: Скорость в м/с
        """
        self.speed = speed
        
        # Если дрон подключен, устанавливаем скорость напрямую
        if self.drone.vehicle:
            self.drone.vehicle.airspeed = speed
            self.drone.vehicle.groundspeed = speed
            self.logger.info(f"Установлена скорость движения: {speed} м/с")
    
    def create_mission(self, name: str = "New Mission") -> Mission:
        """
        Создает новую миссию.
        
        Args:
            name: Название миссии
            
        Returns:
            Mission: Созданная миссия
        """
        self.current_mission = Mission(name)
        self.logger.info(f"Создана новая миссия: {name}")
        return self.current_mission
        
    def load_mission(self, mission: Mission) -> None:
        """
        Загружает существующую миссию.
        
        Args:
            mission: Миссия для загрузки
        """
        self.current_mission = mission
        self.logger.info(f"Загружена миссия: {mission.name} с {len(mission.waypoints)} точками")
        
    def upload_mission_to_drone(self) -> bool:
        """
        Загружает текущую миссию в БПЛА для автономного выполнения в режиме AUTO.
        
        Returns:
            bool: True если миссия успешно загружена
        """
        if not self.drone.vehicle:
            self.logger.error("Ошибка загрузки миссии: нет подключения к БПЛА")
            return False
            
        if len(self.current_mission.waypoints) == 0:
            self.logger.warning("Невозможно загрузить пустую миссию")
            return False
            
        try:
            self.logger.info(f"Загрузка миссии '{self.current_mission.name}' в БПЛА")
            
            # Очищаем текущие команды в дроне
            cmds = self.drone.vehicle.commands
            cmds.clear()
            
            # Добавляем команду взлета, если первая точка не является взлетом
            if self.current_mission.waypoints[0].action != "TAKEOFF":
                takeoff_alt = self.default_altitude
                self.logger.info(f"Добавление команды взлета на высоту {takeoff_alt} м")
                cmds.add(Command(
                    0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, 
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 
                    0, 0, 0, 0, 0, 0, 0, 0, takeoff_alt
                ))
            
            # Добавляем каждую точку маршрута
            for idx, wp in enumerate(self.current_mission.waypoints):
                self.logger.info(f"Добавление точки {idx+1}: {wp}")
                
                # Определяем тип команды по действию
                if wp.action == "TAKEOFF":
                    cmd_id = mavutil.mavlink.MAV_CMD_NAV_TAKEOFF
                elif wp.action == "LAND":
                    cmd_id = mavutil.mavlink.MAV_CMD_NAV_LAND
                elif wp.action == "RTL":
                    cmd_id = mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH
                elif wp.action == "LOITER":
                    cmd_id = mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME
                    loiter_time = 10  # 10 секунд по умолчанию
                else:  # GOTO по умолчанию
                    cmd_id = mavutil.mavlink.MAV_CMD_NAV_WAYPOINT
                
                # Добавляем команду
                cmds.add(Command(
                    0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    cmd_id, 
                    0, 0, 0, 0, 0, 0, 
                    wp.lat, wp.lon, wp.alt
                ))
            
            # Загружаем команды в дрон
            cmds.upload()
            self.logger.info(f"Миссия успешно загружена в БПЛА ({len(self.current_mission.waypoints)} точек)")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке миссии в БПЛА: {str(e)}")
            return False
            
    def start_mission_auto(self) -> bool:
        """
        Запускает выполнение миссии в автоматическом режиме (AUTO).
        Для этого миссия должна быть предварительно загружена через upload_mission_to_drone().
        
        Returns:
            bool: True если миссия успешно запущена
        """
        if not self.drone.vehicle:
            self.logger.error("Ошибка запуска миссии: нет подключения к БПЛА")
            return False
            
        if len(self.current_mission.waypoints) == 0:
            self.logger.warning("Невозможно запустить пустую миссию")
            return False
            
        try:
            # Если дрон не активирован, активируем его
            if not self.drone.vehicle.armed:
                self.logger.info("БПЛА не активирован, выполняем активацию")
                if not self.drone.arm():
                    return False
            
            # Устанавливаем режим AUTO для выполнения миссии
            if not self.drone.set_mode("AUTO"):
                self.logger.error("Не удалось установить режим AUTO")
                return False
                
            self.mission_running = True
            self.current_mission.current_waypoint_index = 0
            self.logger.info(f"Миссия '{self.current_mission.name}' запущена в режиме AUTO")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при запуске миссии: {str(e)}")
            return False
            
    def start_mission_guided(self) -> bool:
        """
        Запускает выполнение миссии в режиме GUIDED, где каждая точка маршрута
        обрабатывается программно, а не автопилотом.
        
        Returns:
            bool: True если миссия успешно запущена
        """
        if not self.drone.vehicle:
            self.logger.error("Ошибка запуска миссии: нет подключения к БПЛА")
            return False
            
        if len(self.current_mission.waypoints) == 0:
            self.logger.warning("Невозможно запустить пустую миссию")
            return False
            
        try:
            # Если дрон не активирован, активируем его
            if not self.drone.vehicle.armed:
                self.logger.info("БПЛА не активирован, выполняем активацию")
                if not self.drone.arm():
                    return False
            
            # Устанавливаем режим GUIDED
            if not self.drone.set_mode("GUIDED"):
                self.logger.error("Не удалось установить режим GUIDED")
                return False
                
            # Начинаем с первой точки
            self.mission_running = True
            self.current_mission.current_waypoint_index = 0
            
            # Если первая точка - взлет, выполняем взлет
            current_wp = self.current_mission.get_current_waypoint()
            if current_wp.action == "TAKEOFF":
                self.logger.info(f"Выполняем взлет на высоту {current_wp.alt} м")
                if not self.drone.takeoff(current_wp.alt):
                    self.logger.error("Ошибка при взлете")
                    self.mission_running = False
                    return False
                self.current_mission.mark_current_waypoint_reached()
                self.current_mission.advance_to_next_waypoint()
            
            # Запускаем первую точку маршрута
            self.navigate_to_next_waypoint()
            
            self.logger.info(f"Миссия '{self.current_mission.name}' запущена в режиме GUIDED")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при запуске миссии: {str(e)}")
            self.mission_running = False
            return False
    
    def stop_mission(self) -> bool:
        """
        Останавливает выполнение текущей миссии.
        
        Returns:
            bool: True если миссия успешно остановлена
        """
        if not self.mission_running:
            self.logger.info("Миссия не выполняется, нечего останавливать")
            return True
            
        try:
            # Устанавливаем режим LOITER для зависания на месте
            if not self.drone.set_mode("LOITER"):
                self.logger.warning("Не удалось установить режим LOITER, пробуем GUIDED")
                self.drone.set_mode("GUIDED")
                
            self.mission_running = False
            self.logger.info("Миссия остановлена")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при остановке миссии: {str(e)}")
            return False
    
    def navigate_to_next_waypoint(self) -> bool:
        """
        Отправляет БПЛА к следующей точке маршрута в режиме GUIDED.
        
        Returns:
            bool: True если навигация успешно начата
        """
        if not self.drone.vehicle or not self.mission_running:
            return False
            
        # Получаем текущую точку маршрута
        wp = self.current_mission.get_current_waypoint()
        if not wp:
            if self.current_mission.completed:
                self.logger.info("Миссия завершена, все точки пройдены")
                self.mission_running = False
                return True
            else:
                self.logger.error("Ошибка: нет текущей точки маршрута")
                return False
        
        # Обрабатываем различные типы действий
        if wp.action == "LAND":
            self.logger.info(f"Выполняем посадку в точке {wp}")
            return self.drone.land()
            
        elif wp.action == "RTL":
            self.logger.info("Выполняем возврат на базу")
            return self.drone.return_to_home()
            
        elif wp.action == "LOITER":
            self.logger.info(f"Зависание в точке {wp}")
            self.drone.goto(wp.lat, wp.lon, wp.alt)
            # Здесь можно добавить задержку для зависания
            return True
            
        else:  # GOTO по умолчанию
            self.logger.info(f"Навигация к точке {wp}")
            return self.drone.goto(wp.lat, wp.lon, wp.alt)
    
    def update_mission_progress(self) -> None:
        """
        Обновляет прогресс выполнения миссии, проверяя достижение текущей точки маршрута.
        Должна вызываться периодически для отслеживания прогресса миссии в режиме GUIDED.
        """
        if not self.drone.vehicle or not self.mission_running:
            return
            
        # Получаем текущую точку маршрута
        wp = self.current_mission.get_current_waypoint()
        if not wp:
            if not self.current_mission.completed:
                self.logger.info("Миссия завершена, все точки пройдены")
                self.current_mission.completed = True
                self.mission_running = False
            return
        
        # Получаем текущую позицию дрона
        current_location = self.drone.vehicle.location.global_relative_frame
        if not current_location:
            return
            
        # Вычисляем расстояние до текущей точки
        distance = self._calculate_distance(
            current_location.lat, current_location.lon,
            wp.lat, wp.lon
        )
        
        # Проверяем, достигли ли мы текущей точки
        if distance <= self.waypoint_radius:
            self.logger.info(f"Достигнута точка маршрута {self.current_mission.current_waypoint_index+1}")
            
            # Если это точка с действием LAND или RTL, завершаем миссию
            if wp.action in ["LAND", "RTL"]:
                self.logger.info(f"Выполняем действие {wp.action} и завершаем миссию")
                self.current_mission.mark_current_waypoint_reached()
                self.current_mission.completed = True
                self.mission_running = False
                return
            
            # Переходим к следующей точке
            next_wp = self.current_mission.advance_to_next_waypoint()
            if next_wp:
                self.logger.info(f"Переход к следующей точке: {next_wp}")
                self.navigate_to_next_waypoint()
            else:
                self.logger.info("Миссия завершена, все точки пройдены")
                self.mission_running = False
    
    def calculate_path(self, start_lat: float, start_lon: float, 
                       end_lat: float, end_lon: float, 
                       waypoints_count: int = 10) -> List[Tuple[float, float]]:
        """
        Вычисляет промежуточные точки на пути между start и end.
        
        Args:
            start_lat, start_lon: Координаты начальной точки
            end_lat, end_lon: Координаты конечной точки
            waypoints_count: Количество промежуточных точек
            
        Returns:
            List[Tuple[float, float]]: Список координат промежуточных точек
        """
        points = []
        
        for i in range(waypoints_count + 1):
            # Линейная интерполяция
            fraction = i / waypoints_count
            lat = start_lat + fraction * (end_lat - start_lat)
            lon = start_lon + fraction * (end_lon - start_lon)
            points.append((lat, lon))
            
        return points
    
    def is_at_home(self) -> bool:
        """
        Проверяет, находится ли БПЛА в зоне домашней точки.
        
        Returns:
            bool: True если БПЛА находится в зоне домашней точки
        """
        if not self.drone.vehicle or not self.drone.home_location:
            return False
            
        current_location = self.drone.vehicle.location.global_frame
        if not current_location:
            return False
            
        # Вычисляем расстояние до домашней точки
        distance = self._calculate_distance(
            current_location.lat, current_location.lon,
            self.drone.home_location.lat, self.drone.home_location.lon
        )
        
        return distance <= self.home_radius
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Вычисляет расстояние между двумя точками по координатам (формула гаверсинуса).
        
        Args:
            lat1, lon1: Координаты первой точки
            lat2, lon2: Координаты второй точки
            
        Returns:
            float: Расстояние в метрах
        """
        # Делегируем вычисление расстояния контроллеру дрона
        return self.drone._calculate_distance(lat1, lon1, lat2, lon2) 
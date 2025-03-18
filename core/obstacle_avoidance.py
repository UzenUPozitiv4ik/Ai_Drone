"""
Модуль для обнаружения и обхода препятствий
"""

import time
import math
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dronekit import Vehicle, LocationGlobalRelative
import sys
import os

# Добавляем корневую директорию проекта в path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OBSTACLE_AVOIDANCE, GPS_SETTINGS
from core.drone_controller import DroneController
from core.navigation import NavigationController

class ObstacleDetector:
    """
    Класс для обнаружения препятствий с помощью датчиков дрона
    """
    
    def __init__(self, drone_controller: DroneController):
        """
        Инициализирует детектор препятствий.
        
        Args:
            drone_controller: Контроллер БПЛА
        """
        self.logger = logging.getLogger(__name__)
        self.drone = drone_controller
        
        # Настройки обнаружения препятствий
        self.min_distance = OBSTACLE_AVOIDANCE["min_distance"]
        self.safe_distance = OBSTACLE_AVOIDANCE["safe_distance"]
        self.scan_angle = OBSTACLE_AVOIDANCE["scan_angle"]
        
        # Последние измерения расстояний в разных направлениях
        # Формат: {угол_в_градусах: (расстояние_в_метрах, временная_метка)}
        self.distance_measurements = {}
        
        # Обнаруженные препятствия
        # Формат: [(x, y, z, радиус), ...]
        self.detected_obstacles = []
        
    def scan_surroundings(self) -> Dict[int, float]:
        """
        Выполняет сканирование окружения для обнаружения препятствий.
        В реальной системе этот метод будет получать данные с датчиков,
        таких как лидары, сонары или камеры.
        
        Returns:
            Dict[int, float]: Словарь {угол: расстояние} с результатами сканирования
        """
        # Это заглушка - в реальной системе здесь будет код для работы с датчиками
        # В нашей имитации предполагаем, что препятствий нет
        
        self.logger.debug("Сканирование окружения для обнаружения препятствий")
        
        scan_results = {}
        
        # Имитация данных от датчиков для демонстрации
        # В реальном дроне здесь был бы код для чтения данных с сенсоров
        for angle in range(0, 360, 15):
            # Генерируем случайное расстояние от 1 до 20 метров
            # В реальном дроне здесь были бы фактические измерения датчиков
            if angle % 45 == 0:  # Искусственно создаем "препятствие" в направлениях, кратных 45 градусам
                distance = np.random.uniform(2, 5)
            else:
                distance = np.random.uniform(10, 20)
                
            scan_results[angle] = distance
            self.distance_measurements[angle] = (distance, time.time())
            
        self.logger.debug(f"Результаты сканирования: {scan_results}")
        return scan_results
        
    def is_obstacle_in_path(self, forward_distance: float = 10.0) -> bool:
        """
        Проверяет наличие препятствий на пути следования дрона.
        
        Args:
            forward_distance: Расстояние проверки впереди дрона в метрах
            
        Returns:
            bool: True если на пути обнаружено препятствие
        """
        # Выполняем сканирование
        scan_results = self.scan_surroundings()
        
        # Проверяем наличие препятствий в направлении полета
        # Для простоты считаем, что препятствие в направлении от -30 до +30 градусов
        # относительно текущего курса дрона будет считаться препятствием на пути
        
        if not self.drone.vehicle:
            return False
            
        # Получаем текущий курс дрона
        heading = self.drone.vehicle.heading
        
        # Проверяем наличие препятствий в секторе перед дроном
        obstacle_detected = False
        min_forward_distance = float('inf')
        
        for angle, distance in scan_results.items():
            # Вычисляем относительный угол к направлению движения
            relative_angle = (angle - heading) % 360
            if relative_angle > 180:
                relative_angle -= 360
                
            # Проверяем, находится ли измерение в секторе перед дроном
            if abs(relative_angle) <= 30:
                if distance < forward_distance and distance < self.min_distance:
                    obstacle_detected = True
                    if distance < min_forward_distance:
                        min_forward_distance = distance
        
        if obstacle_detected:
            self.logger.warning(f"Обнаружено препятствие впереди на расстоянии {min_forward_distance:.2f} м")
        else:
            self.logger.debug("Препятствий на пути не обнаружено")
            
        return obstacle_detected
        
    def get_closest_obstacle_direction(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Определяет направление и расстояние до ближайшего препятствия.
        
        Returns:
            Tuple[Optional[float], Optional[float]]: (направление в градусах, расстояние в метрах)
            или (None, None), если препятствий не обнаружено
        """
        # Выполняем сканирование
        scan_results = self.scan_surroundings()
        
        # Находим ближайшее препятствие
        min_distance = float('inf')
        closest_direction = None
        
        for angle, distance in scan_results.items():
            if distance < min_distance and distance < self.safe_distance:
                min_distance = distance
                closest_direction = angle
                
        if closest_direction is not None:
            self.logger.info(f"Ближайшее препятствие: направление {closest_direction}°, расстояние {min_distance:.2f} м")
            return (closest_direction, min_distance)
        else:
            return (None, None)
            
    def update_obstacle_map(self) -> None:
        """
        Обновляет карту препятствий на основе новых измерений.
        В сложной реализации здесь может быть алгоритм SLAM или другой метод
        построения и обновления карты окружения.
        """
        # Выполняем сканирование
        scan_results = self.scan_surroundings()
        
        # Получаем текущую позицию дрона
        if not self.drone.vehicle:
            return
            
        current_location = self.drone.vehicle.location.global_relative_frame
        if not current_location:
            return
            
        # Получаем текущий курс дрона
        heading = self.drone.vehicle.heading
        
        # Обновляем карту препятствий
        # В простой реализации просто сохраняем обнаруженные препятствия как точки в пространстве
        
        # Очищаем устаревшие данные (старше 30 секунд)
        current_time = time.time()
        self.detected_obstacles = [
            obstacle for obstacle in self.detected_obstacles 
            if obstacle[4] > current_time - 30
        ]
        
        # Добавляем новые измерения на карту препятствий
        for angle, distance in scan_results.items():
            if distance < self.safe_distance:
                # Вычисляем абсолютное направление
                absolute_angle = (heading + angle) % 360
                
                # Конвертируем полярные координаты (угол, расстояние) в декартовы (x, y)
                angle_rad = math.radians(absolute_angle)
                x = distance * math.sin(angle_rad)
                y = distance * math.cos(angle_rad)
                
                # Добавляем препятствие на карту
                # Формат: (x, y, z, радиус, временная_метка)
                # где x, y - координаты относительно текущего положения дрона,
                # z - высота (предполагаем, что все препятствия на уровне дрона),
                # радиус - размер препятствия (предполагаем 1 метр)
                self.detected_obstacles.append((x, y, 0, 1.0, current_time))
                
        self.logger.debug(f"Карта препятствий обновлена, {len(self.detected_obstacles)} препятствий")

class ObstacleAvoidance:
    """
    Класс для реализации стратегий обхода препятствий
    """
    
    def __init__(self, drone_controller: DroneController, navigation_controller: NavigationController):
        """
        Инициализирует систему обхода препятствий.
        
        Args:
            drone_controller: Контроллер БПЛА
            navigation_controller: Контроллер навигации
        """
        self.logger = logging.getLogger(__name__)
        self.drone = drone_controller
        self.navigation = navigation_controller
        self.obstacle_detector = ObstacleDetector(drone_controller)
        
        # Настройки обхода препятствий
        self.enabled = OBSTACLE_AVOIDANCE["enabled"]
        self.min_distance = OBSTACLE_AVOIDANCE["min_distance"]
        self.safe_distance = OBSTACLE_AVOIDANCE["safe_distance"]
        self.strategy = OBSTACLE_AVOIDANCE["avoidance_strategy"]
        
        # Состояния обхода препятствий
        self.avoiding = False
        self.original_target = None
        self.avoidance_waypoints = []
        self.current_avoidance_waypoint = 0
        
    def enable(self) -> None:
        """Включает систему обхода препятствий"""
        self.enabled = True
        self.logger.info("Система обхода препятствий включена")
        
    def disable(self) -> None:
        """Выключает систему обхода препятствий"""
        self.enabled = False
        self.logger.info("Система обхода препятствий выключена")
        
    def set_strategy(self, strategy: str) -> None:
        """
        Устанавливает стратегию обхода препятствий.
        
        Args:
            strategy: Название стратегии ("stop_and_redirect" или "dynamic_path")
        """
        if strategy in ["stop_and_redirect", "dynamic_path"]:
            self.strategy = strategy
            self.logger.info(f"Установлена стратегия обхода препятствий: {strategy}")
        else:
            self.logger.warning(f"Неизвестная стратегия {strategy}, используем текущую: {self.strategy}")
            
    def check_path(self) -> bool:
        """
        Проверяет наличие препятствий на пути следования.
        
        Returns:
            bool: True если путь свободен, False если обнаружено препятствия
        """
        if not self.enabled:
            return True
            
        # Проверяем наличие препятствий на пути
        obstacle_in_path = self.obstacle_detector.is_obstacle_in_path()
        
        if obstacle_in_path:
            self.logger.warning("Обнаружены препятствия на пути следования")
            self.avoiding = True
            return False
        else:
            if self.avoiding:
                self.logger.info("Путь свободен, возобновляем нормальное движение")
                self.avoiding = False
            return True
            
    def avoid_obstacle(self) -> bool:
        """
        Выполняет маневр обхода препятствия.
        
        Returns:
            bool: True если маневр успешно выполняется
        """
        if not self.enabled or not self.avoiding:
            return False
            
        if not self.drone.vehicle:
            self.logger.error("Ошибка обхода препятствия: нет подключения к БПЛА")
            return False
            
        # Получаем направление и расстояние до ближайшего препятствия
        direction, distance = self.obstacle_detector.get_closest_obstacle_direction()
        
        if direction is None or distance is None:
            self.logger.info("Препятствий не обнаружено, возобновляем нормальное движение")
            self.avoiding = False
            return False
            
        # Выбираем стратегию обхода
        if self.strategy == "stop_and_redirect":
            return self._avoid_obstacle_stop_and_redirect(direction, distance)
        elif self.strategy == "dynamic_path":
            return self._avoid_obstacle_dynamic_path(direction, distance)
        else:
            self.logger.warning(f"Неизвестная стратегия {self.strategy}, используем stop_and_redirect")
            return self._avoid_obstacle_stop_and_redirect(direction, distance)
            
    def _avoid_obstacle_stop_and_redirect(self, obstacle_direction: float, obstacle_distance: float) -> bool:
        """
        Стратегия обхода препятствия "стоп и перенаправление".
        Дрон останавливается, а затем меняет направление для обхода препятствия.
        
        Args:
            obstacle_direction: Направление к препятствию в градусах
            obstacle_distance: Расстояние до препятствия в метрах
            
        Returns:
            bool: True если маневр успешно выполняется
        """
        self.logger.info(f"Выполняем обход препятствия по стратегии 'стоп и перенаправление'")
        
        # Сохраняем текущую цель, если еще не сохранили
        if not self.original_target and self.navigation.current_mission.get_current_waypoint():
            wp = self.navigation.current_mission.get_current_waypoint()
            self.original_target = (wp.lat, wp.lon, wp.alt)
            self.logger.info(f"Сохраняем оригинальную цель: lat={wp.lat}, lon={wp.lon}, alt={wp.alt}")
            
        # Получаем текущее положение и курс дрона
        current_location = self.drone.vehicle.location.global_relative_frame
        heading = self.drone.vehicle.heading
        
        # Вычисляем направление обхода
        # Выбираем направление, перпендикулярное к препятствию
        # Если препятствие справа, идем влево, и наоборот
        
        # Вычисляем относительное направление к препятствию
        relative_direction = (obstacle_direction - heading) % 360
        if relative_direction > 180:
            relative_direction -= 360
            
        # Если препятствие справа (0..180), поворачиваем влево (-90)
        # Если препятствие слева (-180..0), поворачиваем вправо (90)
        if -90 <= relative_direction <= 90:
            # Препятствие впереди или справа, идем влево
            avoidance_heading = (heading - 90) % 360
        else:
            # Препятствие слева, идем вправо
            avoidance_heading = (heading + 90) % 360
            
        # Вычисляем расстояние маневра - чем ближе препятствие, тем больше расстояние обхода
        avoidance_distance = max(self.safe_distance * 2, self.safe_distance + (self.safe_distance - obstacle_distance))
        
        # Вычисляем новые координаты для обхода
        avoidance_heading_rad = math.radians(avoidance_heading)
        
        # Вычисляем смещение в метрах
        dx = avoidance_distance * math.sin(avoidance_heading_rad)
        dy = avoidance_distance * math.cos(avoidance_heading_rad)
        
        # Конвертируем смещение в координаты
        # Приближенно: 1 градус широты = 111 км, 1 градус долготы = 111 км * cos(широта)
        lat_offset = dy / 111000  # метры -> градусы широты
        lon_offset = dx / (111000 * math.cos(math.radians(current_location.lat)))  # метры -> градусы долготы
        
        # Вычисляем новые координаты
        new_lat = current_location.lat + lat_offset
        new_lon = current_location.lon + lon_offset
        
        # Сохраняем текущую высоту
        new_alt = current_location.alt
        
        # Отправляем дрон к точке обхода
        self.logger.info(f"Обход препятствия: направляемся к lat={new_lat}, lon={new_lon}, alt={new_alt}")
        self.drone.goto(new_lat, new_lon, new_alt)
        
        return True
        
    def _avoid_obstacle_dynamic_path(self, obstacle_direction: float, obstacle_distance: float) -> bool:
        """
        Стратегия обхода препятствия "динамический путь".
        Дрон вычисляет новый путь в обход препятствия и следует по нему.
        
        Args:
            obstacle_direction: Направление к препятствию в градусах
            obstacle_distance: Расстояние до препятствия в метрах
            
        Returns:
            bool: True если маневр успешно выполняется
        """
        self.logger.info(f"Выполняем обход препятствия по стратегии 'динамический путь'")
        
        # Если у нас еще нет сохраненной исходной цели и нет точек обхода
        if not self.original_target and not self.avoidance_waypoints:
            # Получаем текущую цель из миссии
            if self.navigation.current_mission.get_current_waypoint():
                wp = self.navigation.current_mission.get_current_waypoint()
                self.original_target = (wp.lat, wp.lon, wp.alt)
                self.logger.info(f"Сохраняем оригинальную цель: lat={wp.lat}, lon={wp.lon}, alt={wp.alt}")
                
                # Вычисляем путь обхода
                self._calculate_avoidance_path(obstacle_direction, obstacle_distance)
            else:
                self.logger.error("Нет текущей точки миссии для обхода")
                return False
                
        # Если у нас уже есть точки обхода
        if self.avoidance_waypoints:
            # Если мы еще не начали обход или прошли все точки обхода
            if self.current_avoidance_waypoint >= len(self.avoidance_waypoints):
                # Возвращаемся к исходной цели
                if self.original_target:
                    target_lat, target_lon, target_alt = self.original_target
                    self.logger.info(f"Возвращаемся к исходной цели: lat={target_lat}, lon={target_lon}, alt={target_alt}")
                    self.drone.goto(target_lat, target_lon, target_alt)
                    
                    # Сбрасываем состояние обхода
                    self.avoiding = False
                    self.original_target = None
                    self.avoidance_waypoints = []
                    self.current_avoidance_waypoint = 0
                    
                    return True
            else:
                # Берем следующую точку обхода
                next_lat, next_lon, next_alt = self.avoidance_waypoints[self.current_avoidance_waypoint]
                self.logger.info(f"Следующая точка обхода {self.current_avoidance_waypoint+1}/{len(self.avoidance_waypoints)}: "
                               f"lat={next_lat}, lon={next_lon}, alt={next_alt}")
                self.drone.goto(next_lat, next_lon, next_alt)
                
                # Проверяем, достигли ли мы текущей точки обхода
                if self._reached_waypoint(next_lat, next_lon):
                    self.logger.info(f"Достигнута точка обхода {self.current_avoidance_waypoint+1}")
                    self.current_avoidance_waypoint += 1
                    
                return True
        
        # Если мы здесь, значит что-то пошло не так
        self.logger.warning("Ошибка в алгоритме обхода препятствия")
        return False
        
    def _calculate_avoidance_path(self, obstacle_direction: float, obstacle_distance: float) -> None:
        """
        Вычисляет путь обхода препятствия.
        
        Args:
            obstacle_direction: Направление к препятствию в градусах
            obstacle_distance: Расстояние до препятствия в метрах
        """
        if not self.drone.vehicle or not self.original_target:
            return
            
        self.logger.info("Вычисляем путь обхода препятствия")
        
        # Получаем текущее положение дрона
        current_location = self.drone.vehicle.location.global_relative_frame
        
        # Получаем целевую точку
        target_lat, target_lon, target_alt = self.original_target
        
        # Вычисляем направление к цели
        dx = target_lon - current_location.lon
        dy = target_lat - current_location.lat
        target_direction = math.degrees(math.atan2(dx, dy)) % 360
        
        # Определяем с какой стороны обходить препятствие
        # Вычисляем относительное направление к препятствию
        relative_direction = (obstacle_direction - target_direction) % 360
        if relative_direction > 180:
            relative_direction -= 360
            
        # Если препятствие справа от направления к цели, обходим слева, и наоборот
        if relative_direction >= 0:
            # Препятствие справа, обходим слева
            avoidance_heading = (obstacle_direction - 90) % 360
        else:
            # Препятствие слева, обходим справа
            avoidance_heading = (obstacle_direction + 90) % 360
            
        # Вычисляем расстояние обхода
        avoidance_distance = max(self.safe_distance * 2, self.safe_distance + (self.safe_distance - obstacle_distance))
        
        # Вычисляем промежуточные точки обхода
        self.avoidance_waypoints = []
        
        # Первая точка - отойти от препятствия перпендикулярно
        avoidance_heading_rad = math.radians(avoidance_heading)
        dx1 = avoidance_distance * math.sin(avoidance_heading_rad)
        dy1 = avoidance_distance * math.cos(avoidance_heading_rad)
        
        # Конвертируем смещение в координаты
        lat_offset1 = dy1 / 111000  # метры -> градусы широты
        lon_offset1 = dx1 / (111000 * math.cos(math.radians(current_location.lat)))  # метры -> градусы долготы
        
        wp1_lat = current_location.lat + lat_offset1
        wp1_lon = current_location.lon + lon_offset1
        wp1_alt = current_location.alt
        
        self.avoidance_waypoints.append((wp1_lat, wp1_lon, wp1_alt))
        
        # Вторая точка - двигаться параллельно исходному направлению
        # Вычисляем вектор от текущей позиции к цели
        target_vector_lat = target_lat - current_location.lat
        target_vector_lon = target_lon - current_location.lon
        
        # Добавляем этот вектор к первой точке обхода
        wp2_lat = wp1_lat + target_vector_lat
        wp2_lon = wp1_lon + target_vector_lon
        wp2_alt = wp1_alt
        
        self.avoidance_waypoints.append((wp2_lat, wp2_lon, wp2_alt))
        
        # Третья точка - вернуться к исходной цели
        self.avoidance_waypoints.append((target_lat, target_lon, target_alt))
        
        self.logger.info(f"Вычислен путь обхода из {len(self.avoidance_waypoints)} точек")
        self.current_avoidance_waypoint = 0
        
    def _reached_waypoint(self, lat: float, lon: float, threshold: float = None) -> bool:
        """
        Проверяет, достиг ли дрон указанной точки.
        
        Args:
            lat: Широта точки
            lon: Долгота точки
            threshold: Порог достижения в метрах. Если None, используется waypoint_radius из настроек.
            
        Returns:
            bool: True если дрон достиг точки
        """
        if not self.drone.vehicle:
            return False
            
        # Получаем текущее положение дрона
        current_location = self.drone.vehicle.location.global_relative_frame
        if not current_location:
            return False
            
        # Вычисляем расстояние до точки
        distance = self.drone._calculate_distance(
            current_location.lat, current_location.lon,
            lat, lon
        )
        
        # Используем порог из параметра или из настроек навигации
        if threshold is None:
            threshold = self.navigation.waypoint_radius
            
        return distance <= threshold
        
    def update(self) -> None:
        """
        Обновляет состояние системы обхода препятствий.
        Должна вызываться периодически для отслеживания препятствий и управления обходом.
        """
        if not self.enabled:
            return
            
        # Обновляем данные о препятствиях
        self.obstacle_detector.update_obstacle_map()
        
        # Если мы не в процессе обхода, проверяем наличие препятствий на пути
        if not self.avoiding:
            if not self.check_path():
                # Если есть препятствие и мы еще не начали обход, начинаем обход
                self.avoid_obstacle()
        else:
            # Если мы уже в процессе обхода, продолжаем обход
            self.avoid_obstacle() 
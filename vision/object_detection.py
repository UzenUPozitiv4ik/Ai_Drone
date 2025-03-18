"""
Модуль для обнаружения объектов на изображениях с использованием компьютерного зрения
"""

import cv2
import numpy as np
import torch
import logging
import os
import sys
import time
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

# Добавляем корневую директорию проекта в path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VISION_SETTINGS
from vision.camera import Camera

class ObjectDetector:
    """
    Класс для обнаружения объектов на изображениях с камеры.
    Использует предварительно обученные модели глубокого обучения.
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: Optional[float] = None):
        """
        Инициализирует детектор объектов.
        
        Args:
            model_path: Путь к файлу модели (если None, берется из настроек)
            confidence_threshold: Порог уверенности для обнаружения (если None, берется из настроек)
        """
        self.logger = logging.getLogger(__name__)
        
        # Загружаем настройки из конфигурации
        self.model_path = model_path or VISION_SETTINGS["object_detection"]["model_path"]
        self.confidence_threshold = confidence_threshold or VISION_SETTINGS["object_detection"]["confidence_threshold"]
        self.classes_of_interest = VISION_SETTINGS["object_detection"]["classes_of_interest"]
        self.enabled = VISION_SETTINGS["object_detection"]["enabled"]
        
        # Модель для обнаружения объектов
        self.model = None
        self.device = None
        self.class_names = []
        
        # Статистика обнаружения
        self.detection_count = 0
        self.last_detections = []
        self.processing_time = 0
        
        # Инициализируем модель, если обнаружение объектов включено
        if self.enabled:
            self._initialize_model()
            
    def _initialize_model(self) -> bool:
        """
        Инициализирует модель глубокого обучения для обнаружения объектов.
        
        Returns:
            bool: True если модель успешно инициализирована
        """
        try:
            self.logger.info(f"Инициализация модели обнаружения объектов из {self.model_path}")
            
            # Определяем устройство для вычислений
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Используется устройство: {self.device}")
            
            # Проверяем наличие файла модели
            if not os.path.exists(self.model_path):
                # Если файл не существует, создаем каталог и скачиваем предобученную модель из библиотеки torchvision
                self.logger.warning(f"Файл модели не найден: {self.model_path}. Загружаем предобученную модель YOLOv5.")
                
                # Создаем каталог для моделей, если он не существует
                model_dir = os.path.dirname(self.model_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir, exist_ok=True)
                
                # Загружаем предобученную модель YOLOv5 из PyTorch Hub
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                
                # Сохраняем модель для будущего использования
                torch.save(self.model.state_dict(), self.model_path)
                self.logger.info(f"Модель YOLOv5 сохранена в {self.model_path}")
            else:
                # Загружаем модель из файла
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
            
            # Переносим модель на указанное устройство
            self.model.to(self.device)
            
            # Устанавливаем режим вывода
            self.model.eval()
            
            # Получаем имена классов
            self.class_names = self.model.names
            
            self.logger.info(f"Модель обнаружения объектов успешно инициализирована. Доступные классы: {self.class_names}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации модели обнаружения объектов: {str(e)}")
            return False
            
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Обнаруживает объекты на изображении.
        
        Args:
            frame: Изображение в формате OpenCV (numpy.ndarray)
            
        Returns:
            List[Dict[str, Any]]: Список обнаруженных объектов с информацией о классе, уверенности и координатах
        """
        if not self.enabled or self.model is None:
            return []
            
        try:
            start_time = time.time()
            
            # Преобразуем формат кадра, если нужно
            # YOLOv5 ожидает тензор в формате (B, C, H, W)
            
            # Выполняем обнаружение объектов
            with torch.no_grad():
                results = self.model(frame)
            
            # Обрабатываем результаты
            detections = []
            
            # Получаем данные предсказаний
            pred = results.xyxy[0].cpu().numpy()  # Первый батч, формат: (x1, y1, x2, y2, confidence, class)
            
            # Фильтруем предсказания по порогу уверенности и классам интереса
            for *box, conf, cls_id in pred:
                class_name = self.class_names[int(cls_id)]
                
                # Проверяем, интересует ли нас этот класс
                if self.classes_of_interest and class_name not in self.classes_of_interest:
                    continue
                    
                # Проверяем порог уверенности
                if conf < self.confidence_threshold:
                    continue
                    
                x1, y1, x2, y2 = box
                detection = {
                    "class": class_name,
                    "confidence": float(conf),
                    "box": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                        "center_x": int((x1 + x2) / 2),
                        "center_y": int((y1 + y2) / 2)
                    },
                    "timestamp": time.time()
                }
                
                detections.append(detection)
            
            # Обновляем статистику
            self.detection_count += len(detections)
            self.last_detections = detections
            self.processing_time = time.time() - start_time
            
            if detections:
                self.logger.debug(f"Обнаружено {len(detections)} объектов: {', '.join([d['class'] for d in detections])}")
                
            return detections
            
        except Exception as e:
            self.logger.error(f"Ошибка при обнаружении объектов: {str(e)}")
            return []
            
    def draw_detections(self, frame: np.ndarray, detections: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        """
        Отрисовывает рамки и метки обнаруженных объектов на изображении.
        
        Args:
            frame: Исходное изображение
            detections: Список обнаруженных объектов (если None, используются последние обнаружения)
            
        Returns:
            np.ndarray: Изображение с отрисованными обнаружениями
        """
        if detections is None:
            detections = self.last_detections
            
        if not detections:
            return frame
            
        # Создаем копию кадра для рисования
        output_frame = frame.copy()
        
        # Рисуем рамки и метки для каждого обнаруженного объекта
        for det in detections:
            # Извлекаем информацию о рамке
            x1 = det["box"]["x1"]
            y1 = det["box"]["y1"]
            x2 = det["box"]["x2"]
            y2 = det["box"]["y2"]
            
            # Извлекаем класс и уверенность
            class_name = det["class"]
            confidence = det["confidence"]
            
            # Выбираем цвет в зависимости от класса (для визуального различия)
            color = self._get_color_for_class(class_name)
            
            # Рисуем рамку
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Добавляем метку с классом и уверенностью
            label = f"{class_name}: {confidence:.2f}"
            
            # Получаем размер текста
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Рисуем фон для текста
            cv2.rectangle(output_frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
            
            # Рисуем текст
            cv2.putText(output_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return output_frame
        
    def _get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """
        Возвращает цвет для отрисовки обнаруженного объекта в зависимости от его класса.
        
        Args:
            class_name: Название класса объекта
            
        Returns:
            Tuple[int, int, int]: Цвет в формате BGR
        """
        # Словарь цветов для разных классов
        color_map = {
            "person": (0, 255, 0),      # Зеленый для людей
            "car": (0, 0, 255),         # Красный для машин
            "truck": (255, 0, 0),       # Синий для грузовиков
            "bicycle": (255, 255, 0),   # Голубой для велосипедов
            "motorcycle": (255, 0, 255) # Фиолетовый для мотоциклов
        }
        
        # Возвращаем цвет из словаря или случайный цвет, если класс не найден
        return color_map.get(class_name, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику обработки изображений.
        
        Returns:
            Dict[str, Any]: Словарь со статистикой
        """
        return {
            "detection_count": self.detection_count,
            "last_detection_count": len(self.last_detections),
            "processing_time": self.processing_time,
            "fps": 1.0 / self.processing_time if self.processing_time > 0 else 0,
            "enabled": self.enabled,
            "confidence_threshold": self.confidence_threshold,
            "device": str(self.device)
        }
        
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Устанавливает новый порог уверенности для обнаружения объектов.
        
        Args:
            threshold: Новый порог уверенности (от 0.0 до 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            self.logger.info(f"Установлен новый порог уверенности: {threshold}")
        else:
            self.logger.warning(f"Некорректный порог уверенности: {threshold}. Ожидается значение от 0.0 до 1.0.")
            
    def enable(self) -> None:
        """Включает обнаружение объектов"""
        if not self.enabled:
            self.enabled = True
            
            # Инициализируем модель, если она еще не инициализирована
            if self.model is None:
                self._initialize_model()
                
            self.logger.info("Обнаружение объектов включено")
            
    def disable(self) -> None:
        """Выключает обнаружение объектов"""
        if self.enabled:
            self.enabled = False
            self.logger.info("Обнаружение объектов выключено")
            
    def set_classes_of_interest(self, classes: List[str]) -> None:
        """
        Устанавливает список классов, которые нужно обнаруживать.
        
        Args:
            classes: Список названий классов
        """
        self.classes_of_interest = classes
        self.logger.info(f"Установлены классы интереса: {classes}")
        
    def get_available_classes(self) -> List[str]:
        """
        Возвращает список всех доступных классов, которые может обнаруживать модель.
        
        Returns:
            List[str]: Список названий классов
        """
        return list(self.class_names) if self.class_names else []

class ObjectTracker:
    """
    Класс для отслеживания объектов между кадрами.
    Позволяет отслеживать перемещение объектов и определять их траектории.
    """
    
    def __init__(self, max_disappeared: int = 30, max_distance: int = 50):
        """
        Инициализирует трекер объектов.
        
        Args:
            max_disappeared: Максимальное количество кадров, в течение которых объект может отсутствовать
            max_distance: Максимальное расстояние (в пикселях) для сопоставления объектов между кадрами
        """
        self.logger = logging.getLogger(__name__)
        
        # Настройки трекера
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Словарь отслеживаемых объектов: {object_id: {"box": box, "class": class, "centroids": [centroid1, centroid2, ...], "disappeared": count}}
        self.objects = {}
        
        # Счетчик для назначения уникальных ID объектам
        self.next_object_id = 0
        
        # Статистика трекера
        self.tracked_count = 0
        self.frame_count = 0
        
    def update(self, detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Обновляет состояние трекера с новыми обнаружениями.
        
        Args:
            detections: Список обнаруженных объектов
            
        Returns:
            Dict[int, Dict[str, Any]]: Словарь отслеживаемых объектов, где ключ - ID объекта
        """
        self.frame_count += 1
        
        # Если нет обнаружений, увеличиваем счетчик исчезновений для всех объектов
        if not detections:
            for object_id in list(self.objects.keys()):
                self.objects[object_id]["disappeared"] += 1
                
                # Если объект отсутствует слишком долго, удаляем его
                if self.objects[object_id]["disappeared"] > self.max_disappeared:
                    del self.objects[object_id]
                    
            return self.objects
            
        # Если нет отслеживаемых объектов, регистрируем все обнаружения как новые объекты
        if not self.objects:
            for detection in detections:
                self._register(detection)
        else:
            # Получаем ID текущих объектов и центроиды новых обнаружений
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[object_id]["box"]["center"] for object_id in object_ids]
            
            detection_centroids = [(detection["box"]["center_x"], detection["box"]["center_y"]) for detection in detections]
            
            # Вычисляем матрицу расстояний между центроидами
            distances = np.zeros((len(object_centroids), len(detection_centroids)))
            
            for i, object_centroid in enumerate(object_centroids):
                for j, detection_centroid in enumerate(detection_centroids):
                    # Вычисляем евклидово расстояние
                    d = np.sqrt((object_centroid[0] - detection_centroid[0])**2 +
                               (object_centroid[1] - detection_centroid[1])**2)
                    distances[i, j] = d
            
            # Находим соответствия с минимальным расстоянием
            # Используем венгерский алгоритм для оптимального сопоставления
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(distances)
            
            # Создаем множества для отслеживания уже использованных ID и индексов
            used_object_indices = set()
            used_detection_indices = set()
            
            # Обновляем соответствующие объекты
            for row_idx, col_idx in zip(row_indices, col_indices):
                # Если расстояние превышает максимальное, не считаем это соответствием
                if distances[row_idx, col_idx] > self.max_distance:
                    continue
                    
                # Получаем ID объекта и обновляем его данные
                object_id = object_ids[row_idx]
                detection = detections[col_idx]
                
                # Обновляем центроид и сбрасываем счетчик исчезновений
                self.objects[object_id]["box"] = detection["box"]
                self.objects[object_id]["class"] = detection["class"]
                self.objects[object_id]["confidence"] = detection["confidence"]
                self.objects[object_id]["centroids"].append((detection["box"]["center_x"], detection["box"]["center_y"]))
                self.objects[object_id]["disappeared"] = 0
                
                # Добавляем индексы в множества использованных
                used_object_indices.add(row_idx)
                used_detection_indices.add(col_idx)
                
            # Обработка неиспользованных объектов (увеличиваем счетчик исчезновений)
            unused_object_indices = set(range(len(object_ids))) - used_object_indices
            for idx in unused_object_indices:
                object_id = object_ids[idx]
                self.objects[object_id]["disappeared"] += 1
                
                # Если объект отсутствует слишком долго, удаляем его
                if self.objects[object_id]["disappeared"] > self.max_disappeared:
                    del self.objects[object_id]
                    
            # Регистрируем новые объекты (те, которые не были сопоставлены)
            unused_detection_indices = set(range(len(detections))) - used_detection_indices
            for idx in unused_detection_indices:
                self._register(detections[idx])
                
        return self.objects
        
    def _register(self, detection: Dict[str, Any]) -> None:
        """
        Регистрирует новый объект для отслеживания.
        
        Args:
            detection: Данные об обнаруженном объекте
        """
        # Создаем запись для нового объекта
        object_data = {
            "box": detection["box"],
            "class": detection["class"],
            "confidence": detection["confidence"],
            "centroids": [(detection["box"]["center_x"], detection["box"]["center_y"])],
            "disappeared": 0,
            "first_seen": self.frame_count,
            "last_seen": self.frame_count
        }
        
        # Добавляем объект в словарь с уникальным ID
        self.objects[self.next_object_id] = object_data
        
        # Увеличиваем счетчик ID
        self.next_object_id += 1
        self.tracked_count += 1
        
    def get_object_by_id(self, object_id: int) -> Optional[Dict[str, Any]]:
        """
        Возвращает информацию об отслеживаемом объекте по его ID.
        
        Args:
            object_id: ID объекта
            
        Returns:
            Optional[Dict[str, Any]]: Данные об объекте или None, если объект не найден
        """
        return self.objects.get(object_id)
        
    def get_objects_by_class(self, class_name: str) -> Dict[int, Dict[str, Any]]:
        """
        Возвращает все отслеживаемые объекты указанного класса.
        
        Args:
            class_name: Название класса
            
        Returns:
            Dict[int, Dict[str, Any]]: Словарь объектов, где ключ - ID объекта
        """
        return {object_id: data for object_id, data in self.objects.items() if data["class"] == class_name}
        
    def get_trajectories(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Возвращает траектории движения всех отслеживаемых объектов.
        
        Returns:
            Dict[int, List[Tuple[int, int]]]: Словарь траекторий, где ключ - ID объекта,
                                             значение - список координат центроидов
        """
        return {object_id: data["centroids"] for object_id, data in self.objects.items()}
        
    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """
        Отрисовывает траектории движения объектов на изображении.
        
        Args:
            frame: Исходное изображение
            
        Returns:
            np.ndarray: Изображение с отрисованными траекториями
        """
        # Создаем копию кадра для рисования
        output_frame = frame.copy()
        
        # Отрисовываем траектории для каждого объекта
        for object_id, data in self.objects.items():
            # Получаем центроиды объекта
            centroids = data["centroids"]
            
            # Если меньше двух точек, нет смысла рисовать траекторию
            if len(centroids) < 2:
                continue
                
            # Выбираем цвет в зависимости от ID объекта
            color = self._get_color_by_id(object_id)
            
            # Рисуем линии между последовательными центроидами
            for i in range(1, len(centroids)):
                # Координаты предыдущего и текущего центроидов
                prev_centroid = centroids[i-1]
                curr_centroid = centroids[i]
                
                # Рисуем линию
                cv2.line(output_frame, prev_centroid, curr_centroid, color, 2)
                
            # Рисуем последнюю точку крупнее
            cv2.circle(output_frame, centroids[-1], 5, color, -1)
            
            # Добавляем ID объекта рядом с последней точкой
            label = f"ID: {object_id}"
            cv2.putText(output_frame, label, (centroids[-1][0] + 10, centroids[-1][1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                       
        return output_frame
        
    def _get_color_by_id(self, object_id: int) -> Tuple[int, int, int]:
        """
        Генерирует уникальный цвет для объекта на основе его ID.
        
        Args:
            object_id: ID объекта
            
        Returns:
            Tuple[int, int, int]: Цвет в формате BGR
        """
        # Генерируем псевдослучайный цвет на основе ID
        # Используем хэш от ID для получения воспроизводимых, но разных цветов
        import hashlib
        
        # Получаем хэш от ID
        hash_str = hashlib.md5(str(object_id).encode()).hexdigest()
        
        # Преобразуем первые 6 символов хэша в цвет
        r = int(hash_str[:2], 16)
        g = int(hash_str[2:4], 16)
        b = int(hash_str[4:6], 16)
        
        return (b, g, r)  # OpenCV использует формат BGR
        
    def get_tracking_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику трекера.
        
        Returns:
            Dict[str, Any]: Словарь со статистикой
        """
        return {
            "tracked_count": self.tracked_count,
            "active_count": len(self.objects),
            "frame_count": self.frame_count,
            "max_disappeared": self.max_disappeared,
            "max_distance": self.max_distance
        } 
"""
Модуль для работы с камерой и получения видеопотока
"""

import cv2
import time
import threading
import logging
import os
import sys
import numpy as np
from typing import Optional, Tuple, List, Callable, Any
from datetime import datetime

# Добавляем корневую директорию проекта в path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VISION_SETTINGS

class Camera:
    """
    Класс для получения и обработки видеопотока с камеры БПЛА.
    Поддерживает работу с локальной камерой, USB-камерой, RTSP-потоком и т.д.
    """
    
    def __init__(self, camera_index: Optional[int] = None, resolution: Optional[Tuple[int, int]] = None, fps: Optional[int] = None):
        """
        Инициализирует объект камеры.
        
        Args:
            camera_index: Индекс камеры (если None, берется из настроек)
            resolution: Разрешение видео (ширина, высота) (если None, берется из настроек)
            fps: Частота кадров в секунду (если None, берется из настроек)
        """
        self.logger = logging.getLogger(__name__)
        
        # Загружаем настройки из конфигурации, если не указаны явно
        self.camera_index = camera_index if camera_index is not None else VISION_SETTINGS["camera_index"]
        self.resolution = resolution if resolution is not None else VISION_SETTINGS["resolution"]
        self.fps = fps if fps is not None else VISION_SETTINGS["fps"]
        
        # Опции сохранения изображений
        self.save_images = VISION_SETTINGS["save_images"]
        self.save_path = VISION_SETTINGS["save_path"]
        
        # Инициализация переменных видеопотока
        self.video_capture = None
        self.is_running = False
        self.frame_thread = None
        self.last_frame = None
        self.last_frame_time = 0
        self.frame_count = 0
        
        # Очередь кадров для обработки и обработчики
        self.frame_handlers = []
        
        # Создаем директорию для сохранения изображений, если она не существует
        if self.save_images and not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            
    def open(self) -> bool:
        """
        Открывает камеру и начинает захват видеопотока.
        
        Returns:
            bool: True если камера успешно открыта
        """
        try:
            self.logger.info(f"Открытие камеры с индексом {self.camera_index}")
            
            # Открываем камеру
            self.video_capture = cv2.VideoCapture(self.camera_index)
            
            # Проверяем, открылась ли камера
            if not self.video_capture.isOpened():
                self.logger.error(f"Не удалось открыть камеру с индексом {self.camera_index}")
                return False
                
            # Устанавливаем разрешение
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Устанавливаем частоту кадров
            self.video_capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Получаем фактические параметры камеры
            actual_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Камера открыта. Разрешение: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            # Запускаем поток для получения кадров
            self.is_running = True
            self.frame_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.frame_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при открытии камеры: {str(e)}")
            return False
            
    def close(self) -> None:
        """Закрывает камеру и останавливает захват видеопотока"""
        if not self.is_running:
            return
            
        self.logger.info("Закрытие камеры")
        
        # Останавливаем поток получения кадров
        self.is_running = False
        if self.frame_thread:
            self.frame_thread.join(timeout=2.0)
            
        # Освобождаем ресурсы камеры
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            
        self.logger.info("Камера закрыта")
        
    def _capture_frames(self) -> None:
        """Фоновый поток для непрерывного получения кадров с камеры"""
        self.logger.info("Запущен поток захвата кадров")
        
        while self.is_running and self.video_capture:
            try:
                # Получаем кадр с камеры
                ret, frame = self.video_capture.read()
                
                if not ret:
                    self.logger.warning("Не удалось получить кадр с камеры")
                    time.sleep(0.1)
                    continue
                    
                # Обновляем счетчик кадров и временную метку
                self.frame_count += 1
                self.last_frame_time = time.time()
                
                # Сохраняем последний кадр
                self.last_frame = frame.copy()
                
                # Вызываем обработчики кадров
                for handler in self.frame_handlers:
                    try:
                        handler(frame)
                    except Exception as e:
                        self.logger.error(f"Ошибка в обработчике кадров: {str(e)}")
                        
                # Если нужно сохранять изображения периодически
                if self.save_images and self.frame_count % 30 == 0:  # сохраняем каждый 30-й кадр
                    self._save_frame(frame)
                    
            except Exception as e:
                self.logger.error(f"Ошибка при захвате кадра: {str(e)}")
                time.sleep(0.1)
                
        self.logger.info("Поток захвата кадров остановлен")
        
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Возвращает последний полученный кадр.
        
        Returns:
            numpy.ndarray: Кадр в формате OpenCV или None, если кадр недоступен
        """
        return self.last_frame
        
    def add_frame_handler(self, handler: Callable[[np.ndarray], Any]) -> None:
        """
        Добавляет обработчик кадров.
        
        Args:
            handler: Функция-обработчик, которая принимает кадр (numpy.ndarray)
        """
        self.frame_handlers.append(handler)
        self.logger.info("Добавлен новый обработчик кадров")
        
    def remove_frame_handler(self, handler: Callable[[np.ndarray], Any]) -> None:
        """
        Удаляет обработчик кадров.
        
        Args:
            handler: Функция-обработчик для удаления
        """
        if handler in self.frame_handlers:
            self.frame_handlers.remove(handler)
            self.logger.info("Удален обработчик кадров")
            
    def _save_frame(self, frame: np.ndarray) -> None:
        """
        Сохраняет кадр в файл.
        
        Args:
            frame: Кадр для сохранения
        """
        if not self.save_images:
            return
            
        try:
            # Формируем имя файла с датой и временем
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{self.save_path}/frame_{timestamp}.jpg"
            
            # Сохраняем изображение
            cv2.imwrite(filename, frame)
            self.logger.debug(f"Сохранен кадр: {filename}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении кадра: {str(e)}")
            
    def get_fps(self) -> float:
        """
        Вычисляет фактическую частоту кадров.
        
        Returns:
            float: Фактическая частота кадров
        """
        # Если камера не запущена или получено менее 10 кадров, возвращаем настроенное значение
        if not self.is_running or self.frame_count < 10:
            return float(self.fps)
            
        # Вычисляем FPS на основе количества кадров и времени работы
        elapsed_time = time.time() - self.last_frame_time + 0.001  # избегаем деления на ноль
        return self.frame_count / elapsed_time
        
    def get_resolution(self) -> Tuple[int, int]:
        """
        Возвращает текущее разрешение камеры.
        
        Returns:
            Tuple[int, int]: Текущее разрешение (ширина, высота)
        """
        if not self.video_capture:
            return self.resolution
            
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
        
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Устанавливает новое разрешение камеры.
        
        Args:
            width: Ширина в пикселях
            height: Высота в пикселях
            
        Returns:
            bool: True если разрешение успешно изменено
        """
        if not self.video_capture:
            self.logger.warning("Невозможно изменить разрешение: камера не открыта")
            return False
            
        try:
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Проверяем, удалось ли изменить разрешение
            actual_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                self.resolution = (width, height)
                self.logger.info(f"Установлено новое разрешение: {width}x{height}")
                return True
            else:
                self.logger.warning(f"Не удалось установить разрешение {width}x{height}, "
                                   f"текущее разрешение: {actual_width}x{actual_height}")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка при изменении разрешения: {str(e)}")
            return False
            
    def set_fps(self, fps: int) -> bool:
        """
        Устанавливает новую частоту кадров.
        
        Args:
            fps: Частота кадров в секунду
            
        Returns:
            bool: True если частота кадров успешно изменена
        """
        if not self.video_capture:
            self.logger.warning("Невозможно изменить FPS: камера не открыта")
            return False
            
        try:
            self.video_capture.set(cv2.CAP_PROP_FPS, fps)
            
            # Проверяем, удалось ли изменить FPS
            actual_fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
            
            if actual_fps == fps:
                self.fps = fps
                self.logger.info(f"Установлен новый FPS: {fps}")
                return True
            else:
                self.logger.warning(f"Не удалось установить FPS {fps}, текущий FPS: {actual_fps}")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка при изменении FPS: {str(e)}")
            return False
            
    def take_photo(self, save: bool = True) -> Optional[np.ndarray]:
        """
        Делает снимок с камеры.
        
        Args:
            save: Сохранять ли снимок в файл
            
        Returns:
            numpy.ndarray: Снимок в формате OpenCV или None, если снимок недоступен
        """
        if not self.is_running or self.last_frame is None:
            self.logger.warning("Невозможно сделать снимок: камера не запущена или кадр недоступен")
            return None
            
        # Получаем копию последнего кадра
        photo = self.last_frame.copy()
        
        # Сохраняем снимок, если нужно
        if save:
            self._save_frame(photo)
            
        return photo 
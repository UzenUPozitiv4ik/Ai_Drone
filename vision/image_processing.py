"""
Модуль для обработки изображений и извлечения полезной информации
"""

import cv2
import numpy as np
import logging
import os
import sys
import time
from typing import Tuple, List, Dict, Any, Optional, Callable, Union
from datetime import datetime

# Добавляем корневую директорию проекта в path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VISION_SETTINGS

class ImageProcessor:
    """
    Класс для обработки изображений и извлечения полезной информации.
    Предоставляет методы для предобработки изображений, выделения признаков,
    обнаружения движения, сегментации и т.д.
    """
    
    def __init__(self):
        """Инициализирует процессор изображений"""
        self.logger = logging.getLogger(__name__)
        
        # Параметры обработки изображений
        self.params = {
            "blur_kernel_size": (5, 5),
            "edge_low_threshold": 50,
            "edge_high_threshold": 150,
            "motion_threshold": 25,
            "motion_min_area": 500,
            "clahe_clip_limit": 2.0,
            "clahe_grid_size": (8, 8)
        }
        
        # Состояние для алгоритмов, требующих предыдущие кадры
        self.previous_frame = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False)
            
        # Создаем CLAHE объект для улучшения контраста
        self.clahe = cv2.createCLAHE(
            clipLimit=self.params["clahe_clip_limit"],
            tileGridSize=self.params["clahe_grid_size"]
        )
        
        self.logger.info("Процессор изображений инициализирован")
        
    def resize(self, image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None, 
              keep_aspect_ratio: bool = True) -> np.ndarray:
        """
        Изменяет размер изображения.
        
        Args:
            image: Исходное изображение
            width: Новая ширина (если None, вычисляется с сохранением пропорций)
            height: Новая высота (если None, вычисляется с сохранением пропорций)
            keep_aspect_ratio: Сохранять ли соотношение сторон
            
        Returns:
            np.ndarray: Изображение с измененным размером
        """
        # Если и ширина, и высота не указаны, возвращаем исходное изображение
        if width is None and height is None:
            return image
            
        h, w = image.shape[:2]
        
        if keep_aspect_ratio:
            # Если указана только ширина, вычисляем высоту, сохраняя пропорции
            if width is not None and height is None:
                height = int(h * width / w)
            # Если указана только высота, вычисляем ширину, сохраняя пропорции
            elif width is None and height is not None:
                width = int(w * height / h)
            # Если указаны и ширина, и высота, выбираем меньший масштаб
            elif width is not None and height is not None:
                scale_w = width / w
                scale_h = height / h
                scale = min(scale_w, scale_h)
                width = int(w * scale)
                height = int(h * scale)
        else:
            # Если не нужно сохранять пропорции, используем указанные размеры
            width = width or w
            height = height or h
            
        # Изменяем размер изображения
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return resized
        
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Улучшает контраст изображения с использованием CLAHE.
        
        Args:
            image: Исходное изображение
            
        Returns:
            np.ndarray: Изображение с улучшенным контрастом
        """
        # Проверяем количество каналов в изображении
        if len(image.shape) == 2:
            # Одноканальное изображение (оттенки серого)
            return self.clahe.apply(image)
        elif len(image.shape) == 3:
            # Многоканальное изображение (RGB, BGR)
            # Преобразуем в LAB для применения CLAHE только к L-каналу
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Применяем CLAHE к L-каналу
            l = self.clahe.apply(l)
            
            # Объединяем каналы обратно
            lab = cv2.merge((l, a, b))
            
            # Преобразуем обратно в BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            self.logger.warning(f"Неподдерживаемое количество каналов: {image.shape}")
            return image
            
    def denoise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Удаляет шум с изображения.
        
        Args:
            image: Исходное изображение
            strength: Сила шумоподавления (1-15)
            
        Returns:
            np.ndarray: Изображение с уменьшенным шумом
        """
        # Ограничиваем силу шумоподавления
        strength = max(1, min(15, strength))
        
        # Применяем шумоподавление в зависимости от количества каналов
        if len(image.shape) == 2:
            # Одноканальное изображение
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
        else:
            # Многоканальное изображение
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
            
    def detect_edges(self, image: np.ndarray, low_threshold: Optional[int] = None, 
                    high_threshold: Optional[int] = None) -> np.ndarray:
        """
        Обнаруживает края на изображении с использованием алгоритма Canny.
        
        Args:
            image: Исходное изображение
            low_threshold: Нижний порог для алгоритма Canny
            high_threshold: Верхний порог для алгоритма Canny
            
        Returns:
            np.ndarray: Бинарное изображение с обнаруженными краями
        """
        # Используем параметры по умолчанию, если не указаны явно
        low_threshold = low_threshold or self.params["edge_low_threshold"]
        high_threshold = high_threshold or self.params["edge_high_threshold"]
        
        # Преобразуем в оттенки серого, если изображение цветное
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Применяем размытие для уменьшения шума
        blurred = cv2.GaussianBlur(gray, self.params["blur_kernel_size"], 0)
        
        # Обнаруживаем края
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        return edges
        
    def detect_motion(self, frame: np.ndarray, threshold: Optional[int] = None, 
                     min_area: Optional[int] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Обнаруживает движение между текущим и предыдущим кадрами.
        
        Args:
            frame: Текущий кадр
            threshold: Порог для бинаризации разницы между кадрами
            min_area: Минимальная площадь контура для детекции движения
            
        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: Маска движения и список обнаруженных областей движения
        """
        # Используем параметры по умолчанию, если не указаны явно
        threshold = threshold or self.params["motion_threshold"]
        min_area = min_area or self.params["motion_min_area"]
        
        # Преобразуем в оттенки серого, если изображение цветное
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Применяем размытие для уменьшения шума
        gray = cv2.GaussianBlur(gray, self.params["blur_kernel_size"], 0)
        
        # Если это первый кадр, просто сохраняем его
        if self.previous_frame is None:
            self.previous_frame = gray
            return np.zeros_like(gray), []
            
        # Вычисляем абсолютную разницу между текущим и предыдущим кадрами
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        
        # Бинаризуем разницу
        thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Расширяем области для лучшего обнаружения
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Находим контуры на бинарном изображении
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Создаем список областей движения
        motion_regions = []
        
        for contour in contours:
            # Если площадь контура меньше минимальной, игнорируем
            if cv2.contourArea(contour) < min_area:
                continue
                
            # Получаем ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(contour)
            
            # Добавляем информацию о регионе движения
            region = {
                "box": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "center_x": x + w // 2,
                    "center_y": y + h // 2
                },
                "area": cv2.contourArea(contour),
                "timestamp": time.time()
            }
            
            motion_regions.append(region)
            
        # Сохраняем текущий кадр как предыдущий для следующего вызова
        self.previous_frame = gray
        
        return thresh, motion_regions
        
    def background_subtraction(self, frame: np.ndarray, learning_rate: float = -1) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Выполняет вычитание фона для обнаружения движущихся объектов.
        
        Args:
            frame: Текущий кадр
            learning_rate: Скорость обучения для обновления модели фона (-1 для автоматического выбора)
            
        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: Маска переднего плана и список обнаруженных областей
        """
        # Применяем вычитание фона
        fg_mask = self.background_subtractor.apply(frame, learningRate=learning_rate)
        
        # Применяем морфологические операции для улучшения маски
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Находим контуры на маске переднего плана
        contours, _ = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Создаем список обнаруженных областей
        regions = []
        
        for contour in contours:
            # Если площадь контура меньше минимальной, игнорируем
            if cv2.contourArea(contour) < self.params["motion_min_area"]:
                continue
                
            # Получаем ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(contour)
            
            # Добавляем информацию о регионе
            region = {
                "box": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "center_x": x + w // 2,
                    "center_y": y + h // 2
                },
                "area": cv2.contourArea(contour),
                "timestamp": time.time()
            }
            
            regions.append(region)
            
        return fg_mask, regions
        
    def find_contours(self, image: np.ndarray, threshold: int = 127, 
                     min_area: int = 100) -> List[Dict[str, Any]]:
        """
        Находит контуры на изображении.
        
        Args:
            image: Исходное изображение
            threshold: Порог для бинаризации (если изображение не бинарное)
            min_area: Минимальная площадь контура
            
        Returns:
            List[Dict[str, Any]]: Список найденных контуров с информацией о них
        """
        # Преобразуем в оттенки серого, если изображение цветное
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Бинаризуем изображение, если оно не бинарное
        if np.max(gray) > 1:
            binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        else:
            binary = gray
            
        # Находим контуры
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Создаем список контуров
        contour_info = []
        
        for contour in contours:
            # Если площадь контура меньше минимальной, игнорируем
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # Получаем ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(contour)
            
            # Вычисляем центроид
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
                
            # Создаем аппроксимированный многоугольник
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Добавляем информацию о контуре
            info = {
                "contour": contour,
                "area": area,
                "perimeter": cv2.arcLength(contour, True),
                "box": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                },
                "center": (cx, cy),
                "approx_points": len(approx)
            }
            
            contour_info.append(info)
            
        return contour_info
        
    def detect_lines(self, image: np.ndarray, rho: float = 1, theta: float = np.pi/180, 
                    threshold: int = 50, min_line_length: int = 50, 
                    max_line_gap: int = 10) -> List[Dict[str, Any]]:
        """
        Обнаруживает линии на изображении.
        
        Args:
            image: Исходное изображение (лучше использовать после обнаружения краев)
            rho: Разрешение параметра расстояния в пикселях
            theta: Разрешение параметра угла в радианах
            threshold: Минимальное количество голосов для получения линии
            min_line_length: Минимальная длина линии
            max_line_gap: Максимальный разрыв между точками линии
            
        Returns:
            List[Dict[str, Any]]: Список обнаруженных линий
        """
        # Проверяем, является ли изображение бинарным или оттенками серого
        if len(image.shape) == 3:
            # Преобразуем в оттенки серого
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Обнаруживаем края
            edges = self.detect_edges(gray)
        elif np.max(image) > 1:
            # Изображение в оттенках серого
            edges = self.detect_edges(image)
        else:
            # Изображение уже бинарное (возможно, края)
            edges = image
            
        # Обнаруживаем линии
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, None, min_line_length, max_line_gap)
        
        # Создаем список обнаруженных линий
        detected_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Вычисляем длину и угол линии
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # Нормализуем угол в диапазоне [0, 180]
                if angle < 0:
                    angle += 180
                    
                # Добавляем информацию о линии
                line_info = {
                    "start": (x1, y1),
                    "end": (x2, y2),
                    "length": length,
                    "angle": angle
                }
                
                detected_lines.append(line_info)
                
        return detected_lines
        
    def detect_circles(self, image: np.ndarray, min_radius: int = 10, 
                      max_radius: int = 100) -> List[Dict[str, Any]]:
        """
        Обнаруживает окружности на изображении.
        
        Args:
            image: Исходное изображение
            min_radius: Минимальный радиус окружности
            max_radius: Максимальный радиус окружности
            
        Returns:
            List[Dict[str, Any]]: Список обнаруженных окружностей
        """
        # Преобразуем в оттенки серого, если изображение цветное
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Применяем размытие для уменьшения шума
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Обнаруживаем окружности
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
        )
        
        # Создаем список обнаруженных окружностей
        detected_circles = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for circle in circles[0, :]:
                center_x, center_y, radius = circle
                
                # Добавляем информацию об окружности
                circle_info = {
                    "center": (center_x, center_y),
                    "radius": radius,
                    "area": np.pi * radius**2
                }
                
                detected_circles.append(circle_info)
                
        return detected_circles
        
    def perspective_transform(self, image: np.ndarray, 
                             source_points: np.ndarray, 
                             destination_points: Optional[np.ndarray] = None, 
                             output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Выполняет преобразование перспективы изображения.
        
        Args:
            image: Исходное изображение
            source_points: Массив точек на исходном изображении (4 точки)
            destination_points: Массив точек назначения (4 точки)
            output_size: Размер выходного изображения (ширина, высота)
            
        Returns:
            np.ndarray: Преобразованное изображение
        """
        # Если размер выходного изображения не указан, используем размер исходного
        if output_size is None:
            output_size = (image.shape[1], image.shape[0])
            
        # Если точки назначения не указаны, создаем прямоугольник
        if destination_points is None:
            destination_points = np.array([
                [0, 0],
                [output_size[0] - 1, 0],
                [output_size[0] - 1, output_size[1] - 1],
                [0, output_size[1] - 1]
            ], dtype=np.float32)
        
        # Преобразуем точки в нужный формат
        source_points = source_points.astype(np.float32)
        destination_points = destination_points.astype(np.float32)
        
        # Вычисляем матрицу преобразования
        M = cv2.getPerspectiveTransform(source_points, destination_points)
        
        # Применяем преобразование
        warped = cv2.warpPerspective(image, M, output_size)
        
        return warped
        
    def overlay_mask(self, image: np.ndarray, mask: np.ndarray, 
                    color: Tuple[int, int, int] = (0, 0, 255), 
                    alpha: float = 0.5) -> np.ndarray:
        """
        Накладывает маску на изображение с указанным цветом и прозрачностью.
        
        Args:
            image: Исходное изображение
            mask: Бинарная маска
            color: Цвет в формате BGR
            alpha: Коэффициент прозрачности (0.0 - 1.0)
            
        Returns:
            np.ndarray: Изображение с наложенной маской
        """
        # Создаем копию изображения
        output = image.copy()
        
        # Если изображение в оттенках серого, преобразуем его в цветное
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
            
        # Создаем цветной оверлей для маски
        overlay = np.zeros_like(output)
        
        # Если маска одноканальная, создаем многоканальную маску
        if len(mask.shape) == 2:
            color_mask = np.dstack([mask, mask, mask])
        else:
            color_mask = mask
            
        # Заполняем оверлей цветом только там, где маска не равна нулю
        overlay[color_mask > 0] = color
        
        # Накладываем оверлей на изображение с указанной прозрачностью
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        
        return output
        
    def draw_regions(self, image: np.ndarray, regions: List[Dict[str, Any]], 
                    color: Tuple[int, int, int] = (0, 255, 0), 
                    thickness: int = 2) -> np.ndarray:
        """
        Рисует прямоугольники и метки для регионов на изображении.
        
        Args:
            image: Исходное изображение
            regions: Список регионов с координатами
            color: Цвет рамок и меток в формате BGR
            thickness: Толщина линий
            
        Returns:
            np.ndarray: Изображение с отрисованными регионами
        """
        # Создаем копию изображения
        output = image.copy()
        
        # Если изображение в оттенках серого, преобразуем его в цветное
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
            
        # Рисуем каждый регион
        for i, region in enumerate(regions):
            # Получаем координаты прямоугольника
            x = region["box"]["x"]
            y = region["box"]["y"]
            w = region["box"]["width"]
            h = region["box"]["height"]
            
            # Рисуем прямоугольник
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
            
            # Добавляем метку с номером региона
            label = f"Region {i+1}"
            cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            
        return output
        
    def save_image(self, image: np.ndarray, filename: Optional[str] = None, 
                  directory: Optional[str] = None) -> str:
        """
        Сохраняет изображение в файл.
        
        Args:
            image: Изображение для сохранения
            filename: Имя файла (если None, генерируется автоматически)
            directory: Каталог для сохранения (если None, используется каталог из настроек)
            
        Returns:
            str: Путь к сохраненному файлу
        """
        # Если каталог не указан, используем каталог из настроек
        if directory is None:
            directory = VISION_SETTINGS["save_path"]
            
        # Создаем каталог, если он не существует
        os.makedirs(directory, exist_ok=True)
        
        # Если имя файла не указано, генерируем его на основе текущего времени
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"image_{timestamp}.jpg"
            
        # Формируем полный путь к файлу
        file_path = os.path.join(directory, filename)
        
        # Сохраняем изображение
        cv2.imwrite(file_path, image)
        
        self.logger.debug(f"Изображение сохранено в {file_path}")
        
        return file_path
        
    def get_image_stats(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Вычисляет статистические характеристики изображения.
        
        Args:
            image: Изображение для анализа
            
        Returns:
            Dict[str, Any]: Словарь со статистическими характеристиками
        """
        # Преобразуем в оттенки серого, если изображение цветное
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Вычисляем гистограмму
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Вычисляем статистические характеристики
        mean, std = cv2.meanStdDev(gray)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
        
        # Возвращаем статистику
        return {
            "shape": image.shape,
            "mean": float(mean[0, 0]),
            "std": float(std[0, 0]),
            "min": min_val,
            "max": max_val,
            "min_location": min_loc,
            "max_location": max_loc,
            "histogram": hist.flatten().tolist()
        } 
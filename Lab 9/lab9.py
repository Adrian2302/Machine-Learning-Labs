# B93825 Adrián Hernáncez Young

import numpy as np
import PIL
from random import sample
from PIL import Image, ImageDraw, ImageFont
import random

SHAPE = 256

def load_image(filename, resize):
    image = np.array(PIL.Image.open(filename).resize(resize).convert('RGB')).astype(np.float32)
    return image

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def manhattan_distance(p1, p2):
    distance = 0
    for val1, val2 in zip(p1,p2):
        distance += np.abs(val1-val2)
    return np.sum(distance)

def nearest_centroid(point, centroids, distance):
    index_nearest = -1
    nearest_centroid_distance = 1e15
    current_iteration = 0
    for cen in centroids:
        current_centroid_distance = distance(point, cen)
        if current_centroid_distance < nearest_centroid_distance:
            nearest_centroid_distance = current_centroid_distance
            index_nearest = current_iteration
        current_iteration += 1
    return (index_nearest, nearest_centroid_distance)

def distance_chosen(distance):
    if distance == "euclidean":
        return euclidean_distance
    elif distance == "manhattan":
        return manhattan_distance
    else:
        print("Error: No existe la función de distancia ingresada")
        exit()

def init_centroids(data, k):
    centroids = []
    random_centroids_init_index = sample(range(len(data)),k)
    for index in random_centroids_init_index:
        centroids.append(data[index])
    return centroids

def match_points_centroids(k, data, centroids, distance):
    centroids_points = []
    for _ in range(k):
        centroids_points.append([])

    for point in data:
        nearest = nearest_centroid(point, centroids, distance)
        centroids_points[nearest[0]].append((point, nearest[1]))

    return centroids_points

def sum_distance(center):
    sum = 0
    for value in center:
        sum += value[1]
    return sum

def lloyd(data, k, iters, type, distance):
    distance = distance_chosen(distance)
    centroids = init_centroids(data, k)
    sum_error = 0

    for _ in range(iters): # 2. Itere sobre los siguientes pasos:
        distance_centroid = 0
        points_centroids_matched = match_points_centroids(k, data, centroids, distance) # a. Asigne cada punto a su centro más cercano
        for index, center in enumerate(points_centroids_matched):
            if type == "means":
                centroids[index] = np.mean(center)  # b. Calcule nuevos centros para cada cluster como el promedio de sus puntos
                distance_centroid += sum_distance(center) 

            elif type == "medoids" or type == "mediods": 
                random_index = random.randint(0, len(center))
                suma = 0
                for point in center:
                    suma += distance(center[random_index][0], point[0])
                
                current_error = sum_distance(center) 
                if current_error > suma: # Opción 2: Se elige un punto del cluster, si resulta en un mejor agrupamiento se convierte en el nuevo centro
                    centroids[index] = center[random_index][0]
                    distance_centroid += suma
                else:
                    distance_centroid += current_error

            else:
                print("Error: No existe el tipo ingresado")
                exit()

        sum_error = distance_centroid
        
    return centroids , sum_error

def insert_color(img, position, size, color):
    x, y = position
    draw = ImageDraw.Draw(img)
    draw.rectangle((position, (x + size, y + size)), fill=color, outline ="red")
    return img


if __name__ == "__main__":

    image1 = load_image("img1.jpg", (SHAPE, SHAPE))
    image2 = load_image("img2.jpg", (SHAPE, SHAPE))
    image3 = load_image("img3.jpg", (SHAPE, SHAPE))

    images = [image1, image2, image3]
    types = ["mediods", "means"]
    distances = ["euclidean", "manhattan"]
    iters = 10
    k = 5 # 1. Determine k puntos centrales iniciales
    color_size = 20

    image_number = 0
    for image in images: # Probar las 4 combinaciones de parámetros con las 3 imágenes:
        image_number += 1
        for type in types:
            for distance in distances:

                color_pallette = []
                color_error = []
                for _ in range(0,3): # 3 Pruebas por si se encuentra un mínimo local
                    color = lloyd(image.reshape(SHAPE**2, 3), k, iters, type, distance) # Reshape porque no me sirve de la otra manera o tengo que cambiar mucha cosa
                    color_pallette.append(color[0])
                    color_error.append(color[1])
                
                min_error_index = np.argmin(color_error)
                best_colors = color_pallette[min_error_index]
                real_image = Image.fromarray(np.uint8(image)).convert('RGB')                                      
                position = 1
                print((f'Paleta de colores de img{image_number}.jpg, tipo={type}, distance={distance}'))
                for index, col in enumerate(best_colors):
                    print(f'Código RGB {index}: {int(col[0])}, {int(col[1])}, {int(col[2])}')
                    location = (15, 40*position)
                    real_image = insert_color(real_image, location, color_size, (int(col[0]), int(col[1]), int(col[2])))
                    position+=1

                real_image.save(f'Colores_imagen_{image_number}_{type}_{distance}.jpg', quality=95)

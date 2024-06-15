import collections
import random
import numpy as np
import pdb

def generar_popularidad_random():
    # Se podría cambiar y utilizar otra distribución: la normal o la que sea
    # Uniform popularity distribution
    return random.randint(1, 10)

def rellenar_archivo(archivo, size):
    letras = list("abcdefghijklmnñopqrstuvwxyz")
    random.shuffle(letras)
    with open("files/"+archivo, "wb") as f:
        for _ in range(size):
            if letras:  # Check if letras is not empty before popping
                letra = random.choice(letras).encode("utf-8")
                f.write(letra)

def descargar_archivo_del_almacenamiento_principal(size, tasa_bits):

    # Calcular el tiempo de descarga
    tiempo_descarga = size / tasa_bits

    # Agregar latencia y variabilidad
    tiempo_descarga += random.uniform(0.1, 0.5)  # Segundos

    return tiempo_descarga

def calcular_tiempo_respuesta_promedio(solicitudes):
    # Lista para almacenar los tiempos de respuesta
    tiempos_respuesta = []

    # Recorrer todas las solicitudes
    for solicitud in solicitudes:
        # Calcular el tiempo de respuesta
        tiempo_respuesta = solicitud["tiempo_fin"] - solicitud["tiempo_inicio"]

        # Agregar el tiempo de respuesta a la lista
        tiempos_respuesta.append(tiempo_respuesta)

    # Calcular el tiempo de respuesta promedio
    tiempo_respuesta_promedio = np.mean(tiempos_respuesta)

    return tiempo_respuesta_promedio

def calcular_tasa_aciertos_cache(solicitudes):
    # Número total de solicitudes
    numero_solicitudes = len(solicitudes)
    print(f"Number of requests: {numero_solicitudes}")
    # Número de solicitudes que fueron hits en la caché
    numero_aciertos_cache = 0

    # Recorrer todas las solicitudes
    for solicitud in solicitudes:
    # Si la solicitud fue un hit en la caché
        if solicitud["hit_cache"]:
            # Incrementar el número de aciertos en la caché
            numero_aciertos_cache += 1
    print("Hits in the cache: ", numero_aciertos_cache)
    # Calcular la tasa de aciertos en la caché
    tasa_aciertos_cache = numero_aciertos_cache / numero_solicitudes

    return tasa_aciertos_cache

def comprobar_peticiones_repetidas(peticiones):
    # Obtener la lista de archivos y clientes de las peticiones
    archivos = [peticion["archivo"] for peticion in peticiones]

    # Comprobar si hay archivos o clientes repetidos
    archivos_repetidos = [item for item, count in collections.Counter(archivos).items() if count > 1]

    # Eliminar los archivos y clientes repetidos
    archivos_sin_repetir = list(set(archivos))
    # print(archivos_sin_repetir)
    numero_archivos_repetidos = len(archivos) - len(archivos_sin_repetir)
    return archivos_sin_repetir, numero_archivos_repetidos

def tiempo_envio_archivos(archivos_a_enviar, tasa_bits, size):
    # Simulación del envío del archivo al cliente
    total_size = sum([size for archivo in archivos_a_enviar])
    tiempo_envio = total_size / tasa_bits  # Segundos
    return tiempo_envio


################################################
# Uncoded - Coded
################################################

def comprobar_archivos_repetidos(archivos):
    if len(set(archivos)) == 1:
        return True
    else:
        return False
    

def dividir_archivo(archivo, numero_partes):
    partes = []
    for i in range(numero_partes):
        contenido = archivo
        parte = contenido[i::numero_partes]
        partes.append(parte)
    return partes

def xor_files(filenames):
    result = 0

    # pdb.set_trace()
    for filename in filenames:

        with open("files/"+filename, 'rb') as f:
            for byte in f.read():
                result ^= byte
    with open("files/file_xor", "wb") as f:
        f.write(result.to_bytes(1, 'little'))



# This function takes a ROW K-VECTOR of requests from the different users
# (0 for no request) and a MxK matrix of cache contents and finds a
# largest clique, returning the indices of the corresponding users
def find_largest_clique(requests, caches):
    # Discard users without any request
    usrs_with_req = [i for i, req in enumerate(requests) if req != 0]
    requests_np = np.array(requests)
    requests_aux = requests_np[usrs_with_req]
    matriz_aux = caches[:, usrs_with_req]
    n_left = len(usrs_with_req)  # number of users remaining
    max_clique_len = 0
    usr_clique = []
    
    for i in range(n_left):  # loop over users i
        req_i = requests_aux[i]  # request from current user
        usr_candidates = []  # This will store the candidates to form a clique with current user

        for j in range(i + 1, matriz_aux.shape[1]):  # Loop over users later than i
            if np.any(matriz_aux[:, j] == req_i) or requests_aux[j] == req_i:  # Do nothing unless user j stores or demands req_i
                req_j = requests_aux[j]
                if np.any(matriz_aux[:, i] == req_j) or req_i == req_j:  # If current user stores or demands req_j...
                    usr_candidates.append(j)  # ...store as viable candidate
           
        # Find largest clique that includes current user i
        if not usr_candidates:
            clique_i = [usrs_with_req[i]]
        elif max_clique_len >= 1 + len(usr_candidates):
            clique_i = []  # I already have a clique larger than all the candidates
        else:
            aux = find_largest_clique(requests_aux[usr_candidates], matriz_aux[:, usr_candidates])
            aux = [usr_candidates[k] for k in aux]
            clique_i = [usrs_with_req[i]] + aux

        # If current clique is largest so far, store it in usr_clique (output)
        if max_clique_len < len(clique_i):
            usr_clique = clique_i
            max_clique_len = len(clique_i)

    return usr_clique

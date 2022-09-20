import cv2
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage 
from matplotlib import cm


def caracteristicas(img):
    imgGris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    umbral, _ = cv2.threshold(imgGris, 0, 255, cv2.THRESH_OTSU)
    
    mascara = np.uint8((imgGris < umbral) * 255)
    salida = cv2.connectedComponentsWithStats(mascara, cv2.CV_32S)
    
    cantidadObjetos = salida[0]
    etiquetas = salida[1]
    stats = salida [2]
    
    #El stat 0 es el que tiene mayor area
    mascara = (np.argmax(stats[:,4][1:])+1 == etiquetas) 
    
    mascara = ndimage.binary_fill_holes(mascara).astype(int) 
    
    #Extraer el rasgo rojo y verde de la imagen 
    rojo = np.sum(mascara * img[:,:,0]/255)/np.sum(mascara)
    verde = np.sum(mascara * img[:,:,1]/255)/np.sum(mascara)
    
    #Extraer el rasgo de la tasa de aspecto
    mascara1 = np.uint8(mascara*255)
    contornos, jerarquia = cv2.findContours(mascara1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cnt = contornos[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int32(box) #Para usar la mascara 
    
    #Extraer las dimensiones de la mascara 
    m,n = mascara1.shape
    aux = np.zeros((m,n))
    mascaraRect = cv2.fillConvexPoly(aux, box, 1)
    mascaraRect= np.uint8(mascaraRect.copy()*255)
    
    #Calcular las dimensiones de la mascara 
    contornosMask, _ = cv2.findContours(mascara1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntMask = contornosMask[0]
    
    centro, dimensiones, rotacion = cv2.minAreaRect(cntMask)
    
    #Calcular la tasa de aspecto 
    tasaAspecto = float(dimensiones[1])/float(dimensiones[0]) if dimensiones [1] < dimensiones [0] else  float (dimensiones [0])/float(dimensiones[1])
    return rojo, verde, tasaAspecto

"""imagen = "imagenesFrutas/banano25.jpg"
img = cv2.imread(imagen)"""

#rojo, verde, tasa Aspecto = caracteristicas(img)
datos = []
clases = []

for i in range(1,37):
    datos.append(caracteristicas(cv2.imread("banano"+str(i)+".jpg")))
    clases.append(1)
    datos.append(caracteristicas(cv2.imread("manzana"+str(i)+".jpg")))
    clases.append(-1)

datos = np.array(datos)
clases = np.array(clases)

fig = plt.figure()
grafica = fig.add_subplot(111,projection = "3d")

for i in range(0,72):
    if clases[i] == 1:
        grafica.scatter(datos[i,0],datos[i,1],datos[i,2], marker = '*', c='g')
    else:
         grafica.scatter(datos[i,0],datos[i,1],datos[i,2], marker = '+', c='r')

grafica.set_xlabel('Rojo')
grafica.set_ylabel('Verde')
grafica.set_zlabel('Tasa aspecto')
plt.show()

#Fase aprendizaje 
#Hallar el hiperplano que separa a las ods clases 

a = np.zeros((4,4))
b = np.zeros((4,1))

for i in range(0,72):
    x = np.append([1],datos[i])
    x = x.reshape((4,1))
    y = clases[i]
    a = a + x*x.T
    b = b + x * y

inv = np.linalg.inv(a)
w = np.dot(inv,b)

#funcion del hiperplano
# w0 + w1x + w2y + w3z = 0
#despejar z para poder graficar

X = np.arange(0,1,0.1)
Y = np.arange(0,1,0.1)
X,Y = np.meshgrid(X,Y)

Z = -(w[0] + w[1]*X + w[2]*Y)/w[3]

#Dibujar el plano en la misma grafica 
surf = grafica.plot_surface(X,Y,Z, cmap = cm.Blues)

#Fase de prueba o clasificacion

#Visualizar los datos de prueba
datosPrueba = []
clasesPrueba = []

for i in range(1,7):
    datosPrueba.append(caracteristicas(cv2.imread("banano"+str(i)+".jpg")))
    clasesPrueba.append(1)
    datosPrueba.append(caracteristicas(cv2.imread("manzana"+str(i)+".jpg")))
    clasesPrueba.append(-1)
datosPrueba = np.array(datosPrueba)
clasesPrueba = np.array(clasesPrueba)

for i in range(0,12):
    if clasesPrueba[i] == 1:
        grafica.scatter(datosPrueba[i,0],datosPrueba[i,1],
                        datosPrueba[i,2], marker = '*', c='black')
    else:
         grafica.scatter(datosPrueba[i,0],datosPrueba[i,1],
                         datosPrueba[i,2], marker = '+', c='blue')

#Clasificar una sola imagen
imagen = "manzana4.jpg"
img = cv2.imread(imagen)
x = np.append([1],caracteristicas(img))
if np.sign(np.dot(w.T,x)) == 1:
    print(imagen + " es un banano")
else:
    print(imagen + " es una manzana ")



import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy
import math
import matplotlib.pyplot as plt
rng = numpy.random


# Parametros
gradiente_aprendizaje = 0.1
iteraciones= 200
display_step = 1
meses = 15

#Definir los datos de entrenamiento (train)
arregloNumerosX = []
arregloNumerosY = dataset.percentPositive
n=1
for i in arregloNumerosY:
  arregloNumerosX.append(n)
  n= n+1
train_X = numpy.asarray(arregloNumerosX)

train_Y = numpy.asarray(arregloNumerosY)
n_samples = train_X.shape[0] #tamaño del array



#Creamos los Placeholders
X = tf.placeholder("float", name="Mes")
Y = tf.placeholder("float", name="Ventas")

# Creamos las variables de entreno
W = tf.Variable(rng.randn(), name="peso")
b = tf.Variable(rng.randn(), name="parciales")

# Construimos el modelo lineal
pred = tf.add(tf.multiply(X, W), b)

# Calculamos la media del error cuadrado
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

#  Calculamos el descenso de gradiente
optimizer = tf.train.GradientDescentOptimizer(gradiente_aprendizaje).minimize(cost)

# Inicializamos las variables
init = tf.global_variables_initializer()


with tf.Session() as sess:

    # ejecutamos el inicializador
    sess.run(init)

    # Ajustamos datos de entrenamiento
    for epoch in range(iteraciones):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        # Mostramos en pantalla los registros por cada paso (log)
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
    print("Optimización finalizada!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

  

    # Mostramos resultados
    plt.rcParams['axes.facecolor'] = 'ffffff'
    plt.plot(train_X, train_Y, marker="o", label='Aprobacion de twiters')
    plt.xlabel("Tweet")
    plt.ylabel("Aprobacion positiva")
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), 'r-', label='pronostico')
    plt.legend()
    plt.show()

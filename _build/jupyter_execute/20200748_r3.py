#!/usr/bin/env python
# coding: utf-8

# # Reporte 3: 

# In[1]:


import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sympy import *
import pandas as pd
#from causalgraphicalmodels import CausalGraphicalModel


# ## Parte 1: Código

# Trabajo conjunto con María Fernanda Carrillo - 20201596

# ## Equilibrio en el mercado de Dinero: Explique y Derive

# ### 1. Instrumentos de política monetaria que puede utilizar el Banco Central

# Entendiendo que el Banco Central es la autoridad competente de controlar y gestionar la oferta monetaria, tiene la posibilidad de hacer uso de políticas monetarias expansivas (con el aumento de oferta se reduce la tasa de interés) o contractivas (reducción de la demanda reduce el nivel de producción y empleo)  mediante el usos de instrumentos de política monetaria.

# En primer lugar, existe la oferta monetaria como instrumento de política. Mediante este tipo de políticas el Banco central realiza operaciones mediante la compra o venta de activos financieron o bonos a los bancos comericales (Jiménez 2020, pp. 80). Cuando compran bonos del mercado para inyectarlos en la economía se le llama una política monetaria expansivas. Mediante esta política aumenta la oferta monetaria gracias al incremento de sus activos. Cuando venden bonos al mercado y retiran dinero de la economía se le llama una política monetaria contractiva. Mediante esta política se reduce la oferta monetaria debido a la disminución de la base monetaria. 

# En segundo lugar, el coeficiente legal de encaje es también un instrumento de política. A través de una política monetaria expansiva, se disminuye la tasa de encaje al incrementar el dinero disponibles de los bancos para realizar préstamos. Es decir, "disminución de la tasa de encaje aumenta la 
# posibilidad de creación de dinero bancario porque aumenta el multiplicador de dinero bancario; y, este aumento del dinero bancario implica un aumento de la oferta monetaria" (Jiménez 2020, pp. 82). Lo contrario sucede con una política monetaria contractiva en el cuál aumenta el coeficiente de encaje y disminuye la oferta monetaria ya que los bancos aumenta su proporción de depósitos reduciendo así el multiplicador bancario.

# Y finalmente, está la tasa de interés (r) como instrumento de política. En este caso, el r se convierte en instrumento y la oferta monetaria se convierte en una variable endógena, es decir, la r se vuelve la variable referencial. Con una política monetaria expansiva se reduce la tasa de interés y por ende aumentado la oferta monetaria. Lo contrario sucede con una política monetaria contractiva ya que aumenta la tasa de interés y por ende la oferta monetaria. 

# ### 2. Derive la oferta real de dinero: Ms = Mo/P

# La oferta de dinero (M^s) representa cuando dinero hay de verdad en la economía. Se la considera una variable exógena e instrumento de la política monetaria. Asimismo, vamosa a ver a la oferta dividida entre el nivel general de precios de la economía (P)

# $$ \frac{M^s}{P} = \frac{M^s_0}{P} $$

# ### 3. Demanda de Dinero: L1 + L2 - Md = kY - ji

# Para poder formular y entender la demanda de dinero se deben analizar los motivos por el cuál se demanda dinero: motivos de transacción, precaución y el método especulativo.

# En el primer bloque se encuenta el motivo de transacción y de precaución. En el primer motivo se demanda dinero para poder realizar transacciones y se entiende que esta transacción está en una relación directa con el producto de la economía. Consecuentemente, a partir de este motivo, la demanda de dinero depende positivmente del Ingreso (Y). Con el segundo motivo, se pide dinero como manera de precaución para pagar las deudas. En este sentido, la demanda de dinero dependerá positivamente del ingreso. Bajo estos dos supuestos se entiende la elasticidad del total de los ingresos de la economía (k): mientras más elastacidad, mayor cantidad de dinero que se demanda en cuanto se incrementa el PBI. 

# $$ L_1 = kY $$

# En el siguiente bloque, se entiende a partir del motivo de especulación que la demanda de dinero depende inversamente de las tasas de interés de los bonos. Las personas tienen dos opciones respecto a como manejar su dinero: tenerlo ellos mismos o en el banco. Los individuos preferirán tener su propio dinero cuando la tasa de interés (i) disminuye (= cantidad de dinero sube), pero cuando la tasa de interés aumenta, preferieren depositarlo en el banco (=cantidad de dinero baja). En este sentido existe una elesticidad de sustitución del dinero en el bolsillo propio de los individuos versus depositado en el banco (j)

# $$ L_2 = -ji $$

# Si asumismos que ambas demandas están en términos reales, la función de la demanda de dinero sería:
# 
# $$ M^d = L_1 + L_2 $$ 
# 
# $$ M^d = kY -ji $$

# Recapitulando, los parametros "k" y "j" indican respectivamente la sensibilidad de la demanda de dinero ante variaciones del ingreso “Y”, e indica cuán sensible es la demanda de  dinero ante las variaciones de la tasa de interés nominal de los bonos, “i”.

# ### 4. Ecuación de equilibrio en el mercado de dinero

# Se deriva apartir de la eucación de la Oferta de Dinero (M^s) y Demanda de Dinero (M^d) en equilibrio:
# 
# $$ M^s = M^d $$
# 
# $$ \frac{M^s}{P} = kY - ji $$

# ¿Por qué la i? Suponiendo que la inflación esperada es cero, no habrá mucha diferencia entre la tasa de interés nominal (i) y la real (r). Por estar zazón, se puede reemplazar de la siguiente manera: 
# 
# $$ \frac{M^s}{P} = kY - jr $$
# 
# $$ M_0 = P(kY -jr) $$

# ### 5. Grafique Equilibrio en el mercado

# In[2]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[3]:


# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#6cd429')

# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=7.5, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.5, "$r_0$", fontsize = 12, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# ## Estática Comparativa en el mercado de Dinero

# ### 1.Explique y grafique si ∆Y < 0

# De acuerdo a Jiménez, cuanto el ingreso de la producción aumenta, la demanda monetaria también aumenta (2020, pp. 86). Entonces en el caso contrario, cuando el ingreso de la producción disminuye, la demanda monetaria también disminuye y por ende la tasa de interés también tiene que disminuir para regresar el equilibrio. 

# In[4]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 36
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_0 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[5]:


# Parameters con cambio en el nivel del producto
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y_1 = 20
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P, r, Y_1)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[6]:


# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Caído en el nivel de ingresos (Y)", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#CD5C5C')
#ax1.plot(MD_1, label= '$L_0$', color = '#CD5C5C')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=0, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.5, "$r_0$", fontsize = 12, color = 'black')
ax1.text(50, 8, "$E_0$", fontsize = 12, color = 'black')

# Nuevas curvas a partir del cambio en el nivel del producto
ax1.plot(MD_1, label= '$L_1$', color = '#4287f5', linestyle = 'dashed')
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y=8, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")
ax1.text(0, 0, "$r_1$", fontsize = 12, color = 'black')
ax1.text(50, 0, "$E_1$", fontsize = 12, color = 'black')
ax1.text(60, 1, '∆$Y$', fontsize=12, color='black')
plt.text(60, -0.5, '←', fontsize=15, color='grey')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()


# ### 2.Explique y grafique si ∆k <0

# Ante la disminución de la sensibilidad de demanda de dinero , la demanda de dinero caerá. Esto se reflejará en una contracción de la curva de demanda. Dado que la demanda será menor a la oferta, para volver al equilibrio, la tasa de interés  disminuirá.

# In[7]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_00 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS1 = MS_0 / P
MS1


# In[8]:


# Parameters con cambio en el nivel del producto
r_size = 100

k = 0.2
j = 0.2                
P  = 10 
Y_1 = 20
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_01 = MD(k, j, P, r, Y_1)
# Necesitamos crear la oferta de dinero.
MS1 = MS_0 / P
MS1


# In[9]:


# Equilibrio en el mercado de dinero

# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Caída en la sensibilidad de demanda de Dinero (k)", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_00, label= '$L_0$', color = '#CD5C5C')
#ax1.plot(MD_01, label= '$L_0$', color = '#CD5C5C')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=0, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 8, "$r_0$", fontsize = 12, color = 'black')
ax1.text(51, 8, "$E_0$", fontsize = 12, color = 'black')

# Nuevas curvas a partir del cambio en el nivel del producto
ax1.plot(MD_1, label= '$L_1$', color = '#4287f5', linestyle = 'dashed')
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y=7.5, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")
ax1.text(0, 0, "$r_1$", fontsize = 12, color = 'black')
ax1.text(51, -2.5, "$E_1$", fontsize = 12, color = 'black')
ax1.text(60, 1, '∆$K$', fontsize=12, color='black')
plt.text(60, -0.5, '←', fontsize=15, color='grey')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()


# ### 3.Explique y grafique si ∆Ms < 0

# Una disminución en la oferta hace que en el mercado se tenga que reducir la tasa de interés y así aumentar la demanda y restablecer el equilibrio en el mercado. Esto va generar a que la recta de la oferta de dinero se deplace a la izquierda. 

# In[10]:


# Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P, r, Y)
# Necesitamos crear la oferta de dinero.
MS = MS_0 / P
MS


# In[11]:


# Parameters con cambio en el nivel del producto
r_size = 100

k = 0.5
j = 0.2                
P_1  = 20 
Y = 35
MS_0 = 500

r = np.arange(r_size)

# Necesitamos crear la funcion de demanda 

def MD(k, j, P, r, Y):
    MD_eq = (k*Y - j*r)
    return MD_eq
MD_1 = MD(k, j, P_1, r, Y)
# Necesitamos crear la oferta de dinero.
MS_1 = MS_0 / P_1
MS


# In[12]:



# Creamos el seteo para la figura 
fig, ax1 = plt.subplots(figsize=(10, 8))

# Agregamos titulo t el nombre de las coordenadas
ax1.set(title="Disminucion en la oferta de Dinero (Ms)", xlabel=r'M^s / P', ylabel=r'r')

# Ploteamos la demanda de dinero
ax1.plot(MD_0, label= '$L_0$', color = '#CD5C5C')


# Para plotear la oferta de dinero solo necesitamos crear una linea vertical
ax1.axvline(x = MS,  ymin= 0, ymax= 1, color = "grey")

# Creamos las lineas puntadas para el equilibrio
ax1.axhline(y=8, xmin= 0, xmax= 0.5, linestyle = ":", color = "black")

# Agregamos texto
ax1.text(0, 7.5, "$r_0$", fontsize = 12, color = 'black')
ax1.text(51, 8, "$E_0$", fontsize = 12, color = 'black')


# Nuevas curvas a partir del cambio en el nivel del producto
#ax1.plot(MD_1, label= '$L_1$', color = '#4287f5')
ax1.axvline(x = MS_1,  ymin= 0, ymax= 1, color = "grey")
ax1.axhline(y=13, xmin= 0, xmax= 0.28, linestyle = ":", color = "black")
ax1.text(0, 13, "$r_1$", fontsize = 12, color = 'black')
ax1.text(26, 13, "$E_1$", fontsize = 12, color = 'black')
ax1.text(35, 13, '∆$Ms$', fontsize=12, color='black')
ax1.text(34, 12, '←', fontsize=15, color='grey')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()

plt.show()


# ## Curva LM

# ### 1.Paso a paso LM matemáticamente (a partir del equilibrio en el Mercado Monetario) y grafique

# Siendo la ecuación de equilibrio en el Mercado de Dinero:

# $$  M^s = M^d$$

# $$  \frac{M^s_0}{P} = {kY - jr} $$

# 
# $$ kY -\frac{M^s_0}{P} = jr $$
# 
# $$ \frac{kY}{j} - \frac{M^s_0}{Pj} = r $$
# 
# $$ r = - \frac{M^s_0}{Pj} + \frac{kY}{j} $$ 
# 
# $$ r = - \frac{1}{j}\frac{M^s_0}{Pj} + \frac{k}{j}Y $$ 
# 

# In[13]:


#1----------------------Equilibrio mercado monetario

    # Parameters
r_size = 100

k = 0.5
j = 0.2                
P  = 10 
Y = 35

r = np.arange(r_size)


    # Ecuación
def Ms_MD(k, j, P, r, Y):
    Ms_MD = P*(k*Y - j*r)
    return Ms_MD

Ms_MD = Ms_MD(k, j, P, r, Y)


    # Nuevos valores de Y
Y1 = 45

def Ms_MD_Y1(k, j, P, r, Y1):
    Ms_MD = P*(k*Y1 - j*r)
    return Ms_MD

Ms_Y1 = Ms_MD_Y1(k, j, P, r, Y1)


Y2 = 25

def Ms_MD_Y2(k, j, P, r, Y2):
    Ms_MD = P*(k*Y2 - j*r)
    return Ms_MD

Ms_Y2 = Ms_MD_Y2(k, j, P, r, Y2)

#2----------------------Curva LM

    # Parameters
Y_size = 100

k = 0.5
j = 0.2                
P  = 10               
Ms = 30            

Y = np.arange(Y_size)


# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)


# In[14]:


# Gráfico de la derivación de la curva LM a partir del equilibrio en el mercado monetario

    # Dos gráficos en un solo cuadro
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 8)) 


#---------------------------------
    # Gráfico 1: Equilibrio en el mercado de dinero
    
ax1.set(title="Money Market Equilibrium", xlabel=r'M^s / P', ylabel=r'r')
ax1.plot(Y, Ms_MD, label= '$L_0$', color = '#CD5C5C')
ax1.plot(Y, Ms_Y1, label= '$L_1$', color = '#CD5C5C')
ax1.plot(Y, Ms_Y2, label= '$L_2$', color = '#CD5C5C')
ax1.axvline(x = 45,  ymin= 0, ymax= 1, color = "grey")

ax1.axhline(y=35, xmin= 0, xmax= 1, linestyle = ":", color = "black")
ax1.axhline(y=135, xmin= 0, xmax= 1, linestyle = ":", color = "black")
ax1.axhline(y=85, xmin= 0, xmax= 1, linestyle = ":", color = "black")

ax1.text(47, 139, "C", fontsize = 12, color = 'black')
ax1.text(47, 89, "B", fontsize = 12, color = 'black')
ax1.text(47, 39, "A", fontsize = 12, color = 'black')

ax1.text(0, 139, "$r_2$", fontsize = 12, color = 'black')
ax1.text(0, 89, "$r_1$", fontsize = 12, color = 'black')
ax1.text(0, 39, "$r_0$", fontsize = 12, color = 'black')

ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.legend()
 

#---------------------------------
    # Gráfico 2: Curva LM
    
ax2.set(title="LM SCHEDULE", xlabel=r'Y', ylabel=r'r')
ax2.plot(Y, i, label="LM", color = '#3D59AB')

ax2.axhline(y=160, xmin= 0, xmax= 0.69, linestyle = ":", color = "black")
ax2.axhline(y=118, xmin= 0, xmax= 0.53, linestyle = ":", color = "black")
ax2.axhline(y=76, xmin= 0, xmax= 0.38, linestyle = ":", color = "black")

ax2.text(67, 164, "C", fontsize = 12, color = 'black')
ax2.text(51, 122, "B", fontsize = 12, color = 'black')
ax2.text(35, 80, "A", fontsize = 12, color = 'black')

ax2.text(0, 164, "$r_2$", fontsize = 12, color = 'black')
ax2.text(0, 122, "$r_1$", fontsize = 12, color = 'black')
ax2.text(0, 80, "$r_0$", fontsize = 12, color = 'black')

ax2.text(72.5, -14, "$Y_2$", fontsize = 12, color = 'black')
ax2.text(56, -14, "$Y_1$", fontsize = 12, color = 'black')
ax2.text(39, -14, "$Y_0$", fontsize = 12, color = 'black')

ax2.axvline(x=70,  ymin= 0, ymax= 0.69, linestyle = ":", color = "black")
ax2.axvline(x=53,  ymin= 0, ymax= 0.53, linestyle = ":", color = "black")
ax2.axvline(x=36,  ymin= 0, ymax= 0.38, linestyle = ":", color = "black")

ax2.yaxis.set_major_locator(plt.NullLocator())   
ax2.xaxis.set_major_locator(plt.NullLocator())

ax2.legend()

plt.show()


# ### 2.¿Cuál es el efecto de una disminución en la Masa Monetaria ∆Ms < 0? 

# $$ M_s↓ → Ms↓ < M^d  → r↑ $$
# 

# In[15]:


#--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 500             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM

# Definir SOLO el parámetro cambiado
Ms = 150

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[16]:


# Dimensiones del gráfico
y_max = np.max(i)
v = [0, Y_size, 0, y_max]   
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, i, label="LM", color = 'black')
ax.plot(Y, i_Ms, label="LM_Ms", color = '#3D59AB', linestyle = 'dashed')

# Texto agregado
plt.text(45, 76, '∆$M^s$', fontsize=12, color='black')
plt.text(45, 70, '←', fontsize=15, color='grey')

# Título y leyenda
ax.set(title = "Disminución en la Masa Monetaria $(M^s)$", xlabel=r'Y', ylabel=r'r')
ax.legend()


plt.show()


# ### 3.¿Cuál es el efecto de un aumento  ∆k>0?

# $$ k↑ → \frac{k}{j}↑ → M^d > M^s →  r↑ $$

# In[17]:


#--------------------------------------------------
    # Curva LM ORIGINAL

# Parámetros

Y_size = 100

k = 2
j = 1                
Ms = 500             
P  = 20               

Y = np.arange(Y_size)

# Ecuación

def i_LM( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i = i_LM( k, j, Ms, P, Y)

#--------------------------------------------------
    # NUEVA curva LM

# Definir SOLO el parámetro cambiado
k = 6

# Generar la ecuación con el nuevo parámetro
def i_LM_Ms( k, j, Ms, P, Y):
    i_LM = (-Ms/P)/j + k/j*Y
    return i_LM

i_Ms = i_LM_Ms( k, j, Ms, P, Y)


# In[18]:


# Dimensiones del gráfico
y_max = np.max(i)
v = [0, Y_size, 0, y_max]   
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, i, label="LM", color = 'black')
ax.plot(Y, i_Ms, label="LM_Ms", color = '#3D59AB', linestyle = 'dashed')

# Texto agregado
plt.text(45, 130, '∆$K$', fontsize=12, color='black')
plt.text(43, 130, '←', fontsize=15, color='grey')

# Título y leyenda
ax.set(title = "Aumento en la sensibilidad de demanda de dinero $(k)$", xlabel=r'Y', ylabel=r'r')
ax.legend()


plt.show()


# ## Parte 2: Reporte

# El artículo busca responder a la siguiente pregunta: ¿Qué análisis se puede realizar sobre los efectos de la pandemia y la cuarentena respecto a su impacto en la caída del PBI en al año 2020 mediante un modelo macroeconómico? Asimismo, también se busca ahondar en como a pesar de este fuerte impacto en la economía, la recuperación peruana en el año 2021 ha sido una de las más vigorosas en la región tras el fin de la cuarenta y la implementación de políticas que aumentaron la capacidad productiva de la economía. Este artículo enfoca su análisis a partir de la creación de un modelo macroeconómico inspirado en Blanchard (201) que analiza dos sectores: sector 1 referido a los servicios turísticos, gastronómicos, aéreos, etc. que fueron afectados directamente por la cuarentena; y el sector 2 relacionado con viene y servicios indispensables que fueron afectados de manera indirecta por el shock de demanda negativo producido en el sector 1. Mediante el análisis del efecto individual y comparativo de ambos sectores se logra entender como la pandemia afectó algunos sectores en tanto en la demanda y por ende afectando la producción de otros sectores, produciendo una contracción de la producción de manera desigual. 

# Considero que la fortaleza más importante del artículo es como se realizó un análisis del impacto de la cuarentena mediante un modelo macroeconómico que no solo analiza independientemente cada sector y luego el efecto que produce del primero sobre el segundo, sino que también introduce tres subsistemas,  el subsistema de corto plazo, el del equilibrio estacionario y el del tránsito hacia el equilibrio estacionario, que mediante la estática comparativa logran obtener distintas explicaciones para explicar la caída del PBI. Por medio del análisis del efecto individual y comparativo de ambos sectores se logra entender como la pandemia afectó algunos sectores en la demanda y el precio, y por ende afectando la producción de otros sectores, produciendo una contracción de la producción de manera desigual y un choque combinado de oferta y demanda.  Todo este análisis se logra gracias a la implementación de un modelo macroeconómico que nos permite recrear la realidad utilizando ecuaciones matemáticas para deducir y simular la relación entre la demanda, oferta, precios y la producción. Asimismo, el uso de la estática comparativa en el análisis es muy útil para comparar los dos puntos de desequilibrio y equilibrio en la economía: pre pandemia, durante la pandemia y ‘post pandemia’. Sin embargo, el mismo uso del modelo y la estática comparativa también tienen sus limitaciones. La creación y uso de modelos macroeconómicos sueles enfocarse en variables netamente cuantificables y puede entonces discriminar otras variables que jugaron un papel importante en el impacto de la cuarentena en la economía como fue las políticas públicas en el terreno de la economía (alivio fiscal y estímulo monetario) o también el papel de agente racionales que podrían haber igualado el precio de equilibrio. De la misma manera, la estática comparativa es muy útil para comparar estados de equilibrio, pero en este caso no analiza profundamente todos los procesos posibles que llevan al ajuste de equilibrio. En general, hubo otros enfoques y variables que no se tomaron en cuenta y que pudieron haber ilustrado un panorama más inclusivo y explicativo. Por ejemplo, el autor menciona como no se tuvo en cuenta el efecto de la cuarentena sobre el producto exponencial debido a que se enfocaron en la variable de demanda. 

# La contribución del artículo es el enfoque macroeconómico keynesiano que nos permite entender el impacto y el proceso de recuperación de uno de los eventos más significativos del siglo XXI. Debido a que ahora vivimos en un mundo cada vez más interconectado e interdependiente, no son solo los eventos y efectos positivos de la economía los que se comparten, sino también las crisis económicas.  Incluso dentro de una sola economía, todos los sectores están interconectados y sufren las consecuencias de una recesión debido a un shock negativo en alguno de los sectores como fue en el caso de Perú. De igual modo, podemos apreciar como a pesar de todos los problemas políticos existente, aunque no olvidando que el modelo tampoco tomo en cuenta las variables políticas, en el 2021 logramos implementar un proceso de recuperación riguroso sobre la producción y empleo que permitió un exponencial crecimiento crediticio dirigido a las empresas del sector productivo.  En un sentido, hubo un ‘milagro del crédito’ nunca antes visto en el país durante una época de contracción económica. Es así como a través del modelo se permite ver como el crédito bancario permitió el restablecimiento del PBI potencial previo a la pandemia.

# Y continuando con el tema de la recuperación post-pandemia, otros autores señalan otras variables y políticas que se deben tomar en cuenta para la recuperación económica peruana. De acuerdo a Ricardo Pérez (2020), señala que no solo va importar el crédito bancario, sino también la capacidad de las empresas de reinsertarse o reinventarse bajo las nuevas condiciones externas e internas. Asimismo, tanto este autor como el exministro de Economía y Finanzas Alonso Segura (2021), señalan que no solo se tendrá que tener en cuenta políticas económicas, sino también sanitarias y políticas. En el caso de Segura, el menciona va ser necesario elevar los ingresos públicos ya que el Estado peruano debe ser más fuerte y tener mayor presencia para enfrentar los problemas económicos. Asimismo, él señala que se tiene que distinguir entre la recuperación de actividad económica con el crecimiento sostenible, este último necesitando más tiempo para poder recuperarse. 

# BIBLIOGRAFÍA:
# Luyo, R. P. (2020, mayo 19). Planeamiento estratégico empresarial en la nueva normalidad. RPP. https://rpp.pe/columnistas/ricardoperezluyo/planeamiento-estrategico-empresarial-en-la-nueva-normalidad-noticia-1266381
# Los retos de la economía pospandemia. (2021, marzo 31). PuntoEdu PUCP; Pontificia Universidad Católica del Perú. https://puntoedu.pucp.edu.pe/voces-pucp/los-retos-de-la-economia-pospandemia/

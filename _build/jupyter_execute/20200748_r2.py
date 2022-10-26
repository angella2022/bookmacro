#!/usr/bin/env python
# coding: utf-8

# # Reporte 2:

# In[1]:


import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sympy import *
import pandas as pd
#from causalgraphicalmodels import CausalGraphicalModel


# ## Parte 1: Código

# Trabajo conjunto con Maria Fernanda Carrillo - 20201596

# ## a. Modelo Ingreso-Gasto: la Curva IS

# ## Derive paso a paso la curva IS matemáticamente (Y=DA)

# La curva IS se deriva de la igualdad entre el ingreso (Y)  y la demanda agregada (DA):
# $$ Y = C + I + G + X - M$$
# 
# Considerando que: 
# 
# $$ C = C_0 + bY^d $$
# $$ I = I_0 - hr $$
# $$ G = G_0 $$
# $$ X = X_0 $$
# $$ M = mY^d $$
# 
# $$ Y^d = 1 - t $$

# Para llegar al equilibrio Ahorro-Inversión, debemos crear un artificio. Por ende pasamos a restar la tributación (T) de ambos miembros de la igualdad.
# 
# $$ Y - T = C + I - T + G + X - M$$
# $$ Y^d = C + I - T + G + X - M$$

# Esta igualdad se puede reescribir de la siguiente forma:
# $$ (Y^d - C) + (T-G) + (M-X) = I = S$$

# Las tres partes de la derecha constituyen los tres componentes del ahorro total (S) : ahorro privado/interno (Sp), ahorro del gobierno (Sg)  y ahorro externo (Se):
# $$ S = Sp + Sg + Se$$

# Entonces, el ahorro total es igual a la inversión:
# $$ Sp + Sg + Se = I$$
# $$ S(Y) = I(r)$$

# Hacemos reemplazos tomando en cuenta las ecuaciones básicas, se obtiene que:
# $$ (Y^d - C_0 - bY^d) + (T - G_0) + (mY^d - X_0) = I_0 -hr$$
# $$ Y^d -(C_0 + bY^d) +[tY - G_0] + [mY^d - X_0] = Io - hr$$
# $$ [1-(b-m)(1-t)]Y-(C_0 + G_0 + X_0) = Io - hr$$

# Continuamos derivando:
# $$ hr = Io -[1-(b-m)(1-t)Y - (C_0 + G_0 + X_0)]$$
# 
# $$ hr = (C_0 + G_0 + I_0 + X_0) - (1-(b-m)(1-t))Y$$

# La curva IS se puede expresar con una ecuación donde la tasa de interés es una función del ingreso:
# 
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1-(b-m(1-t)}{h}(Y)$$

# Es en esta condición de igual en el cual el mercado de bienes hay una relación negativa entre el producto y el ingreso

# ## Encuentre ∆r/∆Y

# La ecuación alcanzada en función de la tasa de interés (r) es entonces: 
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1-(b-m(1-t)}{h}(Y)$$

# La curva IS también se puede entender como:
# 
# $$\frac{1}{h}(C_0 + G_0 + I_0 + X_0) => a1$$
# 
# $$ \frac{1-(b-m(1-t)}{h}(Y) => a2$$

# Esto va ser igual a como cambia la tasa de interés ante una variación en el producto:
# 
# $$\frac{∆r}{∆Y} = -\frac{1-(b-m)(1-t)}{h} < 0$$
# $$\frac{∆r}{∆Y} = (-)*(+) < 0$$
# $$\frac{∆r}{∆Y} = (-) < 0$$

# ## Explique cómo se derive la Curva IS (Y=DA)

# Partimos del hecho de que el ingreso está determinado por el consumo, inversión, el gasto y exportaciones netas. Ahora, lo que queremos es llegar a un equilibrio llamado ‘ahorro inversión’ y para ello debemos crear un artificio (Y - T  (Yd)= C + I + G + X - M - T).  Luego vamos a agrupar por sectores: interno, gobierno y externo (Yd - C) + (T-G) + (M-X). Con estas tres formas de gobierno obtenemos la inversión (S = I). Luego continuamos derivando y remplazando las ecuaciones hasta llegar a la ecuación que  expresa la tasa de interés en función del ingreso. Entonces a partir de la ecuación de la Demanda Agregada podemos llegar a saber que el equilibrio en ahorro es igual a la inversión y que en una condición de igual en el mercado de bienes hay una relación negativa entre el producto y el ingreso que se puede graficar a partir del equilibrio DA=Y en una recta de 45°.

# - Demanda Agregada

# In[2]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)

#--------------------------------------------------
# Recta de 45°

a = 2.5 

def L_45(a, Y):
    L_45 = a*Y
    return L_45

L_45 = L_45(a, Y)

#--------------------------------------------------
# Segunda curva de ingreso de equilibrio

    # Definir cualquier parámetro autónomo
Go = 90

# Generar la ecuación con el nuevo parámetro
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_G = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


# - Curva IS

# In[3]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

IS = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[4]:


# Gráfico de la derivación de la curva IS a partir de la igualdad (DA = Y)

    # Dos gráficos en un solo cuadro (ax1 para el primero y ax2 para el segundo)
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
    # Gráfico 1: ingreso de Equilibrio
ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.plot(Y, DA_IS_K, label = "DA_0", color = "C0") 
ax1.plot(Y, DA_G, label = "DA_1", color = "C0") 
ax1.plot(Y, L_45, color = "#8367e8") 

ax1.axvline(x = 80, ymin= 0, ymax = 0.80, linestyle = ":", color = "grey")
ax1.axvline(x = 70,  ymin= 0, ymax = 0.70, linestyle = ":", color = "grey")

ax1.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
ax1.text(2.5, -3, '$◝$', fontsize = 30, color = 'black')
ax1.text(72, 0, '$Y_0$', fontsize = 12, color = 'black')
ax1.text(84, 0, '$Y_1$', fontsize = 12, color = 'black')
ax1.text(77, 207, 'E_1', fontsize = 12, color = 'black')
ax1.text(85, 207, 'DA(r_1)', fontsize = 12, color = 'black')
ax1.text(67, 183, 'E_0', fontsize = 12, color = 'black')
ax1.text(83, 169, 'DA(r_0)', fontsize = 12, color = 'black')


ax1.set(title = "Derivación de la curva IS a partir del equilibrio $Y=DA$", xlabel = r'IS', ylabel = r'DA')
ax1.legend()
#---------------------------------
    # Gráfico 2: Curva IS

ax2.yaxis.set_major_locator(plt.NullLocator())   
ax2.xaxis.set_major_locator(plt.NullLocator())

ax2.plot(Y, IS, label = "IS", color = "#e88367") 

ax2.axvline(x = 80, ymin= 0, ymax = 0.99, linestyle = ":", color = "grey")
ax2.axvline(x = 70,  ymin= 0, ymax = 0.99, linestyle = ":", color = "grey")
plt.axhline(y = 151.5, xmin= 0, xmax = 0.69, linestyle = ":", color = "grey")
plt.axhline(y = 143, xmin= 0, xmax = 0.79, linestyle = ":", color = "grey")

ax2.text(72, 128, '$Y_0$', fontsize = 12, color = 'black')
ax2.text(80, 128, '$Y_1$', fontsize = 12, color = 'black')
ax2.text(1, 153, '$r_0$', fontsize = 12, color = 'black')
ax2.text(1, 145, '$r_1$', fontsize = 12, color = 'black')
ax2.text(72, 152, 'E_0', fontsize = 12, color = 'black')
ax2.text(83, 144, 'E_1', fontsize = 12, color = 'black')


ax2.legend()

plt.show()


# ## b. La Curva IS o el equilibrio Ahorro- Inversión

# Entonces, el ahorro total es igual a la inversión:
# $$ (Y^d - C) + (T-G) + (M-X) = S = I$$
# $$ Sp + Sg + Se = I$$
# $$ S(Y) = I(r)$$

# Hacemos reemplazos tomando en cuenta las ecuaciones básicas se obtiene que:
# $$ (Y^d - C_0 - bY^d) + (T - G_0) + (mY^d - X_0) = I_0 -hr$$
# 
# $$ Y^d -(C_0 + bY^d) +[tY - G_0] + [mY^d - X_0] = Io - hr$$
# 
# $$ [1-(b-m)(1-t)]Y-(C_0 + G_0 + X_0) = Io - hr$$
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1-(b-m(1-t)}{h}(Y)$$

# In[5]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

IS_IS = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)
# Gráfico de la curva IS

# Dimensiones del gráfico
y_max = np.max(IS_IS)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, IS_IS, label = "IS_IS", color = "#4d25de") #Demanda agregada
ax.text(98, 130, 'IS', fontsize = 14, color = 'black')
# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Título, ejes y leyenda
ax.set(title = "Curva IS de Equilibrio en el Mercado de Bienes", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# ## c. Desequilibrios en el mercado de bienes

# In[6]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r1 = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[7]:


#Dimensiones:
y_max = np.max(r1)
fig, ax = plt.subplots(figsize=(10, 8))
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())
ax.plot(Y, r1, label = "IS", color = "C1") 
#Lineas punteadas:
ax.axvline(x = 70.5, ymin= 0, ymax = 0.46, linestyle = ":", color = "grey")
ax.axvline(x = 54,  ymin= 0, ymax = 0.46, linestyle = ":", color = "grey")
ax.axvline(x = 37,  ymin= 0, ymax = 0.46, linestyle = ":", color = "grey")
plt.axhline(y = 165, xmin= 0, xmax = 0.693, linestyle = ":", color = "grey")
#Leyenda:
ax.text(71, 128, '$Y_B$', fontsize = 12, color = 'black')
ax.text(55, 128, '$Y_A$', fontsize = 12, color = 'black')
ax.text(38, 128, '$Y_C$', fontsize = 12, color = 'black')
ax.text(1, 167, '$r_A$', fontsize = 12, color = 'black')
ax.text(71, 166, 'B', fontsize = 12, color = 'black')
ax.text(55, 166, 'A', fontsize = 12, color = 'black')
ax.text(38, 166, 'C', fontsize = 12, color = 'black')

#Coordenadas de exceso:

ax.text(70,170, 'Exceso  de  Ofertas', fontsize = 14, color = 'black')
ax.text(34,150, 'Exceso  de  Demanda', fontsize = 14, color = 'black')

ax.set(title = "Equilibrio y desequilibrio en el Mercado de Bienes-Exceso de Demanda y Oferta", xlabel = 'Y', ylabel= 'r')
ax.legend()

plt.show()


# Si la Curva IS representa los puntos donde hay un equilibrio en el mercado de bienes, los puntos fuera de la curva señalan un desequilibrio en el mercado. Al lado derecho se encuentra el exceso de demanda (Inversión (I) < Ahorro (S)), mientras que al lado izquierdo se encuentra el desequilibrio por execeso de oferta. 

# ## d. Movimientos de la curva IS

# ## Política Fiscal Contractica con caída del Gasto del Gobierno

# ### Intuición: (∆G < 0)

# $$ Go↓ → G↓ → DA↓ → DA < Y → Y↓$$

# ### Gráfico

# In[8]:


#--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

G_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # NUEVA curva IS
    
# Definir SOLO el parámetro cambiado
Go = 52

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

G_Go = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[9]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(G_G)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, G_G, label = "G_G", color = "black") #IS orginal
ax.plot(Y, G_Go, label = "G_Go", color = "C1", linestyle = 'dashed') #Nueva IS

# Texto agregado
plt.text(40, 162, '∆Go', fontsize=12, color='black')
plt.text(40, 159, '↓', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title = "Caída del Gasto de Gobierno $(G_0)$", xlabel= 'Y', ylabel= 'Go')
ax.legend()

plt.show()


# ## Política Fiscal Expansiva con caída de la Tasa de Impuesto 

# ### Intuición: (∆t < 0)

# $$ t↓ → Co↑ -> DA↑ → DA > Y → Y↑ $$
# $$ t↓ → M↑ → DA↓ → DA < Y → Y↓ $$

# ### Grafico:

# In[10]:


#--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.5

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

t1 = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # NUEVA curva IS
    
# Definir SOLO el parámetro cambiado
t = 0.3

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h,Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

IS_t = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[11]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(t1)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, t1, label = "t", color = "black") #IS orginal
ax.plot(Y, IS_t, label = "IS_t", color = "C1", linestyle = 'dashed') #Nueva IS

# Texto agregado
plt.text(51, 162, '∆t', fontsize=12, color='black')
plt.text(54, 158, '→', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title = "Caída de la Tasa de Impuesto $(t)$", xlabel= 'Y', ylabel= 't')
ax.legend()

plt.show()


# ## Caída de la Propensión Marginal a Consumir

# ### Intuición: (∆b < 0)

# $$ b↓ → C↓  → DA↓  → DA < Y → Y↓  $$

# ### Gráfico:

# In[12]:


#--------------------------------------------------
    # Curva IS ORIGINAL

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.5

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

b1 = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # NUEVA curva IS
    
# Definir SOLO el parámetro cambiado
b = 0.4

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h,Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

IS_b = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[13]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(b1)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, b1, label = "b", color = "black") #IS orginal
ax.plot(Y, IS_b, label = "IS_b", color = "C1", linestyle = 'dashed') #Nueva IS

# Texto agregado
plt.text(58, 143, '∆b', fontsize=12, color='black')
plt.text(58, 139, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title = "Caída de la Propensión Marginal a Consumir $(t)$", xlabel= 'Y', ylabel= 't')
ax.legend()

plt.show()


# ## PARTE 2: Reporte

# El artículo busca responder a la siguiente pregunta: ¿Qué análisis se puede hacer desde un enfoque monetarista sobre el predominio del régimen fiscal y su impacto en la inflación en el Perú durante la segunda mitad del siglo XX? En este sentido, los autores explican los procesos de inflación e hiperinflación de la década de los 70 y lo que ellos consideran malas políticas implementadas por parte de esos gobiernos, para luego compararlo con el cambio de régimen y enfoque económico durante la época de los 90 con Fujimori, época donde hubo mayor estabilidad y menor inflación. 

# La fortaleza de este artículo es su enfoque institucional y cultural. Hay no solo un análisis de las mismas políticas fiscales y sus efectos, sino que son explicadas a partir del tipo de régimen que había (de los militares a una democracia a una autocracia) y la cultura y opinión política y económica (de una percepción negativa del mercado a un rechazo al intervencionismo del Estado). Estos factores externos a la misma economía jugaron un papel sumamente importante al momento de implementar los programas económicos, y específicamente para explicar el predominio de las políticas fiscales sobre las monetarias previo a la década de los 90. Asimismo, considero que para entender mejor por qué las anteriores políticas fiscales empeoraron la inflación fue muy útil un enfoque monetario ya que al final fueron las políticas monetarias las que mejoraron la estabilidad económica. 

# Una debilidad sobre el artículo sería lo muy técnico que llega a ser a veces al momento de explicar los programas políticos implementados durante los distintos gobiernos. A pesar del gran enfoque en variables extras que influyeron en la implementación de este tipo de políticas y lo útil que es el análisis desde el enfoque monetario, aun así tiene sus limitaciones ya que se enfoca en ciertas variables más que otras como son las reservas internacionales, el tipo de cambio y los precios domésticos. Estos son importantes, pero podría haber otras variables igual de importantes que también ameritan mayor análisis. 

# Considero que los autores contribuyen a la pregunta de investigación en tanto logran hacer un detallado recuento y análisis de los programas fiscales y monetarios, y específicamente contribuyen en el análisis de la década de los 90.  Uno pensaría que a pesar de que fue una época muy compulsiva debido al contexto social, cultural y político, las posibilidades de una estabilización económica y reducción de la inflación sería imposible. Sin embargo, gracias al análisis de los autores, podemos comprender mejor que es debido a un enfoque monetario, un alejamiento de las políticas intervencionistas así como una liberalización de la economía lo que permitieron que a pesar de no estar en una democracia, hubiera mayor estabilidad económica. Asimismo, logramos entender lo fundamental que fue le apoyo popular a esta estabilidad luego de años de hiperinflación y un cambio en el esquema constitucional que otorgó mayor autonomía y responsabilidad a otros actores no estatales como el Banco Central.

# Y finalmente otro tema crucial del artículo y la discusión es cuán intervencionista este se sitúa en materia económica. Ya que si bien el enfoque de Martínez en el artículo “Intervencionismo estatal en América Latina: los casos de Colombia, Perú y Venezuela” considera que de hecho fue necesaria para impulsar el desarrollo en los casos estudiados del Perú y Colombia; es que conviene con Martinelli y Vega en tanto critica la excesiva intervención, pues considera fue la razón primordial de la situación actual de Venezuela: un colapso económico recesivo. Por lo que contraponiendo ambas visiones es que resulta lógico el que se ha de ver al Estado como un actor fundamental en el sistema, no obstante, su participación dentro del mercado ha de ser regulada, mas no limitada en extremo, así como tampoco imprecisada a fin de que esta acapare por completo la economía. 

# Fuente: https://revistaestudiospoliticaspublicas.uchile.cl/index.php/REPP/article/view/51727/65382 

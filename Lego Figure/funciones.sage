# -*- coding: utf-8 -*-

#funciones de la practica 1
#--------------------------------------------------------------------------------------

def vector_puntos(P,Q):
    
    # Input: matrices columna con los puntos P y Q en coordenadas homogéneas.
    # Output: matriz columna con el vector PQ, diferencia de los puntos Q y P.
    
    return (P.augment(Q))*matrix([[-1],[1]])

def producto_escalar(U,V):
    
    # Input: matrices columna con los vectores U y V en coordenadas homogéneas.
    # Output: matriz 1x1 con su producto escalar.    
    
    return (transpose(U)*V)[0,0]

def modulo(U):
    
    # Input: matriz columna con un vector U en coordenadas homogéneas.
    # Output: el escalar que da su módulo.    
    
    return sqrt(producto_escalar(U,U))

def vector_unitario(U):
    
    # Input: matriz columna con un vector U en coordenadas homogéneas.
    # Output: matriz columna con el vector unitario en la misma dirección y sentido que U.    
    
    return U/modulo(U)

def coseno_del_angulo_que_forman(U,V):
    
    # Input: matrices columna con los vectores U y V en coordenadas homogéneas.
    # Output: escalar que representa el coseno del ángulo que forman.        
    
    return producto_escalar(U,V)/(modulo(U)*modulo(V))

def producto_vectorial(U,V):
    
    # Input: matrices columna con los vectores U y V en coordenadas homogéneas.
    # Output: matriz columna con su producto vectorial.        
    
    W0 = U[1,0]*V[2,0] - U[2,0]*V[1,0]
    W1 = -U[0,0]*V[2,0] + U[2,0]*V[0,0]
    W2 = U[0,0]*V[1,0] - U[1,0]*V[0,0]
    return matrix([[W0],[W1],[W2],[0]])

def dibujar_punto(P,**kwds):
    
    #Input: punto P con coordenadas homogéneas 3D y en columna
    #Output: gráfica del punto
    
    return points(P[0:3].column(0),**kwds) 

def dibujar_linea(P,Q,**kwds):
    
    #Input: puntos P y Q con coordenadas homogéneas 3D y en columna
    #Output: gráfica de la linea
    
    return line([P[0:3].column(0),Q[0:3].column(0)],**kwds)

def dibujar_poligono(vertices,**kwds):
    
    # Input: matriz con los vértices del polígono en homogénea 3D y por columnas
    # Output: gráfica del polígono
    
    return polygon(tuple(vertices[0:3].transpose()),**kwds)

def dibujar_poligono_delimitado(vertices,**kwds):
    
    # Input: matriz con los vértices del polígono en homogénea 3D y por columnas
    # Output: gráfica del polígono
    V=list(vertices[0:3].transpose())
    dib=polygon(V,**kwds)
    V.extend(vertices[0:3,0].transpose())
    dib+=line3d(V,color='black',thickness=8)
    return dib

def dibujar_mallado_poligonal(vertices,caras,**kwds):
    
    # Input: matriz con los vértices del polígono en homogénea 3D y por columnas, matriz de caras.
    # Output: gráfica del mallado. 
    
    mallado = 0
    for cara in caras:        
        indices = [n for n in cara if n>-1]
        vertices_de_la_cara = vertices[:,indices]
        mallado = mallado + dibujar_poligono(vertices_de_la_cara,**kwds)
    return mallado

def dibujar_mallado_poligonal_delimitado(vertices,caras,**kwds):
    
    # Input: matriz con los vértices del polígono en homogénea 3D y por columnas, matriz de caras.
    # Output: gráfica del mallado, con todas las aristas resaltadas. 
    
    mallado = 0
    for cara in caras:        
        indices = [n for n in cara if n>-1]
        vertices_de_la_cara = vertices[:,indices]
        mallado = mallado + dibujar_poligono_delimitado(vertices_de_la_cara,**kwds)
    return mallado



def dibujar_vertices_poligono(vertices,**kwds):
    
    # Input: matriz con los vértices del polígono en homogénea 3D y por columnas
    # Output: gráfica del polígono
    return point3d((vertices.delete_rows([3])).transpose(),size=20,**kwds)

def dibujar_etiquetas_vertices(vertices,**kwds):
    
    # Input: matriz con los vértices del polígono en homogénea 3D y por columnas
    # Output: gráfica con la etiqueta de los vértices en su posición 3D.    
    
    M=vertices.delete_rows([3])
    M=M.transpose()
    l=0
    for k in range(M.nrows()):
        l+=text3d(str(k),M.row(k),**kwds)
    show(l)
    return


def vertices_no_alineados(vertices): 
    
    # Input: matriz con 3 vértices por columnas y en coordenadas homogéneas.
    # Output: 1 si los vértices no están alineados; 0 en caso contrario. 
    
    U = vector_puntos(vertices[:,1],vertices[:,2])
    V = vector_puntos(vertices[:,1],vertices[:,0])
    UxV = producto_vectorial(U,V)
    
    if modulo(UxV)!=0: #Los vértices NO están alineados
        return 1
    else: #Los vértices SI están alineados
        return 0


def es_cara_no_trivial(vertices,*i):
    
    # Input: matriz con los vértices por columna y en coordenadas homogéneas de una cara
    # Output: lista [] si están alineados, o bien [0,1,k] si los vértices de esos índices no están alineados (se asume P0 y P1 distintos entre sí).
    
    L = []
    num_vertices = vertices.ncols()
    
    for j in range(num_vertices-2):
        V = vertices[:,[0,1,j+2]]
        if vertices_no_alineados(V) == 1: #vértices no alineados -> cara no trivial
            print('Cara no trivial')
            L = [0,1,j+2]
            return L
    print('Cara trivial')
    return L #L=[], luego todos los vértices alineados    
    
def transformacion_de_coordenadas(vertices):

    # Input: matriz con los vértices por columna y en coordenadas homogéneas de una cara, con P0,P1 y Pn determinando un plano.
    # Output: matriz de la transformación de coordenadas del SRU al nuevo SR.
    
    
    P0 = vertices[:,0]
    P1 = vertices[:,1]
    Pn = vertices[:,-1]
    U1 = vector_puntos(P0,P1)         # vector P0P1 = P1-P0
    V1 = vector_puntos(P0,Pn)         # vector P0Pn = Pn-P0
    W1 = producto_vectorial(U1,V1)    # vector W = U x V

    U = vector_unitario(U1)                  # vector U normalizado
    W = vector_unitario(W1)                  # vector W normalizado
    V = producto_vectorial(W,U)              # vector v = w x u

    a = producto_escalar(P0,U)
    b = producto_escalar(P0,V)
    c = producto_escalar(P0,W)

    return matrix([[U[0,0],U[1,0],U[2,0],-a],[V[0,0],V[1,0],V[2,0],-b],[W[0,0],W[1,0],W[2,0],-c],[0,0,0,1]])


def mallado_trivialidad(vertices,caras):
    
    # Input: matriz con los vértices por columna y en coordenadas homogéneas, y matriz de caras.
    # Output: 0 si hay caras triviales, 1 en caso de que no haya caras triviales. Adicionalmente, se imprime por pantalla el carácter trivial o no de cada cara.
    
    i = 0
    bool=1
    for c in caras:
        indices = [n for n in c if n>-1]
        vertices_de_la_cara = vertices[:,indices]
        print('La cara ',i,': ',c,':')
        L = es_cara_no_trivial(vertices_de_la_cara,i)
        if L==[]:
            bool=0
        i +=1
        print(L)   
    return bool


def es_cara_plana(vertices):
    
    # Input: matriz con los vértices por columna y en coordenadas homogéneas de una cara.
    # Output: 1, si es una cara plana; 0, en caso contrario.
    
    M = transformacion_de_coordenadas(vertices)
    vertices_nuevoSR = M*vertices    
    if vertices_nuevoSR[2]==0:      # Si la tercera fila de los vértices transformados es nula entonces: 
        #print('La cara es plana')           
        return 1
    else:
        #print('La cara no es plana')
        return 0
    
def mallado_consistente_planaridad(vertices,caras):
    
    # Input: matriz con los vértices por columna y en coordenadas homogéneas de un mallado poligonal así como la matriz de las caras de dicho mallado.
    # Output: 1, si todas las caras son planas; 0, en caso contrario. Adicionalmente, imprime por pantalla si cada cara es plana o no.
    
    bool=1
    i = 0
    for c in caras:
        indices = [n for n in c if n>-1]
        vertices_de_la_cara = vertices[:,indices]
        if es_cara_plana(vertices_de_la_cara)==0:
            print('La cara ',i,' no es plana:',c)
            bool = 0
            #return 0
        else:
            print('La cara ',i,' es plana',c)
        i+=1
    #print('Todas las caras son planas.')
    return bool


def ecuaciones_implicitas(vertices2D):

    # Input: matriz con los vértices 2D por columna y en coordenadas homogéneas de una cara que ya sabemos que es plana.
    # Output: matriz donde cada fila representa los coeficientes de la ecuación implícita de la recta que contiene a cada una de las aristas de la cara.
    
    M = matrix(RR,vertices2D.ncols(),3)       
    
    for i in range(vertices2D.ncols()):
        P = vertices2D[:,i-1]
        Q = vertices2D[:,i]
        vector_director = vector_puntos(P,Q)                 
        vector_normal = matrix([[-vector_director[1,0]],[vector_director[0,0]],[0]])   # vector normal a la recta
        M[i] = [vector_normal[0,0], vector_normal[1,0], -producto_escalar(P,vector_normal)]

    return M

def es_cara_simple_convexa(vertices2D):
    
    # Input: matriz con los vértices 2D por columna y en coordenadas homogéneas de una cara.
    # 1 si es una cara simple-convexa; 0 en caso contrario.
    
    M = ecuaciones_implicitas(vertices2D)
    indicesvertices=[0..vertices2D.ncols()-1]
    for i in indicesvertices:
        signos = []
        for j in indicesvertices:
            if j==i or j==indicesvertices[i-1]: #se está comparando con la arista de extremos i-1 e i
                continue
            else:
                s = sign((M[i,:]*vertices2D[:,j])[0,0])
                if s==0:
                    if producto_escalar(vector_puntos(vertices2D[:,indicesvertices[i-1]],vertices2D[:,j]),vector_puntos(vertices2D[:,i],vertices2D[:,j]))<0:
                        #print('El vértice %s influye en que la cara no sea simple, respecto de la arista %s'%(j,i))
                        return 0 #no es simple
                else:
                    signos.append(s)
        sentido_anti_normal = signos.count(-1)
        sentido_normal = signos.count(1)
        if sentido_anti_normal!=0 and sentido_normal!=0:
            #print('Según la arista que llega al vértice %s falla la convexidad'%i)
            return 0 # no es convexa
        
    #print('La cara es simple-convexa')    
    return 1

def mallado_consistente_simple_convexidad(vertices,caras):
    
    # Input: matriz con los vértices por columna y en coordenadas homogéneas de un mallado poligonal así como la matriz de las caras de dicho mallado.
    # Output: 1, si todas las caras son simple-convexas; 0, en caso contrario. Adicionalmente, imprime por pantalla si cada cara es simple-convexa o no.
    
    bool=1
    i = 0
    for c in caras:
        indices = [n for n in c if n>-1]
        vertices_de_la_cara = vertices[:,indices]
        M = transformacion_de_coordenadas(vertices_de_la_cara)
        Vt = M*vertices_de_la_cara
        V2D = Vt.delete_rows([2])        
        if es_cara_simple_convexa(V2D)==0:
            print('La cara ',i,' no es simple-convexa:',c)
            bool=0
        else:
            print('La cara ',i,' es simple-convexa:',c)
        i += 1
    
    return bool 



def baricentro(vertices):  
    
    # Input: matriz de vértices por columnas con coordenadas homogéneas de una cara (que se asume plana).
    # Output: baricentro de la cara.
    
    
    V=vertices
    B = matrix([[sum(V.row(0))],[sum(V.row(1))],[sum(V.row(2))],[sum(V.row(3))]])/sum(V.row(3))
    return B    

def triangulacion_baricentrica(vertices,caras):
    
    # Input: matrices de vértices y caras de un mallado (se asume las caras no triviales, planas y simple-convexas).
    # Output: nuevo conjunto de vértices y caras propio de una triangulación baricéntrica de cada cara del modelo original.
    
    nuevas_caras=[]
    nuevos_vertices=matrix(RR,vertices)
    cont=vertices.ncols()
    for cara in caras:
        indices=[n for n in cara if n>-1]
        vert_cara=vertices[:,indices]
        B=baricentro(vert_cara)
        nuevos_vertices=nuevos_vertices.augment(B)
        for v in range(len(indices)-1):
            nuevas_caras.append([indices[v],indices[v+1],cont])
        nuevas_caras.append([indices[-1],indices[0],cont])
        cont=cont+1
    nuevas_caras=matrix(nuevas_caras)
    show('Nuevo conjunto de vértices:',nuevos_vertices)
    show('Nuevo conjunto de caras:',nuevas_caras)
    return [nuevos_vertices,nuevas_caras]


def comprobar_coherencia_aristas(vertices,caras):
    
    # Input: matrices de vértices y caras de un mallado.
    # Output: 1, si cada arista se recorre en ambos sentidos; 0, en otro caso. No comprueba si una arista aparece más de 2 veces. Devuelve también la matriz de adyacencia.
    
    orden=vertices.ncols()
    M=matrix(ZZ,orden,orden,0)
    for c in caras:
        indices=[v for v in c if v>-1]
        for n in range(len(indices)):
            M[indices[n-1],indices[n]]+=1
    if M.is_symmetric(): # si no es simétrica, hay aristas que no se recorren en ambos sentidos; si la norma infinito es mayor que 1, hay aristas que se usan más de una vez en el mismo sentido
        print(1) 
        return M # se devuelve la matriz de adyacencia del grafo subyacente
    print(0)
    return M # se devuelve la matriz de adyacencia del grafo subyacente


def ecuacion_implicita_plano(vertices):
    
    # Input: matriz de vértices por columnas en coordenadas homogéneas de una cara plana.
    # Output: una matriz fila con los coeficientes de la ecuación implícita que define el plano que contiene la cara, tomando como normal el producto vectorial P0P1 x P0Pn.

    M = matrix(RR,1,4)       
    P0 = vertices[:,0]
    P1 = vertices[:,1]
    Pn = vertices[:,-1]
    U1 = vector_puntos(P0,P1)         # vector P0P1 = P1-P0
    V1 = vector_puntos(P0,Pn)         # vector P0Pn = Pn-P0
    W1 = producto_vectorial(U1,V1)    # vector N=W = U x V
    M[0] = [W1[0,0], W1[1,0], W1[2,0], -producto_escalar(P0,W1)]

    return M

def genero(vertices,caras):
    
    # Input: matrices de vértices y caras de un mallado consistente que encierra una cavidad.
    # Output: vértices, aristas, caras y género de la superficie cerrada que delimita.
    
    v=vertices.ncols()
    c=caras.nrows()
    a=0
    for cara in caras:
        indices = [n for n in cara if n>-1]
        a=a+len(indices)
    a=a/2
    print('Tiene %s vértices, %s aristas, %s caras y género %s'%(v,a,c,1-(v-a+c)/2))
    return [v,a,c]
    
    
    
    
#funciones de la practica 2
#------------------------------------------------------------------------

# Prontuario de la Práctica 2

def traslacion(D): 
    
    #INPUT: vector D en coordenadas homogéneas
    #OUTPUT: matriz de traslación
         
    return matrix([[1,0,0,D[0,0]],[0,1,0,D[1,0]],[0,0,1,D[2,0]],[0,0,0,1]]) 

def rotacion(eje,angulo):
    
    #INPUT: eje='x', 'y' o 'z'
    #       angulo en radianes
    #OUTPUT: matriz de rotación
    
    if eje == 'x':
        M= matrix([[1,0,0,0],[0,cos(angulo),-sin(angulo),0],[0,sin(angulo),cos(angulo),0],[0,0,0,1]])
    elif eje == 'y':
        M= matrix([[cos(angulo),0,sin(angulo),0],[0,1,0,0],[-sin(angulo),0,cos(angulo),0],[0,0,0,1]])    
    else:
        M= matrix([[cos(angulo),-sin(angulo),0,0],[sin(angulo),cos(angulo),0,0],[0,0,1,0],[0,0,0,1]])   
        
    return M

def rotacion_coseno(eje,coseno,seno):
    
    #INPUT: eje='x', 'y' o 'z'
    #       angulo en radianes
    #OUTPUT: matriz de rotación
    
    if eje == 'x':
        M= matrix([[1,0,0,0],[0,coseno,-seno,0],[0,seno,coseno,0],[0,0,0,1]])
    elif eje == 'y':
        M= matrix([[coseno,0,seno,0],[0,1,0,0],[-seno,0,coseno,0],[0,0,0,1]])    
    else:
        M= matrix([[coseno,-seno,0,0],[seno,coseno,0,0],[0,0,1,0],[0,0,0,1]])   
        
    return M

def escalado(r):  
    
    #INPUT: r=[rx,ry,rz]
    #OUTPUT: matriz de rotación
         
    return matrix([[r[0],0,0,0],[0,r[1],0,0],[0,0,r[2],0],[0,0,0,1]]) 

def simetria(N):
    
    #INPUT: N vector normal al plano de simetría en coordenadas homogéneas
    #OUTPUT: matriz de simetría
    
    N = N/norm(N)
    I = matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    
    return I-2*N*N.transpose()

def sesgo(direccion,ang1,ang2):
    
    #INPUT: direccion ='x', 'y' o 'z'
    #       angulos ang1 y ang2  en radianes
    #OUTPUT: matriz de sesgo

    if direccion == 'x':
        M= matrix([[1, tan(ang1), tan(ang2), 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    
    elif direccion == 'y':
        M= matrix([[1 ,0 ,0 ,0],[ tan(ang1), 1, tan(ang2), 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    
    else:
        M= matrix([[1, 0, 0, 0],[0, 1, 0, 0],[tan(ang1), tan(ang2), 1, 0],[0, 0, 0, 1]])
        
    return M

def sesgo_tangente(direccion,tan1,tan2):
    
    #INPUT: direccion ='x', 'y' o 'z'
    #       tangentes de los angulos ang1 y ang2 
    #OUTPUT: matriz de sesgo

    if direccion == 'x':
        M= matrix([[1, tan1, tan2, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    
    elif direccion == 'y':
        M= matrix([[1 ,0 ,0 ,0],[ tan1, 1,tan2, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
    
    else:
        M= matrix([[1, 0, 0, 0],[0, 1, 0, 0],[tan1, tan2, 1, 0],[0, 0, 0, 1]])
        
    return M


def afilar(eje,f1,f2,P):
    
    #INPUT: 'x','y' ó 'z'
    #       funciones de escalado
    #       punto P a transformar    
    #OUTPUT: punto transformado
         
    if eje == 'x':
        M= matrix([[P[0,0]],[f1(P[0,0])*P[1,0]],[f2(P[0,0])*P[2,0]],[1]])
    elif eje == 'y':
        M= matrix([[f1(P[1,0])*P[0,0]],[P[1,0]],[f2(P[1,0])*P[2,0]],[1]])
    else:
        M= matrix([[f1(P[2,0])*P[0,0]],[f2(P[2,0])*P[1,0]],[P[2,0]],[1]])
        
    return M

def retorcer(eje,f,P):
    
    #INPUT: eje='x', 'y' o 'z'
    #       función de giro
    #       punto P a transformar
    #OUTPUT: punto transformado
    
    if eje == 'x':
        M= matrix([[P[0,0]],[cos(f(P[0,0]))*P[1,0]-sin(f(P[0,0]))*P[2,0]],[sin(f(P[0,0]))*P[1,0]+cos(f(P[0,0]))*P[2,0]],[1]])
    elif eje == 'y':
        M= matrix([[cos(f(P[1,0]))*P[0,0]+sin(f(P[1,0]))*P[2,0]],[P[1,0]],[-sin(f(P[1,0]))*P[0,0]+cos(f(P[1,0]))*P[2,0]],[1]])
    else:
        M= matrix([[cos(f(P[2,0]))*P[0,0]-sin(f(P[2,0]))*P[1,0]],[sin(f(P[2,0]))*P[0,0]+cos(f(P[2,0]))*P[1,0]],[P[2,0]],[1]]) 
        
    return M
    
    
#funciones de la practica 3
def punto_curva_bezier(PC,t):

    # Input: matriz PC con los puntos de control de una cúbica de Bézier, y valor t en [0,1] de la variable que parametriza la curva.
    # Output: el vector columna del punto de la curva de Bézier en el valor t dado.
    
    B = matrix([[-1,3,-3,1],[3,-6,3,0],[-3,3,0,0],[1,0,0,0]])    
    T = matrix([[t^3],[t^2],[t],[1]])
    
    return PC*B*T

def puntos_curva_bezier(PC,h):
    
    # Input: matriz PC con los puntos de control de una cúbica de Bézier, y valor h del salto con que conformar un soporte del intervalo [0,1] que parametriza la curva.
    # Output: la matriz cuyas columnas conforman los puntos de la curva de Bézier en el soporte definido.    
    
    B = matrix([[-1,3,-3,1],[3,-6,3,0],[-3,3,0,0],[1,0,0,0]])
    t = srange(0,1,h)     # devuelve la lista (0,h,2h,...,nh) con nh<1
    t.append(1)           #incluimos el valor 1
    T = matrix([[i^3,i^2,i,1] for i in t])
    T = T.transpose()
    
    return PC*B*T

def dibujar_curva(puntos):
    
    # Input: matriz de puntos de una cúbica de Bézier, sobre un soporte implícito.
    # Output: la gráfica de la poligonal que une dichos puntos.    
    
    P = [n(puntos.column(i)[0:3]) for i in range(puntos.ncols())]
    
    return points(P,size=8) + line3d(P)

def dibujar_curva_bezier_y_poliedro(PC,h):

    # Input: matriz de puntos de control de una cúbica de Bézier, y valor h del salto con que conformar un soporte del intervalo [0,1] que parametriza la curva.
    # Output: la gráfica de la poligonal que une los puntos del soporte, así como la envolvente convexa que determinan los puntos de control.
    
    puntos = puntos_curva_bezier(PC,h)
    grafica3 = dibujar_curva(puntos)
    grafica3.set_texture(color = 'blue')
    
    pc= [n(PC.column(i)[0:3]) for i in range(PC.ncols())]
    grafica1 = Polyhedron(pc)     
    grafica2 = points(pc,color='red',size=14)
        
    return grafica1.plot(alpha=0.1) + grafica2 + grafica3

def punto_bezier_compuesta(PCs,u):
    
    # Input: matriz PCs con los 3N+1 puntos de control definiendo una cúbica de Bézier a trozos, y valor "u" en [0,N] de la variable que parametriza la curva.
    # Output: el vector columna del punto de la curva de Bézier a trozos en el valor "u" dado.
    
    n = abs(u-10^(-15)).floor()
    t = u-n
       
    PC = PCs[:,3*n:3*n+4]
    punto = punto_curva_bezier(PC,t)    
    
    return punto

def puntos_bezier_compuesta(PCs,h):
    
    # Input: matriz PCs con los 3N+1 puntos de control definiendo una cúbica de Bézier a trozos, y valor h del salto con que conformar un soporte del intervalo [0,N] que parametriza la curva.
    # Output: la matriz cuyas columnas conforman los puntos de la curva de Bézier en el soporte definido.    
    
    puntos = matrix(RR,PCs[:,0])
    
    N = (PCs.ncols()-1)/3
    N = N.floor()
    #print(N)
    u = srange(h,N,h)
    u.append(N)
    #print(u)
    for i in u:
        punto = punto_bezier_compuesta(PCs,i)
        puntos = puntos.augment(punto)   
    
    return puntos

def punto_curva_bezier_derivada1(PC,t):

    # Input: matriz PC de puntos de control de una curva de Bézier, valor t del parámetro de la curva en [0,1].
    # Output: matriz columna con el punto de la curva derivada en t.
    
    
    B1 = matrix([[-3,6,-3],[9,-12,3],[-9,6,0],[3,0,0]])    
    T = matrix([[t^2],[t],[1]])
    punto = PC*B1*T
    
    return punto

def punto_curva_bezier_derivada2(PC,t):

    # Input: matriz PC de puntos de control de una curva de Bézier, valor t del parámetro de la curva en [0,1].
    # Output: matriz columna con el punto de la curva derivada segunda en t.
    
    B2 = matrix([[-6,6],[18,-12],[-18,6],[6,0]])    
    T = matrix([[t],[1]])
    punto = PC*B2*T
    
    return punto

def esClaseC1(PCs):

    # Input: matriz PCs con los 3N+1 puntos de control definiendo una cúbica de Bézier a trozos.
    # Output: 1, si la curva es de Clase 1, 0 en otro caso.    
    
    for i in range(PCs.ncols()):
        if i%3==0 and i>0 and i<PCs.ncols()-1:
            PC1 = PCs[:,i-3:i+1]
            PC2 = PCs[:,i:i+4]
            if punto_curva_bezier_derivada1(PC1,1)!=punto_curva_bezier_derivada1(PC2,0):
                print('La curva no es de clase C1')
                return 0      
    print('La curva es de clase C1')
    return 1

def esClaseC2(PCs):
    
    # Input: matriz PCs con los 3N+1 puntos de control definiendo una cúbica de Bézier a trozos.
    # Output: 1, si la curva es de Clase 2, 0 en otro caso.        
    
    #Hay que comprobar primero si es clase C1
    if esClaseC1(PCs)==0:
        return 0 
    else:
        for i in range(PCs.ncols()):
            if i%3==0 and i>0 and i<PCs.ncols()-1:
                PC1 = PCs[:,i-3:i+1]
                PC2 = PCs[:,i:i+4]
                if punto_curva_bezier_derivada2(PC1,1)!=punto_curva_bezier_derivada2(PC2,0):
                    print('La curva no es de clase C2')
                    return 0      
        print('La curva es de clase C2')
        return 1

def dame_puntos_de_control(M,indices,trozos,clase,cerrada):
    
    # INPUT: una matriz "M" con ciertos puntos de control $P_{i_j}$ por columnas en homogéneas, 
    #        la lista "indices" con los subíndices de estos puntos de control, {i0,...,ik},
    #        el número de "tramos" de cúbicas de que consta la curva y la "clase"=0,1,2 de la curva.
    #
    # OUTPUT: si la curva no está unívocamente determinada, muestra la cantidad de parámetros libres que faltan por precisar y devuelve la lista vacía;
    #         caso de que esté unívocamente determinada, el conjunto completo de puntos de control, en forma matricial.
    
    ###
    ####### definimos variables de los puntos de control de una cúbica de Bézier compuesta por m tramos.
    ###
    variables=[]
    m=trozos # número de tramos en la curva de Bézier
    for i in range(3*m+1):
        variables.extend([var('x'+str(i)),var('y'+str(i)),var('z'+str(i))])
    matriz_coeficientes=matrix([[0 for i in range(9*m+3)]])
    termino_independiente=[0]
    
    ###
    ####### ecuaciones inferidas de los puntos de control conocidos
    ###
    
    for i in range(len(indices)):
        for k in range(3):
            vec=vector([0 for j in range(9*m+3)])
            vec[3*indices[i]+k]=1
            matriz_coeficientes=matriz_coeficientes.stack(vec)
            termino_independiente.append(M[k,i])
    
    if clase>0:
        
        ###
        ####### ecuaciones inferidas de que sea Clase 1
        ###
        for i in range(1,m):
            for k in range(3):
                vec=vector([0 for j in range(9*m+3)])
                vec[9*i+k]=2
                vec[9*i+k-3]=-1
                vec[9*i+k+3]=-1
                matriz_coeficientes=matriz_coeficientes.stack(vec)
                termino_independiente.append(0)
        if clase>1:
            
            ###
            ####### ecuaciones inferidas de que sea Clase 2
            ###
            for i in range(1,m):
                for k in range(3):
                    vec=vector([0 for j in range(9*m+3)])
                    vec[9*i+k-6]=1
                    vec[9*i+k-3]=-2
                    vec[9*i+k+3]=2
                    vec[9*i+k+6]=-1
                    matriz_coeficientes=matriz_coeficientes.stack(vec)
                    termino_independiente.append(0)
    
    if cerrada==1:
        ###
        ####### si la curva compuesta es cerrada, añadimos las condiciones correspondientes
        ###
        for k in range(3):
            vec=vector([0 for j in range(9*m+3)])
            vec[9*m+k]=1
            vec[k]=-1
            matriz_coeficientes=matriz_coeficientes.stack(vec)
            termino_independiente.append(0)
        
        if clase>0:
        
            ###
            ####### para que sea Clase 1
            ###
            for k in range(3):
                vec=vector([0 for j in range(9*m+3)])
                vec[9*m+k]=1
                vec[9*m+k-3]=-1
                vec[k+3]=-1
                vec[k]=1
                matriz_coeficientes=matriz_coeficientes.stack(vec)
                termino_independiente.append(0)
                
            if clase>1:

                ###
                ####### para que sea Clase 2
                ###
                for k in range(3):
                    vec=vector([0 for j in range(9*m+3)])
                    vec[9*m+k-6]=1
                    vec[9*m+k-3]=-2
                    vec[k+3]=2
                    vec[k+6]=-1
                    matriz_coeficientes=matriz_coeficientes.stack(vec)
                    termino_independiente.append(0)    
    libres=(9*m+3-matriz_coeficientes.rank())/3
    if libres>0:
        print('Número de grados de libertad (esto es, puntos de control por determinar para caracterizar unívocamente la curva): ',libres)
        return []
    
    ###
    ###### determinación del conjunto de puntos de control
    ###
    sol=matriz_coeficientes.solve_right(vector(termino_independiente))
    PC=(matrix(len(sol)/3,3,sol).transpose()).stack(vector([1 for i in range(len(sol)/3)]))
    
    return PC

def dibujar_curva_parametrica(C,a,b):
    
     #INPUT: C = Ecuación paramétrica de la curva en coordenadas homogéneas
     #       a,b = límites del parámetro u
     #OUTPUT: Gráfica de la curva
        
    return parametric_plot3d((lambda u: C(u)[0,0],lambda u: C(u)[1,0],lambda u: C(u)[2,0]), (a,b),color='red') 

def dibujar_superficie_parametrica(S,a,b,c,d):
    
     #INPUT: S = Ecuación paramétrica de la superficie en coordenadas homogéneas
     #       a,b = límites del parámetro u
     #       c,d = límites del parámetro v
     #OUTPUT: Gráfica de la superficie
    
    return parametric_plot3d((lambda u,v: S(u,v)[0,0],lambda u,v: S(u,v)[1,0],lambda u,v: S(u,v)[2,0]), (a,b), (c,d), opacity = 0.8) 


def dibujar_curvas_isoparametricas(S,a,b,c,d,h1,h2):
    
     #INPUT: S = Ecuación paramétrica de la superficie en coordenadas homogéneas
     #       a,b = límites del parámetro u
     #       c,d = límites del parámetro v
     #OUTPUT: Gráfica de un mallado de la superficie, tomando un salto h1 ó h2 en cada soporte
    
    dib=0
    sop1=[]
    for i in range(0,floor((b-a)/h1)):
        sop1.append(a+i*h1)
    sop1.append(b)
    sop2=[]
    for i in range(0,floor((d-c)/h2)):
        sop2.append(c+i*h2)
    sop2.append(d)
    for i in sop1:
        dib=dib+parametric_plot3d((lambda v: S(i,v)[0,0],lambda v: S(i,v)[1,0],lambda v: S(i,v)[2,0]), (c,d),color='yellow') 
    for i in sop2:
        dib=dib+parametric_plot3d((lambda u: S(u,i)[0,0],lambda u: S(u,i)[1,0],lambda u: S(u,i)[2,0]), (a,b),color='blue') 
    
    return dib

def superficie_cilindrica(C,D):
    
    #INPUT: C = ecuación paramétrica de la curva en coordenadas homogéneas
    #       D = vector que indica la dirección         
    #OUTPUT: Ecuación paramétrica de la superficie en coordenadas homogéneas    
    
    return lambda u,v: C(u) + v*D

def superficie_revolucion_eje_z(C):
    
    #INPUT: C = ecuación paramétrica de la curva en coordenadas homogéneas con coordenada y igual a 0    
    #OUTPUT: Ecuación paramétrica de la superficie en coordenadas homogéneas  
    
    #Matriz que produce la revolución sobre el eje z
    

    
    return lambda u,v: matrix([[cos(v),0,0,0],[sin(v),0,0,0],[0,0,1,0],[0,0,0,1]])*C(u)

def superficie_reglada(C1,C2):
    
    #INPUT: C1 y C2 = ecuaciones paramétricas de curvas en coordenadas homogéneas.
    #Ejemplo:  Curva de Bezier:
    #          U=matrix([[u^3],[u^2],[u],[1]])
    #          C1=PC*B*U       
    #OUTPUT: Ecuación paramétrica de la superficie en coordenadas homogéneas    
    
    return lambda u,v: (1-v)*C1(u)+v*C2(u)

def superficie_bilineal(PC):
    
    #INPUT: PC = matriz de cuatro puntos en coordenadas homogéneas.          
    #OUTPUT: Ecuación paramétrica de la superficie en coordenadas homogéneas 
    
    #Obtenemos cada punto por separado en coordenadas homogéneas.
    P0 = PC[:,0]
    P1 = PC[:,1]
    P2 = PC[:,2]
    P3 = PC[:,3]
    
    #Ecuación paramétrica de las curvas C1 y C2:
     
    C1 = lambda u: (1-u)*P0+u*P1
    C2 = lambda u: (1-u)*P2+u*P3
    
    return superficie_reglada(C1,C2)

def superficie_bezier(PCs):
    
    #INPUT: PCs = matriz de 16 puntos  en coordenadas homogéneas        
    #OUTPUT: Ecuación paramétrica de la superficie en coordenadas homogéneas     
    
    #Obtenemos los puntos de control de cada una de las cuatro curvas por separado en coordenadas homogéneas.
    PC1 = PCs[:,0:4]
    PC2 = PCs[:,4:8]
    PC3 = PCs[:,8:12]
    PC4 = PCs[:,12:16]
    
    #Matriz de Bézier
    B = matrix([[-1,3,-3,1],[3,-6,3,0],[-3,3,0,0],[1,0,0,0]])  
    
    
    return lambda u,v: ((PC1*B*matrix([[u^3],[u^2],[u],[1]])).augment((PC2*B*matrix([[u^3],[u^2],[u],[1]]))).augment((PC3*B*matrix([[u^3],[u^2],[u],[1]]))).augment((PC4*B*matrix([[u^3],[u^2],[u],[1]]))))*B*matrix([[v^3],[v^2],[v],[1]])
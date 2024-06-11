# RolEmpatIA

## Descripción del Proyecto
En una sociedad cada vez más individualizada, la creación de vínculos reales y significativos puede ser un desafío. El objetivo de RolEmpatIA es ofrecer un entorno seguro donde los usuarios puedan experimentar y practicar diferentes estilos de socialización sin el temor al fracaso o las consecuencias negativas. Nuestro chatbot, diseñado para interactuar de manera natural y empática, ofrece una forma divertida y educativa de mejorar las interacciones sociales a través de juegos de rol.

## Dataset
En este caso, el data set es creado. No encontramos dataset en español que cumplieran los requisitos que necesitábamos. Procedimos a realizar WebScraping en foros de rol donde había partidas terminadas, pero al ser creada por los usuarios sin tener un patron claro, tras comenzar el WebScraping con Uipath, Scrapy o BeautifulSoup, finalmente completamos el dataset de forma manual, extrayendo nosotras mismas el dataset de dichos foros. Es por ello que no es tan extenso como nos gustaría. Además, decidimos hacer una selección de qué partidas utilizar, ya que había muchas +18, +21 o algunas con un claro componente folklórico como "Poseidos en Vallekas" que utilizaba personajes históricos del imaginario español como Antonio Resines, Bertín Osborne o el Rey Emérito Juan Carlos de Borbón. 

## Preprocesamiento de datos 

Lo primero que debíamos completar era la limpieza de los datos. El dataset se compone de tres columnas: Personaje - Hora del Mensaje - Texto. Para organizarnos a la hora de crear el dataset venía perfecto, pero debíamos limpiar la columna que nos interesaba, "Texto", para poder empezar a utilizarla.  

```python
nltk.download("stopwords")
    # Convertimos el contenido de la comuna texto en string
data["Texto"] = data["Texto"].astype(str)

    # Preprocesamiento de texto
data["texto_limpio"] = data["Texto"].str.lower()  # Convertimos a minúsculas
data["texto_limpio"] = data["texto_limpio"].apply(unidecode)  # Eliminamos acentos
data["texto_limpio"] = data["texto_limpio"].apply(lambda x: re.sub(r"\d+", " ", x))  # Eliminamos números
data["texto_limpio"] = data["texto_limpio"].str.translate(
    str.maketrans(string.punctuation, " " * len(string.punctuation))
)  # Eliminamos signos de puntuación
data["texto_limpio"] = data["texto_limpio"].str.replace(r"\s{2,}", " ", regex=True).str.strip()  # Eliminamos espacios innecesarios
    # Palabras a eliminar adicionales
stop = stopwords.words("spanish") + palabras_a_eliminar
data["texto_limpio"] = data["texto_limpio"].apply(
    lambda x: " ".join([word for word in x.split() if word not in (stop)])
)  # Eliminamos las stopwords
``` 
Más tarde vimos, por las características de las partidas de Rol, quedebíamos eliminar algunas palabras extras: 

```python
palabras_a_eliminar = ['mas', 'si', 'tan', 'habia', 'asi', 'oh', 'vez', 'y', 'h', 'mismo', 'aunque', 'mientras',
                       'que', 'aun', 'seras', 'cualquier', 'misma', 'mmpppfff', 'rurik', 'jum', 'wilfrick', 'jeet',
                       'julgram', 'thrommel', 'beran', 'dennek', 'caranthir', 'groak', 'jimblecap', 'gulgram', 'orsik',
                       'soren', 'ay', 'grit', 'tambien', 'groac']

```

## Tokenización y Vectorización

Para poder utilizar el contenido de la nueva columna llamada 'texto_limpio', realizamos una tokenización palabra por palabra y creamos un modelo word2vec para vectorizar las mismas. Lo aplicamos a la lista creada con las palabras y verificamos si funciona a través de la propiedad del modelo most_similar: 

```python
similar_words_magia = rol2vec.wv.most_similar("magia")
print("Palabras similares a 'magia':", similar_words_magia)

```
Palabras similares a 'magia': [('continuo', 0.5517002940177917), ('grabo', 0.46825912594795227), ('lado', 0.46211525797843933), ('disculpad', 0.42598873376846313), ('forma', 0.41744449734687805), ('cautivo', 0.4152204692363739), ('ocurriros', 0.41037052869796753), ('sintieron', 0.4026062786579132), ('jugar', 0.40196770429611206), ('guardias', 0.3986649513244629)]

```python
similar_words_disciplina = rol2vec.wv.most_similar("disciplina")
print("Palabras similares a 'disciplina':", similar_words_disciplina)

```
Palabras similares a 'disciplina': [('recuerdos', 0.5104620456695557), ('creo', 0.4909777045249939), ('horda', 0.4846702516078949), ('marcho', 0.451127290725708), ('asientos', 0.38866427540779114), ('embargo', 0.38759520649909973), ('rebosa', 0.3807078003883362), ('detente', 0.37873977422714233), ('pilares', 0.3767443001270294), ('sorprendente', 0.37432533502578735)]
```python
similar_words_dano = rol2vec.wv.most_similar("dano")
print("Palabras similares a 'dano':", similar_words_dano)

```
Palabras similares a 'dano': [('cercenemos', 0.46414145827293396), ('pequeno', 0.4534483850002289), ('desalmados', 0.43120548129081726), ('cazadores', 0.4102203845977783), ('pretencioso', 0.40368127822875977), ('caso', 0.3865625858306885), ('requieren', 0.385661780834198), ('vas', 0.3841119706630707), ('sali', 0.3817184567451477), ('ventrue', 0.3800927400588989)]


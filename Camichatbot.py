from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

datos_entrenamiento = [
    "Hola chat", "¿Cuál es tu nombre?",
    "¿Qué es un chatbot?",
    "¿Qué puedes hacer?",
    "¿Cuál es tu propósito?",
    "¿Qué lenguajes entiendes?",
    "¿Qué es la inteligencia artificial?",
    "¿Cómo estás?",
    "¿Qué día es hoy?",
    "¿Cuál es la capital de Uruguay?",
    "¿Cuál es el cuadro más grande de Uruguay?",
    "¿Cuál es tu color favorito?",
    "¿Qué hora es en Montevideo?",
    "¿Qué significa HTML?",
    "¿Qué significa CSS?",
    "¿Qué significa JS?",
    "¿Me puedes contar un chiste?",
    "¿Qué es Python?",
    "¿Qué es un liceo?",
    "¿Quién te creó?",
    "¿Eres humano?",
    "¿Puedes aprender?",
    "¿Cuál es el mejor liceo en Uruguay?",
    "¿Quién es Camilo Sosa?"
]

respuestas_entrenamiento = [
    "¡Hola! ¿En qué puedo ayudarte?",
    "Mi nombre es Chatbot de Camilo.",
    "Un chatbot es un programa que conversa con personas.",
    "Puedo responder preguntas y ayudarte con información.",
    "Mi propósito es ayudarte y responder tus dudas.",
    "Entiendo español y algunos términos en inglés.",
    "La inteligencia artificial es la simulación de la inteligencia humana por máquinas.",
    "Estoy bien, gracias por preguntar.",
    "Hoy es un gran día, pero no tengo acceso a la fecha exacta ya que puede variar.",
    "La capital de Uruguay es Montevideo.",
    "El cuadro más grande de Uruguay es el Club Nacional de Football.",
    "No tengo un color favorito, pero me gustan todos.",
    "No tengo acceso a la hora exacta, pero puedes consultarla en tu celu.",
    "HTML significa HyperText Markup Language.",
    "CSS significa Cascading Style Sheets.",
    "JS significa JavaScript.",
    "Claro, aquí va uno: ¿Por qué el libro de matemáticas está triste? Porque tiene muchos problemas.",
    "Python es un lenguaje de programación muy popular.",
    "Un liceo es una institución educativa de nivel secundario de Uruguay.",
    "Fui creado por un crack, Camilo Sosa.",
    "No, soy una inteligencia artificial.",
    "Puedo aprender si me entrenan con nuevos datos.",
    "El mejor liceo de Uruguay sin duda es el IAVA",
    "Camilo Sosa es mi creador."
]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(datos_entrenamiento)
modelo = MultinomialNB()
modelo.fit(X_train, respuestas_entrenamiento)

while True:
    entrada = input("Tú: ")
    if entrada.lower() in ["salir", "adiós", "chau"]:
        print("Chatbot: Chau Crack, que tengas un lindo día rey.")
        break

    entrada_vector = vectorizer.transform([entrada])
    respuesta = modelo.predict(entrada_vector)
    print("Chatbot:", respuesta[0])

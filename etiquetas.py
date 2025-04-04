"""
SA:
0 -> Negative
1 -> Neutral
2 -> Positive

NER:
0 -> O		No es una entidad
1 -> B-PER	Inicio de un nombre de persona
2 -> I-PER	
3 -> B-ORG	Inicio de una organización
4 -> I-ORG	
5 -> B-LOC	Inicio de una ubicación (país, ciudad, etc.)
6 -> I-LOC	
7 -> B-MISC	Inicio de una entidad miscelánea (nacionalidad, eventos, etc.)
8 -> I-MISC"
"""

def map_ner_tags(ner_tag):
    # Mapa de etiquetas NER de CoNLL-2003 a las etiquetas del esquema propuesto
    ner_map = {
        0: "O",   # No es una entidad
        1: "B-PER",  # Inicio de Persona
        2: "I-PER",  # Continuación de Persona
        3: "B-ORG",  # Inicio de Organización
        4: "I-ORG",  # Continuación de Organización
        5: "B-LOC",  # Inicio de Ubicación
        6: "I-LOC",  # Continuación de Ubicación
        7: "B-MISC",  # Inicio de Entidad Miscelánea
        8: "I-MISC"   # Continuación de Entidad Miscelánea
    }

    return [ner_map[tag] for tag in ner_tag]
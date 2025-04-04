
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

def map_sa_tags(sa_tag):

    ner_map = {
        0: "Negative", 
        1: "Neutral",
        2: "Positive", 
    }

    return [sa_tag[tag] for tag in sa_tag]
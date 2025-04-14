# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule
import torch.nn.functional as F

# other libraries
from typing import Final
import re
import os
import json
from shutil import rmtree

# own modules
from src.data import load_data, load_embeddings, map_sa_tags, map_ner_tags
from src.utils import set_seed, load_model
from src.train_functions import t_step, t_step_nersa

# static variables
DATA_PATH: Final[str] = "data"
TEMP_FILE: Final[str] =  "unique_test.json"
EMBEDINGS_PATH: Final[str] = "embeddings"

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)

def create_temp_json(phrase: str) -> None:
    """Crea un archivo JSON temporal con la frase proporcionada."""
    tokens = phrase.split()
    dummy_ner = [1] * len(tokens)  # "O" -> 1 según tu mapeo
    dummy_sa = 1  # Neutral

    sample = {
        "tokens": tokens,
        "ner": dummy_ner,
        "sa": dummy_sa
    }
    with open(DATA_PATH + "/" + TEMP_FILE, "w") as f:
        json.dump(sample, f)
        f.write("\n")  # línea para JSONL

def delete_temp_json() -> None:
    """Elimina el archivo temporal después de evaluar."""
    rmtree(DATA_PATH + "/" + TEMP_FILE)


def main() -> None:
    """
    This function is the main program.
    """
    name = "glove_50d_SA_5_128_64_2_0.001_0.0"
    
    prueba_externa = False
    phrase = "I hate you so much I would kill you"
    # phrase_list = ["EU","rejects","German","call","to","boycott","British","lamb","."]
    # phrase = " ".join(phrase_list)
    
    w2v_model = load_embeddings(EMBEDINGS_PATH)
    model: RecursiveScriptModule = load_model(f"{name}").to(device)
    
    match = re.search(r"glove_50d_([^_]+)_", name)
    modo = match.group(1)
    
    if prueba_externa:
        create_temp_json(phrase)

        test_loader, _, _ = load_data(w2v_model=w2v_model,
                                        save_path=DATA_PATH,
                                        batch_size=1,
                                        shuffle=False,
                                        unique_test=True
                                    )
        
        for inputs, ner_targets, sa_targets, lengths in test_loader:
            inputs, lengths = inputs.to(device), lengths.to(device)
            out1, out2 = model(inputs, lengths)

            if modo == "SA":
                probs = F.softmax(out1, dim=-1)
                pred = probs.argmax(dim=-1).item()
                label = map_sa_tags([pred])[0]
                print(f"\nFrase: \"{phrase}\"")
                print(f"Sentimiento: {label}")

            elif modo == "NER":
                probs = F.softmax(out1, dim=-1)
                preds = probs.argmax(dim=-1).squeeze(0).tolist()
                tags = map_ner_tags(preds)
                print(f"\nFrase con etiquetas NER:")
                for tok, tag in zip(phrase.split(), tags):
                    print(f"{tok:10} → {tag}")

            else:  # NERSA
                probs_sa = F.softmax(out2, dim=-1)
                pred_sa = probs_sa.argmax(dim=-1).item()
                tag_sa = map_sa_tags([pred_sa])[0]

                probs_ner = F.softmax(out1, dim=-1)
                preds_ner = probs_ner.argmax(dim=-1).squeeze(0).tolist()
                tags_ner = map_ner_tags(preds_ner)

                print(f"\nFrase: \"{phrase}\"")
                print(f"Sentimiento: {tag_sa}")
                print("Etiquetas NER:")
                for tok, tag in zip(phrase.split(), tags_ner):
                    print(f"{tok:10} → {tag}")

        # 4. Borrar archivo temporal
        delete_temp_json()
                
    else:
        test_data: DataLoader
        w2v_model = load_embeddings(EMBEDINGS_PATH)

        _, _, test_data = load_data(w2v_model = w2v_model,
                                    save_path=DATA_PATH, 
                                    batch_size=64, 
                                    shuffle=True, 
                                    drop_last=False, 
                                    num_workers=4)

        if modo == "NERSA":
            test_ner, test_sa = t_step_nersa(model, test_data, device)
        else:
            test = t_step(modo, model, test_data, device)
        


if __name__ == "__main__":
    main()

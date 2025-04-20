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

# own modules
from src.data import load_data, load_embeddings, map_sa_tags, map_ner_tags
from src.utils import set_seed, load_model
from src.train_functions import t_step_sa, t_step_ner, t_step_nersa

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
    temp_path = os.path.join(DATA_PATH, TEMP_FILE)
    if os.path.exists(temp_path):
        os.remove(temp_path)

def alert_creator(alert_promt, tag, tok):
    if tag in [1,2]:
        alert_promt += f"{tok} as a person entity, "
    elif tag in [3,4]:
        alert_promt += f"{tok} as a organization entity, "
    elif tag in [5,6]:
        alert_promt += f"{tok} as a ubication entity, "
    elif tag in [7,8]:
        alert_promt += f"{tok} as a miscellaneous entity, "
    else:
        alert_promt += ""
    return alert_promt

def main() -> None:
    """
    This function is the main program.
    """
    name = "glove_50d_NERSA_30_128"  # Nombre del modelo que se va a ejecutar, tiene que ser de este formato, no otro
    
    prueba_externa = True
    alert = False
    # phrase = "John suffered from an awful fall to the ground" # input_prompt

    phrase_list = ["it","all","culminated","in","the","fact","that","i","now","have","lots","of","great",",","great","friends","in","ireland","."]
    phrase = " ".join(phrase_list)
    
    match = re.search(r"glove_([0-9]+)d_([^_]+)_", name)
    emb_dim = match.group(1)
    modo = match.group(2)
    
    w2v_model = load_embeddings(EMBEDINGS_PATH, int(emb_dim))
    model: RecursiveScriptModule = load_model(f"{name}").to(device)
    
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
                label = map_sa_tags(pred)
                print(f"\nFrase: \"{phrase}\"")
                print(f"Sentimiento: {label}")

            elif modo == "NER":
                probs = F.softmax(out1, dim=-1)
                preds = probs.argmax(dim=-1).squeeze(0).tolist()
                print(f"\nFrase con etiquetas NER:")
                for tok, tag in zip(phrase.split(), preds):
                    print(f"{tok:10} → {map_ner_tags(tag)}")

            else:  # NERSA
                probs_sa = F.softmax(out2, dim=-1)
                pred_sa = probs_sa.argmax(dim=-1).item()
                tag_sa = map_sa_tags(pred_sa)

                probs_ner = F.softmax(out1, dim=-1)
                preds_ner = probs_ner.argmax(dim=-1).squeeze(0).tolist()

                print(f"\nFrase: \"{phrase}\"")
                print(f"Sentimiento: {tag_sa}")
                
                if pred_sa == 0:
                    alert = True
                    alert_prompt = "\nGenerate an alert for: "
                elif pred_sa == 2:
                    alert = True
                    alert_prompt = "\nGenerate a congratulatory message for: "
                print("Etiquetas NER:")
                for tok, tag in zip(phrase.split(), preds_ner):
                    print(f"{tok:10} → {map_ner_tags(tag)}")
                    if alert:
                        alert_prompt = alert_creator(alert_prompt, tag, tok)

        if alert:
            print(alert_prompt)
        # 4. Borrar archivo temporal
        delete_temp_json()
                
    else:
        test_data: DataLoader

        _, _, test_data = load_data(w2v_model = w2v_model,
                                    save_path=DATA_PATH, 
                                    batch_size=64, 
                                    shuffle=True, 
                                    drop_last=False, 
                                    num_workers=4)
        
        loss_ner = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss_sa = torch.nn.CrossEntropyLoss() 
    
        if modo == "NERSA":
            t_step_nersa(model, test_data, loss_ner, loss_sa, device)
        elif modo == "NER":
            t_step_ner(model, test_data, loss_ner, device)
        else:
            t_step_sa(model, test_data, loss_sa, device)
        


if __name__ == "__main__":
    main()

# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule
import torch.nn.functional as F

# other libraries
from typing import Final
import re

# own modules
from src.data import load_data, load_embeddings, map_sa_tags, map_ner_tags
from src.utils import set_seed, load_model, create_temp_json, delete_temp_json, alert_creator
from src.train_functions import t_step_sa, t_step_ner, t_step_nersa
from src.deepseek import generate_text

# static variables
DATA_PATH: Final[str] = "data"
TEMP_FILE: Final[str] =  "unique_test.json"
EMBEDINGS_PATH: Final[str] = "embeddings"

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    """
    This function is the main program.
    """
    name = "glove_50d_NERSA_30_128"  # Nombre del modelo que se va a ejecutar, tiene que ser de este formato, no otro
    
    prueba_externa = True

    # phrase = "John suffered from an awful fall to the ground" # input_prompt
    # phrase = " ".join(phrase)
    
    phrase = ["but","china","saw","their","luck","desert","them","in","the","second","match","of","the","group",",","crashing","to","a","surprise","2-0","defeat","to","newcomers","uzbekistan","."]

    alert = False
    match = re.search(r"glove_([0-9]+)d_([^_]+)_", name)
    emb_dim = match.group(1)
    modo = match.group(2)
    
    w2v_model = load_embeddings(EMBEDINGS_PATH, int(emb_dim))
    model: RecursiveScriptModule = load_model(f"{name}").to(device)
    
    phrase_clean = [word for word in phrase if word in w2v_model.key_to_index]
    phrase_clean_str = " ".join(phrase_clean)
    
    if prueba_externa:
        create_temp_json(phrase_clean, DATA_PATH, TEMP_FILE)

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
                print(f"\nFrase: \"{phrase_clean_str}\"")
                print(f"Sentimiento: {label}")

            elif modo == "NER":
                probs = F.softmax(out1, dim=-1)
                preds = probs.argmax(dim=-1).squeeze(0).tolist()
                print(f"\nFrase con etiquetas NER:")
                for tok, tag in zip(phrase_clean_str, preds):
                    print(f"{tok:10} → {map_ner_tags(tag)}")

            else:  # NERSA
                probs_sa = F.softmax(out2, dim=-1)
                pred_sa = probs_sa.argmax(dim=-1).item()
                tag_sa = map_sa_tags(pred_sa)

                probs_ner = F.softmax(out1, dim=-1)
                preds_ner = probs_ner.argmax(dim=-1).squeeze(0).tolist()

                print(f"\nFrase: \"{phrase_clean_str}\"")
                print(f"Sentimiento: {tag_sa}")
                
                if pred_sa == 0:
                    alert = True
                    alert_prompt = "Generate an alert for: "
                elif pred_sa == 2:
                    alert = True
                    alert_prompt = "Generate a congratulatory message for: "
                print("Etiquetas NER:")
                for tok, tag in zip(phrase_clean, preds_ner):
                    print(f"{tok:10} → {map_ner_tags(tag)}")
                    if alert:
                        alert_prompt = alert_creator(alert_prompt, tag, tok)

        if alert:
            # texto_generado = generate_text(alert_prompt, device)
            texto_generado = ""
            print("\nPrompt: ", alert_prompt)
            print("Respuesta del modelo preentrenado: ", texto_generado)
            
        # 4. Borrar archivo temporal
        delete_temp_json(DATA_PATH, TEMP_FILE)
                
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

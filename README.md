# PF_NPL-DL
 
Authors:
- Jorge Enebral
- Matteo Ferrari
- Pedro Meseguer


INSTRUCTIONS:

- RUN train.py
    - DOWNLOAD GLOVE.TWITTER EMBEDDINGS (DEFAULT SIZE 50, BUT IT CAN BE CHANGED TO THESE SIZES: 25/50/100/200)
    - DOWNLOAD CONLL2003 DATA
    - CHANGE ANY HYPERPARAMETER IF DESIRED
    - IT CREATES A MODEL glove_{emb_dim}d_{mode}{epochs}{batch_size}

- TO GENERATE THE ALERT:
    - IN evaluate.py ASSIGN THE NAME OF THE TRAINED MODEL TO THE MODEL NAME VARIABLE 
    - SET prueba_externa = True
    - ENTER A PHRASE/TWITT IN STR OR SEPARATED IN A LIST AND RUN
    - OUTPUT (DEPENDING ON THE MODE AND RESULT):
        - SENTIMENT
        - TAGS FOR EACH WORD
        - PROMPT
        - ALERT
    - It is important to have the .env script that contains the key for the api that allows to use the DeepSeek R1 model.


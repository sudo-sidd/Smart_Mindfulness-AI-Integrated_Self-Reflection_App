from datasets import load_dataset

# GoEmotions
ds_go = load_dataset("go_emotions") 
ds_go.save_to_disk("/Datasets/goemotions")

#Empathetic dialogues
ds_ed = load_dataset("empathetic_dialogues")
ds_ed.save_to_disk("Datasets/empathetic_dialogues")
import spacy
import random
from spacy.training import Example
from spacy.training import offsets_to_biluo_tags

# Load a pre-trained model
nlp = spacy.load("en_core_web_sm")

TRAIN_DATA = [
    ("Mr. Dul took a paracetamol 500mg and had a headache.", {"entities": [(0, 7, "PERSON"), (15, 26, "MEDICATION"), (27, 32, "QUANTITY")]}),
    ("Dr. Stone prescribed Amoxicillin 250mg twice daily.", {"entities": [(0, 9, "PERSON"), (21, 32, "MEDICATION"), (33, 38, "QUANTITY")]}),
    ("The nurse recorded that the patient received 100mg of Metoprolol.", {"entities": [(45, 50, "QUANTITY"), (54, 64, "MEDICATION")]}),
    ("She has been advised to take two tablets of Cetirizine daily.", {"entities": [(44, 54, "MEDICATION")]}),
    ("Jonathan needs 10ml of liquid Prednisone after meals.", {"entities": [(0, 8, "PERSON"), (15, 19, "QUANTITY"), (30, 40, "MEDICATION")]}),
    ("The prescription was for 50mg Lasix to be taken once a day.", {"entities": [(25, 29, "QUANTITY"), (30, 35, "MEDICATION")]}),
    ("Please administer Diazepam 5mg as needed for anxiety.", {"entities": [(18, 26, "MEDICATION"), (27, 30, "QUANTITY")]}),
    ("Emily received 200mg of Ibuprofen after her surgery.", {"entities": [(0, 5, "PERSON"), (15, 20, "QUANTITY"), (24, 33, "MEDICATION")]}),
    ("The doctor gave John 15mg of Oxycodone for his severe pain.", {"entities": [(16, 20, "PERSON"), (21, 25, "QUANTITY"), (29, 38, "MEDICATION")]}),
    ("Susan needs to take 500mg of Amoxicillin for her throat infection.", {"entities": [(0, 5, "PERSON"), (20, 25, "QUANTITY"), (29, 40, "MEDICATION")]}),
]

TRAIN_DATA.extend([
    ("He was prescribed 2 pills of Naproxen to relieve the muscle aches.", {"entities": [(18, 25, "QUANTITY"), (29, 37, "MEDICATION")]}),
    ("Please give her 30ml of liquid Tylenol every six hours.", {"entities": [(16, 20, "QUANTITY"), (24, 38, "MEDICATION")]}),
    ("The prescription indicates 45mg of Prednisolone daily.", {"entities": [(27, 31, "QUANTITY"), (35, 47, "MEDICATION"), (48, 53, "WHEN")]}),
    ("Administer 50mg of Zoloft to the patient this evening.", {"entities": [(11, 15, "QUANTITY"), (19, 25, "MEDICATION"), (41, 53, "WHEN")]}),
    ("Martin was advised to use 10mg of Lexapro each morning.", {"entities": [(0, 6, "PERSON"), (26, 30, "QUANTITY"), (34, 41, "MEDICATION"), (42, 54, "WHEN")]}),
    ("For mild allergies, 10ml of Loratadine is recommended twice daily.", {"entities": [(20, 24, "QUANTITY"), (28, 38, "MEDICATION"), (54, 65, "WHEN")]}),
    ("In case of fever, prescribe 250mg of Acetaminophen.", {"entities": [(28, 33, "QUANTITY"), (37, 50, "MEDICATION")]}),
])

TRAIN_DATA.extend([
    ("Each tablet contains 100mg of aspirin.", {"entities": [(21, 26, "QUANTITY"), (30, 37, "MEDICATION")]}),
    ("Administer three capsules daily, each with 250mg of Vitamin C.", {"entities": [(38, 43, "QUANTITY"), (47, 56, "MEDICATION"), (11, 31, "WHEN")]}),
    ("The patient requires 1.5 liters of saline solution for hydration.", {"entities": [(21, 31, "QUANTITY")]}),
    ("Prescribe two doses of 500mg Metformin with meals.", {"entities": [(23, 28, "QUANTITY"), (29, 38, "MEDICATION"), (10, 19, "WHEN")]}),
    ("Mix 20ml of this syrup into the drink twice a day.", {"entities": [(4, 8, "QUANTITY"), (38, 49, "WHEN")]}),
    ("The new dosage is 75mg twice daily.", {"entities": [(18, 22, "QUANTITY")]}),
    ("Inject 0.5ml of the vaccine into the upper arm region.", {"entities": [(7, 12, "QUANTITY"), (20, 27, "MEDICATION")]}),
    ("A daily intake of 2 grams of fiber is recommended.", {"entities": [(18, 25, "QUANTITY"), (2, 7, "WHEN")]}),
    ("Use 10ml of the solution per 5kg of body weight.", {"entities": [(4, 8, "QUANTITY")]}),
    ("For severe pain, give 30mg of Morphine as needed.", {"entities": [(22, 26, "QUANTITY"), (30, 38, "MEDICATION")]}),
    ("Dispense 15ml of cough syrup every 6 hours.", {"entities": [(9, 13, "QUANTITY"), (17, 28, "MEDICATION"), (29, 42, "WHEN")]}),
    ("Take one 50mg tablet of Losartan every morning.", {"entities": [(9, 13, "QUANTITY"), (24, 32, "MEDICATION"), (33, 46, "WHEN")]}),
    ("Patient has prescribed 200mg of Aspirin two times daily.", {"entities": [(23, 28, "QUANTITY"), (32, 39, "MEDICATION"), (40, 55, "WHEN")]}),
])

TRAIN_DATA.extend([
    ("Mr. Dul took a paracetamol 500mg and had a headache.", {"entities": [(0, 7, "PERSON"), (15, 26, "MEDICATION"), (27, 32, "QUANTITY")]}),
    ("Dr. Stone prescribed Amoxicillin 250mg twice daily.", {"entities": [(0, 9, "PERSON"), (21, 32, "MEDICATION"), (33, 38, "QUANTITY")]}),
    ("The nurse recorded that the patient received 100mg of Metoprolol.", {"entities": [(45, 50, "QUANTITY"), (54, 64, "MEDICATION")]}),
    ("She has been advised to take two tablets of Cetirizine daily.", {"entities": [(44, 54, "MEDICATION")]}),
    ("Jonathan needs 10ml of liquid Prednisone after meals.", {"entities": [(0, 8, "PERSON"), (15, 19, "QUANTITY"), (30, 40, "MEDICATION")]}),
    ("The prescription was for 50mg Lasix to be taken once a day.", {"entities": [(25, 29, "QUANTITY"), (30, 35, "MEDICATION")]}),
    ("Please administer Diazepam 5mg as needed for anxiety.", {"entities": [(18, 26, "MEDICATION"), (27, 30, "QUANTITY")]}),
    ("Emily received 200mg of Ibuprofen after her surgery.", {"entities": [(0, 5, "PERSON"), (15, 20, "QUANTITY"), (24, 33, "MEDICATION")]}),
    ("The doctor gave John 15mg of Oxycodone for his severe pain.", {"entities": [(16, 20, "PERSON"), (21, 25, "QUANTITY"), (29, 38, "MEDICATION")]}),
    ("Susan needs to take 500mg of Amoxicillin for her throat infection.", {"entities": [(0, 5, "PERSON"), (20, 25, "QUANTITY"), (29, 40, "MEDICATION")]}),
    ("He was prescribed 2 pills of Naproxen to relieve the muscle aches.", {"entities": [(18, 25, "QUANTITY"), (29, 37, "MEDICATION")]}),
    ("Please give her 30ml of liquid Tylenol every six hours.", {"entities": [(16, 20, "QUANTITY"), (24, 38, "MEDICATION"), (39, 54, "WHEN")]}),
    ("The prescription indicates 45mg of Prednisolone daily.", {"entities": [(27, 31, "QUANTITY"), (35, 47, "MEDICATION"), (48, 53, "WHEN")]}),
    ("Administer 50mg of Zoloft to the patient this evening.", {"entities": [(11, 15, "QUANTITY"), (19, 25, "MEDICATION"), (41, 53, "WHEN")]}),
    ("Martin was advised to use 10mg of Lexapro each morning.", {"entities": [(0, 6, "PERSON"), (26, 30, "QUANTITY"), (34, 41, "MEDICATION"), (42, 54, "WHEN")]}),
    ("For mild allergies, 10ml of Loratadine is recommended twice daily.", {"entities": [(20, 24, "QUANTITY"), (28, 38, "MEDICATION"), (54, 65, "WHEN")]}),
    ("In case of fever, prescribe 250mg of Acetaminophen.", {"entities": [(28, 33, "QUANTITY"), (37, 50, "MEDICATION")]}),
    ("Each tablet contains 100mg of aspirin.", {"entities": [(21, 26, "QUANTITY"), (30, 37, "MEDICATION")]}),
    ("Administer three capsules daily, each with 250mg of Vitamin C.", {"entities": [(38, 43, "QUANTITY"), (47, 56, "MEDICATION"), (11, 31, "WHEN")]}),
    ("The patient requires 1.5 liters of saline solution for hydration.", {"entities": [(21, 31, "QUANTITY")]}),
    ("Prescribe two doses of 500mg Metformin with meals.", {"entities": [(23, 28, "QUANTITY"), (29, 38, "MEDICATION"), (10, 19, "WHEN")]}),
    ("Mix 20ml of this syrup into the drink twice a day.", {"entities": [(4, 8, "QUANTITY"), (38, 49, "WHEN")]}),
    ("The new dosage is 75mg twice daily.", {"entities": [(18, 22, "QUANTITY")]}),
    ("Inject 0.5ml of the vaccine into the upper arm region.", {"entities": [(7, 12, "QUANTITY"), (20, 27, "MEDICATION")]}),
    ("A daily intake of 2 grams of fiber is recommended.", {"entities": [(18, 25, "QUANTITY"), (2, 7, "WHEN")]}),
    ("Use 10ml of the solution per 5kg of body weight.", {"entities": [(4, 8, "QUANTITY")]}),
    ("For severe pain, give 30mg of Morphine as needed.", {"entities": [(22, 26, "QUANTITY"), (30, 38, "MEDICATION")]}),
    ("Dispense 15ml of cough syrup every 6 hours.", {"entities": [(9, 13, "QUANTITY"), (17, 28, "MEDICATION"), (29, 42, "WHEN")]}),
    ("Take one 50mg tablet of Losartan every morning.", {"entities": [(9, 13, "QUANTITY"), (24, 32, "MEDICATION"), (33, 46, "WHEN")]}),
    ("Patient has prescribed 200mg of Aspirin two times daily.", {"entities": [(23, 28, "QUANTITY"), (32, 39, "MEDICATION"), (40, 55, "WHEN")]}),
])

# Prepare TRAIN_DATA with BILUO tags
TRAIN_DATA_BILUO = []
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    tags = spacy.training.offsets_to_biluo_tags(doc, annotations['entities'])

    # Check for misalignment and report if necessary
    if '-' in tags:
        print(f"Misaligned entity in text: '{text}'")
        continue  # Skip adding this to training data if misaligned

    TRAIN_DATA_BILUO.append((doc, tags))

# Check if TRAIN_DATA_BILUO is empty
if not TRAIN_DATA_BILUO:
    raise ValueError("All training data items were misaligned and ignored. Review entity offsets.")


# Add the new entity label to the NER pipeline
# if "NER" not in nlp.pipe_names:
#     ner = nlp.create_pipe("ner")
#     nlp.add_pipe("ner")
# else:
ner = nlp.get_pipe("ner")

# Add new entity labels to the NER model
ner.add_label("MEDICATION")
ner.add_label("QUANTITY")
ner.add_label("PERSON")
ner.add_label("WHEN")

# Disable other components to focus training on NER
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()
    for itn in range(10):  # Number of iterations
        random.shuffle(TRAIN_DATA)
        losses = {}
        #for text, annotations in TRAIN_DATA:
        for doc, tags in TRAIN_DATA_BILUO:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            #example = Example.from_dict(doc, {"words": [token.text for token in doc], "entities": tags})
            nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
        #print("Iteration {}, Losses: {}".format(itn, losses))

# Interactive user input for NER testing
while True:
    test_text = input("Enter your text for entity recognition or type 'exit' to quit: ")
    if test_text.lower() == 'exit':
        break
    doc = nlp(test_text)
    for ent in doc.ents:
        print(f"{ent.text} ({ent.label_})")

#nlp.to_disk("/path/to/updated_model")

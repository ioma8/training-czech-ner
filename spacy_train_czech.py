import os
import xml.etree.ElementTree as ET

# TRAIN_DATA = [
#     ('Who is Shaka Khan?', {
#         'entities': [(7, 17, 'PERSON')]
#     }),
#     ('I like London and Berlin.', {
#         'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
#     }),
# ]

import spacy
import random
from spacy.util import minibatch, compounding
from spacy.training import Example


def train_ner_spacy(train_data):
    # start with blank Czech model
    spacy.prefer_gpu()
    nlp = spacy.blank('cs')
    # create the built-in pipeline components and add them to the pipeline
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner', last=True)
    else:
        ner = nlp.get_pipe('ner')
    # add labels
    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.initialize()

        for itn in range(100):  # number of iterations can be adjusted
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=100)
            for batch in batches:
                examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
                nlp.update(
                    examples,  # batch of Example objects
                    drop=0.2,  # dropout - make it harder to memorise data
                    losses=losses,
                    sgd=optimizer
                )
            print('Losses', losses)
    # save the model
    output_dir = os.path.join(os.getcwd(), "cs_ner")
    nlp.to_disk(output_dir)
    print('Saved model to', output_dir)


def extract_inner_text(element):
    inner_text = element.text if element.text else ''
    for child in element:
        inner_text += extract_inner_text(child)
    inner_text += element.tail if element.tail else ''
    return inner_text


def get_inner_text(element):
    inner_text = element.text or ''
    for child in element:
        child_text = ET.tostring(child, encoding='unicode', method='text')
        inner_text += child_text
    return inner_text


def convert_input_file(input_file):
    data = []
    with open(input_file, 'r') as file:
        for line in file:
            line = "<doc>" + line.strip() + "</doc>"
            root = ET.fromstring(line)

            text = extract_inner_text(root)
            entities_data = []
            elements = root.findall('ne')
            for ne in elements:
                entity_type = ne.get('type').upper()
                entity_text = get_inner_text(ne).strip()
                if entity_text is None:
                    continue
                start = text.find(entity_text)
                end = start + len(entity_text)
                if start == -1:
                    continue
                entities_data.append((start, end, entity_type))

            entry = (text, {'entities': entities_data})
            print(entry)
            data.append(entry)

    return data


def modify_train_data(train_data):
    modified_data = []
    for text, annotations in train_data:
        entities = annotations['entities']
        entities = sorted(entities, key=lambda x: x[0])  # Sort entities by start position

        new_entities = []
        prev_end = -1
        for start, end, entity_type in entities:
            if start >= prev_end:  # Check if the current entity starts after the previous entity ends
                new_entities.append((start, end, entity_type))
                prev_end = end

        modified_data.append((text, {'entities': new_entities}))

    return modified_data


if __name__ == '__main__':
    train_data = convert_input_file("named_ent_xml_simple_cleared.txt")
    train_data = modify_train_data(train_data)
    train_ner_spacy(train_data)

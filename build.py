from common import load_knowledge, get_knowledge_hash
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import hashlib


def load_index(fname):
    vecdb = AnnoyIndex(384, "angular")
    vecdb.load(fname + ".index")

    with open(fname + ".pickle", "rb") as f:
        mapping = pickle.loads(f.read())

    digest = get_knowledge_hash()
    with open(fname + ".sha512", "rt") as f:
        stored_digest = f.read()

    if stored_digest != digest:
        print("Warning: Index out of data... consider re-building it")

    return {"vecdb": vecdb, "mapping": mapping, "digest": stored_digest}


def build_index(fname):
    knowledge = load_knowledge()

    sentenceTransformerModel = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    mapping = []
    vecdb = AnnoyIndex(384, "angular")
    for tag, entry in tqdm(knowledge.items()):
        vecdb.add_item(len(mapping), sentenceTransformerModel.encode(entry["document"]))
        mapping.append(tag)

        if "topics" in entry:
            for topic in entry["topics"]:
                vecdb.add_item(len(mapping), sentenceTransformerModel.encode(topic))
                mapping.append(tag)

    vecdb.build(10)
    vecdb.save(fname + ".index")

    with open(fname + ".pickle", "wb") as f:
        f.write(pickle.dumps(mapping))

    digest = get_knowledge_hash()
    with open(fname + ".sha512", "wt") as f:
        f.write(digest)

    return {"vecdb": vecdb, "mapping": mapping, "digest": digest}

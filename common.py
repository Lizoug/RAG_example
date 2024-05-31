import json
import hashlib


def get_knowledge_hash():
    m = hashlib.sha512()
    with open("knowledge.json", "rt", encoding="utf-8") as f:
        m.update(f.read().encode("utf-8"))
    return m.hexdigest()


def load_knowledge():
    with open("knowledge.json", "rt", encoding="utf-8") as f:
        return json.loads(f.read())


def save_knowledge(knowledge):
    with open("knowledge.json", "wt", encoding="utf-8") as f:
        f.write(json.dumps(knowledge, indent=3, ensure_ascii=False))

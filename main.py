from tqdm import tqdm
import ollama
import argparse
from build import build_index, load_index
from sentence_transformers import SentenceTransformer
from common import load_knowledge, save_knowledge
from uuid import uuid4
import json
from suggest import suggest


def display_knowledge(knowledge, tag):
    doc = {tag: knowledge[tag]}
    print(json.dumps(doc, indent=3, ensure_ascii=False))


knowledge = load_knowledge()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", action="store_true", help="Build fresh index based on current knowledge"
)
parser.add_argument(
    "--query", action="store", default=None, help="Execute RAG Query to LLM"
)
parser.add_argument("--index", action="store", default="vecdb", help="IndexDB to use")
parser.add_argument(
    "--new",
    action="store",
    default=None,
    help="Add a new knowledge document to database. Can specify tag optionally by using --tag",
)
parser.add_argument(
    "--delete",
    action="store_true",
    default=None,
    help="Delete a document from database. Use --tag to specify which document",
)
parser.add_argument(
    "--show", action="store_true", help="Display a document identified by its --tag"
)
parser.add_argument(
    "--tag", action="store", help="Used to specify the tag of a document"
)
parser.add_argument(
    "--suggest",
    action="store_true",
    help="Suggest topics for a given document specified by --tag",
)
parser.add_argument(
    "--save",
    action="store_true",
    help="Set this flag to save the suggestions made by the --suggest flag into the knowledge database",
)
args = parser.parse_args()

if args.new:
    if args.tag is None:
        tag = str(uuid4())
    else:
        tag = args.tag
        if tag in knowledge:
            print(f"Tag {tag} already in use")
            exit()

    knowledge[tag] = {"document": args.new}
    save_knowledge(knowledge)
    display_knowledge(knowledge, tag)
    exit()

if args.tag:
    couldBe = []

    for key in knowledge.keys():
        if key.startswith(args.tag):
            couldBe.append(key)

    if len(couldBe) == 0:
        print(f"Unknown knowledge tag {args.tag}")
        exit()

    if len(couldBe) > 1:
        print(f"Ambigous knowledge tag.. possible tags are")
        print(json.dumps(couldBe, indent=3))
        exit()

    args.tag = couldBe[0]

if args.build:
    print("Building new index...")
    index = build_index(args.index)
    exit()
else:
    index = load_index(args.index)

if args.show:
    display_knowledge(knowledge, args.tag)
    exit()

if args.delete:
    if args.tag is None:
        print("Specify tag to delete by using --tag")
        exit()

    del knowledge[args.tag]
    save_knowledge(knowledge)
    exit()

if args.suggest:
    if args.tag is None:
        print("Specify tag to suggest topics by using --tag")
        exit()

    suggest(knowledge, args.tag, index, args.save)

    exit()

if args.query is None:
    print("Please specify --query parameter")
    exit()


def retrieve_context(query, index, knowledge):
    sentenceTransformerModel = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    vec = sentenceTransformerModel.encode(query)
    indices = index["vecdb"].get_nns_by_vector(vec, 1)
    knowledgeTag = index["mapping"][indices[0]]
    return knowledge[knowledgeTag]["document"]


def clean_mistral_response(response):
    response = response.strip()
    openBrackets = 0
    index = 1
    while True:
        if response[-index] == ")":
            openBrackets += 1
        elif response[-index] == "(":
            openBrackets -= 1

        if openBrackets <= 0:
            break

        if index >= len(response):
            index = 1
            break

        index += 1

    response = response[:-index]

    return response


def prompt_llm(query, context):
    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": "Hier ist ein Dokument:\n\n"
                + context
                + "\n\nAnswer the following question: "
                + query
                + "\n\nAnswer only in German!",
            },
        ],
    )

    return clean_mistral_response(response["message"]["content"])


context = retrieve_context(args.query, index, knowledge)
answer = prompt_llm(args.query, context)

print("Query:   ", args.query)
print("Context: ", context)
print("Answer:  ", answer)

from tqdm import tqdm
import ollama
from sentence_transformers import SentenceTransformer
from common import save_knowledge


def _prompt_llm(context):
    return ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": "Hier ist ein Dokument:\n\n"
                + context
                + "\n\nSchlage eine Frage vor die sich auf das Dokument bezieht."
                + "\n\nAntworte nur in deutsch.",
            },
        ],
    )
    # return ollama.chat(
    #     model="mistral",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "This is a document:\n\n"
    #             + context
    #             + "\n\nSuggest a question related to this document"
    #             + "\n\nAnwser in english only",
    #         },
    #     ],
    # )


def suggest(knowledge, tag, index, store):
    if store is None:
        store = False

    context = knowledge[tag]["document"]
    sentenceTransformerModel = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    print("Thinking about good questions...")

    suggestions = []
    questions = []
    for i in tqdm(range(30)):
        # Generate a unique question
        while True:
            response = _prompt_llm(context)
            question = response["message"]["content"]
            if question in questions:
                continue

            questions.append(question)
            break

        # Encode it
        vec = sentenceTransformerModel.encode(question)

        # Find best matching document to this question
        result = index["vecdb"].get_nns_by_vector(vec, 1, include_distances=True)
        idx = result[0][0]
        distance = result[1][0]

        knowledgeTag = index["mapping"][idx]
        if knowledgeTag == tag:
            suggestions.append((question, distance))

    suggestions.sort(key=lambda x: x[1], reverse=True)
    suggestions = suggestions[:10]

    for suggestion in suggestions:
        print(" * ", suggestion[0])

    if store:
        if "topics" not in knowledge[tag]:
            knowledge[tag]["topics"] = []

        for suggestion in suggestions:
            topic = suggestion[0]
            knowledge[tag]["topics"].append(topic)

        save_knowledge(knowledge)

    exit()

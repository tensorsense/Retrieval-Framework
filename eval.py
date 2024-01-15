from factory import build_rag_chain

from trulens_eval import TruChain, Feedback, Tru
from trulens_eval.feedback.provider import AzureOpenAI
from trulens_eval.feedback import Groundedness
from trulens_eval.app import App

import numpy as np
from demo import config

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv("azure.env"))

QUESTIONS = [
    "What is the link between AU12 and cognitive load?",
    "How does cognitive load affect facial expressions?",
    "What action units get activated under heavy cognitive load?",
]

tru = Tru()
tru.reset_database()


def init_feedbacks(chain):
    # Initialize provider class
    openai = AzureOpenAI(deployment_name=config.AZURE_OPENAI_DEPLOYMENT_AGENT)

    # select context to be used in feedback. the location of context is app specific.

    context = App.select_context(chain)

    grounded = Groundedness(groundedness_provider=openai)

    # Define a groundedness feedback function
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons)
        .on(context.collect())  # collect context chunks into a list
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    # Question/answer relevance between overall question and answer.
    f_qa_relevance = Feedback(openai.relevance_with_cot_reasons).on_input_output()
    # Question/statement relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(openai.qs_relevance_with_cot_reasons)
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )
    return [f_qa_relevance, f_context_relevance, f_groundedness]


def run_eval(app_id):
    rag_chain = build_rag_chain()
    feedbacks = init_feedbacks(rag_chain)

    tru_recorder = TruChain(rag_chain,
                            app_id=app_id,
                            feedbacks=feedbacks)

    for q in QUESTIONS:
        with tru_recorder as recording:
            rag_chain.invoke(q)


if __name__ == "__main__":
    tru.run_dashboard(port=8000)
    run_eval("Chain1_ChatApplication")

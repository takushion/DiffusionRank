import argparse
import json
import os
import time
from functools import partial
from transformers.utils import logging

from llm4ranking.evaluation.evaluator import evaluate
from llm4ranking.ranker.base import ListwiseSilidingWindowReranker

from eval_utils import LladaForEval, PermutationListwiseWrapper

logging.set_verbosity_error()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-1.5")
    parser.add_argument("--rope-scaling-factor", type=float, default=1.0)

    parser.add_argument("--reranking-args", type=str, default="")
    parser.add_argument("--model-args", type=str, default="")

    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--retriever", type=str, default="bm25")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    print(args)

    model = LladaForEval(
        model_path=args.model,
        rope_scaling_factor=args.rope_scaling_factor,
        device="cuda",
    )

    ranker = ListwiseSilidingWindowReranker()
    rerank = partial(
        ranker.rerank,
        ranking_func=PermutationListwiseWrapper(
            model, **eval(f"dict({args.model_args})") if args.model_args else {}
        ),
        **eval(f"dict({args.reranking_args})") if args.reranking_args else {},
    )

    results = evaluate(
        rerank,
        datasets=args.datasets,
        retriever=args.retriever,
        topk=args.topk,
        output_dir=os.path.join(
            "outputs", time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S")
        ),
    )

    if args.output_dir is not None:
        with open(args.output_dir, "a") as f:
            f.write(
                json.dumps({"args": vars(args), "results": results}, default=str) + "\n"
            )
        print(f"Results saved to {args.output_dir}")

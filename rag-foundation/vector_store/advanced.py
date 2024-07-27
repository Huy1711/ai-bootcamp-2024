from typing import List
from collections import defaultdict

from .node import VectorStoreQueryResult

class HybridSearch():

    @staticmethod
    def rrf_rerank(
        query_results: List[VectorStoreQueryResult],
        top_k: int = 3,
        k: float = 60.,
    ) -> VectorStoreQueryResult:
        
        node_rrf_dict = defaultdict(float)
        node_dict = {}
        for result in query_results:
            for i, node in enumerate(result.nodes):
                doc_rank = i + 1
                node_dict[node.id_] = node
                node_rrf_dict[node.id_] += 1. / (k + doc_rank)
        
        sorted_node_list = list(node_rrf_dict.items())
        sorted_node_list.sort(reverse=True, key = lambda x : x[1])
        sorted_node_list = sorted_node_list[:top_k]
        
        return VectorStoreQueryResult(
            nodes=[node_dict[node[0]] for node in sorted_node_list],
            similarities=[node[1] for node in sorted_node_list],
            ids=[node[0] for node in sorted_node_list],
        )


class Reranker():
    pass
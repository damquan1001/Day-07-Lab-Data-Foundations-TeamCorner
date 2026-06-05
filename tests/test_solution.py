"""
Day 7 — Data Foundations: Embedding & Vector Store
Test suite for student solution.

Run from the day folder:
    pytest tests/ -v

No real API keys required — uses mock embeddings.
"""

import importlib
import os
import unittest
from pathlib import Path

DAY_DIR = Path(__file__).parent.parent
PACKAGE_NAME = os.getenv("LAB_SOLUTION_PACKAGE", "src")

_m = importlib.import_module(PACKAGE_NAME)

Document = getattr(_m, 'Document')
EmbeddingStore = getattr(_m, 'EmbeddingStore')
KnowledgeBaseAgent = getattr(_m, 'KnowledgeBaseAgent')
_mock_embed = getattr(_m, '_mock_embed')
FixedSizeChunker = getattr(_m, 'FixedSizeChunker')
SentenceChunker = getattr(_m, 'SentenceChunker')
RecursiveChunker = getattr(_m, 'RecursiveChunker')
ChunkingStrategyComparator = getattr(_m, 'ChunkingStrategyComparator')
MockEmbedder = getattr(_m, 'MockEmbedder')
template = _m

SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "A fox is a small omnivorous mammal. "
    "Dogs are loyal companions and working animals. "
    "Brown bears live in forests across the northern hemisphere. "
    "Jumping is a physical activity that requires leg strength. "
)

LONG_TEXT = "word " * 200


class TestProjectStructure(unittest.TestCase):

    def test_root_main_entrypoint_exists(self):
        self.assertTrue((DAY_DIR / 'main.py').exists())

    def test_src_package_exists(self):
        self.assertTrue((DAY_DIR / 'src' / '__init__.py').exists())


class TestClassBasedInterfaces(unittest.TestCase):

    def test_chunker_classes_exist(self):
        self.assertTrue(all([FixedSizeChunker, SentenceChunker, RecursiveChunker, ChunkingStrategyComparator]))

    def test_mock_embedder_exists(self):
        embedder = MockEmbedder()
        self.assertEqual(len(embedder("hello")), 64)


class TestFixedSizeChunker(unittest.TestCase):

    def test_returns_list(self):
        chunks = FixedSizeChunker(chunk_size=50, overlap=0).chunk(SAMPLE_TEXT)
        self.assertIsInstance(chunks, list)

    def test_single_chunk_if_text_shorter(self):
        short = "hello world"
        chunks = FixedSizeChunker(chunk_size=100, overlap=0).chunk(short)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short)

    def test_chunks_respect_size(self):
        chunks = FixedSizeChunker(chunk_size=50, overlap=0).chunk(LONG_TEXT)
        for c in chunks[:-1]:
            self.assertLessEqual(len(c), 50)

    def test_correct_number_of_chunks_no_overlap(self):
        text = "a" * 100
        chunks = FixedSizeChunker(chunk_size=25, overlap=0).chunk(text)
        self.assertEqual(len(chunks), 4)

    def test_overlap_creates_shared_content(self):
        text = "abcdefghijklmnopqrst"
        chunks = FixedSizeChunker(chunk_size=10, overlap=2).chunk(text)
        if len(chunks) >= 2:
            overlap_from_first = chunks[0][-2:]
            start_of_second = chunks[1][:2]
            self.assertEqual(overlap_from_first, start_of_second)

    def test_no_overlap_no_shared_content(self):
        text = "abcdefghij"
        chunks = FixedSizeChunker(chunk_size=5, overlap=0).chunk(text)
        self.assertEqual(chunks[0], "abcde")
        self.assertEqual(chunks[1], "fghij")

    def test_empty_text_returns_empty_list(self):
        chunks = FixedSizeChunker(chunk_size=50, overlap=0).chunk("")
        self.assertIsInstance(chunks, list)


class TestSentenceChunker(unittest.TestCase):

    def test_returns_list(self):
        chunks = SentenceChunker(max_sentences_per_chunk=2).chunk(SAMPLE_TEXT)
        self.assertIsInstance(chunks, list)

    def test_respects_max_sentences(self):
        chunks = SentenceChunker(max_sentences_per_chunk=2).chunk(SAMPLE_TEXT)
        self.assertGreaterEqual(len(chunks), 2)

    def test_single_sentence_max_gives_many_chunks(self):
        chunks_1 = SentenceChunker(max_sentences_per_chunk=1).chunk(SAMPLE_TEXT)
        chunks_3 = SentenceChunker(max_sentences_per_chunk=3).chunk(SAMPLE_TEXT)
        self.assertGreaterEqual(len(chunks_1), len(chunks_3))

    def test_chunks_are_strings(self):
        chunks = SentenceChunker(max_sentences_per_chunk=2).chunk(SAMPLE_TEXT)
        for c in chunks:
            self.assertIsInstance(c, str)


class TestRecursiveChunker(unittest.TestCase):

    def test_returns_list(self):
        chunks = RecursiveChunker(chunk_size=100).chunk(SAMPLE_TEXT)
        self.assertIsInstance(chunks, list)

    def test_chunks_within_size_when_possible(self):
        chunks = RecursiveChunker(chunk_size=100).chunk(LONG_TEXT)
        within = sum(1 for c in chunks if len(c) <= 110)
        self.assertGreater(within, len(chunks) * 0.8)

    def test_empty_separators_falls_back_gracefully(self):
        text = "no separators here at all"
        chunks = RecursiveChunker(separators=[], chunk_size=100).chunk(text)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

    def test_handles_double_newline_separator(self):
        text = "paragraph one\n\nparagraph two\n\nparagraph three"
        chunks = RecursiveChunker(separators=["\n\n"], chunk_size=200).chunk(text)
        self.assertGreaterEqual(len(chunks), 1)


class TestEmbeddingStore(unittest.TestCase):

    def _make_store(self) -> EmbeddingStore:
        return EmbeddingStore(collection_name="test", embedding_fn=_mock_embed)

    def _make_docs(self, n: int = 3) -> list[Document]:
        return [
            Document(id=f"doc{i}", content=f"This is document number {i}.", metadata={})
            for i in range(n)
        ]

    def test_initial_size_is_zero(self):
        store = self._make_store()
        self.assertEqual(store.get_collection_size(), 0)

    def test_add_documents_increases_size(self):
        store = self._make_store()
        docs = self._make_docs(3)
        store.add_documents(docs)
        self.assertEqual(store.get_collection_size(), 3)

    def test_add_more_increases_further(self):
        store = self._make_store()
        store.add_documents(self._make_docs(2))
        store.add_documents(self._make_docs(3))
        self.assertEqual(store.get_collection_size(), 5)

    def test_search_returns_list(self):
        store = self._make_store()
        store.add_documents(self._make_docs(3))
        results = store.search("document", top_k=2)
        self.assertIsInstance(results, list)

    def test_search_returns_at_most_top_k(self):
        store = self._make_store()
        store.add_documents(self._make_docs(10))
        results = store.search("document", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_search_results_have_content_key(self):
        store = self._make_store()
        store.add_documents(self._make_docs(3))
        results = store.search("document", top_k=3)
        for r in results:
            self.assertIn("content", r)

    def test_search_results_have_score_key(self):
        store = self._make_store()
        store.add_documents(self._make_docs(3))
        results = store.search("document", top_k=3)
        for r in results:
            self.assertIn("score", r)

    def test_search_results_sorted_by_score_descending(self):
        store = self._make_store()
        store.add_documents(self._make_docs(5))
        results = store.search("document", top_k=5)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestKnowledgeBaseAgent(unittest.TestCase):

    def _make_agent(self) -> KnowledgeBaseAgent:
        store = EmbeddingStore(collection_name="kb_test", embedding_fn=_mock_embed)
        docs = [
            Document(id="d1", content="Python is a high-level programming language.", metadata={}),
            Document(id="d2", content="Machine learning uses algorithms to learn from data.", metadata={}),
            Document(id="d3", content="Vector databases store embeddings for similarity search.", metadata={}),
        ]
        store.add_documents(docs)
        return KnowledgeBaseAgent(store=store, llm_fn=lambda prompt: "Answer based on context.")

    def test_answer_returns_string(self):
        agent = self._make_agent()
        result = agent.answer("What is Python?")
        self.assertIsInstance(result, str)

    def test_answer_non_empty(self):
        agent = self._make_agent()
        result = agent.answer("What is machine learning?")
        self.assertGreater(len(result), 0)


class TestComputeSimilarity(unittest.TestCase):
    def test_identical_vectors_return_1(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(template.compute_similarity(v, v), 1.0, places=5)

    def test_orthogonal_vectors_return_0(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(template.compute_similarity(a, b), 0.0, places=5)

    def test_opposite_vectors_return_minus_1(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(template.compute_similarity(a, b), -1.0, places=5)

    def test_zero_vector_returns_0(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        self.assertEqual(template.compute_similarity(a, b), 0.0)


class TestCompareChunkingStrategies(unittest.TestCase):
    SAMPLE_TEXT = (
        "Artificial intelligence is transforming industries. "
        "Machine learning enables systems to learn from data. "
        "Deep learning uses neural networks with many layers. "
        "Natural language processing handles text understanding. "
        "Computer vision processes images and video streams."
    ) * 3

    def test_returns_three_strategies(self):
        result = ChunkingStrategyComparator().compare(self.SAMPLE_TEXT, chunk_size=100)
        self.assertIn('fixed_size', result)
        self.assertIn('by_sentences', result)
        self.assertIn('recursive', result)

    def test_each_strategy_has_count_and_avg_length(self):
        result = ChunkingStrategyComparator().compare(self.SAMPLE_TEXT, chunk_size=100)
        for strategy_name, stats in result.items():
            self.assertIn('count', stats)
            self.assertIn('avg_length', stats)
            self.assertIn('chunks', stats)

    def test_counts_are_positive(self):
        result = ChunkingStrategyComparator().compare(self.SAMPLE_TEXT, chunk_size=100)
        for strategy_name, stats in result.items():
            self.assertGreater(stats['count'], 0)


class TestEmbeddingStoreSearchWithFilter(unittest.TestCase):
    def setUp(self):
        self.store = template.EmbeddingStore("test_filter")
        docs = [
            template.Document("doc1", "Python programming tutorial", {"department": "engineering", "lang": "en"}),
            template.Document("doc2", "Marketing strategy guide", {"department": "marketing", "lang": "en"}),
            template.Document("doc3", "Kỹ thuật lập trình Python", {"department": "engineering", "lang": "vi"}),
        ]
        self.store.add_documents(docs)

    def test_filter_by_department(self):
        results = self.store.search_with_filter("programming", top_k=5, metadata_filter={"department": "engineering"})
        for r in results:
            self.assertEqual(r['metadata']['department'], 'engineering')

    def test_no_filter_returns_all_candidates(self):
        results_filtered = self.store.search_with_filter("programming", top_k=10, metadata_filter=None)
        results_unfiltered = self.store.search("programming", top_k=10)
        self.assertEqual(len(results_filtered), len(results_unfiltered))

    def test_returns_at_most_top_k(self):
        results = self.store.search_with_filter("programming", top_k=1, metadata_filter={"department": "engineering"})
        self.assertLessEqual(len(results), 1)


class TestEmbeddingStoreDeleteDocument(unittest.TestCase):
    def setUp(self):
        self.store = template.EmbeddingStore("test_delete")
        self.store.add_documents([
            template.Document("doc_to_delete", "Content that will be removed", {}),
            template.Document("doc_to_keep", "Content that stays", {}),
        ])

    def test_delete_returns_true_for_existing_doc(self):
        result = self.store.delete_document("doc_to_delete")
        self.assertTrue(result)

    def test_delete_returns_false_for_nonexistent_doc(self):
        result = self.store.delete_document("does_not_exist")
        self.assertFalse(result)

    def test_delete_reduces_collection_size(self):
        size_before = self.store.get_collection_size()
        self.store.delete_document("doc_to_delete")
        size_after = self.store.get_collection_size()
        self.assertLess(size_after, size_before)


class TestEmbeddingStoreBM25(unittest.TestCase):
    def setUp(self):
        self.store = template.EmbeddingStore("test_bm25")
        self.store.add_documents([
            template.Document(
                "policy_housing",
                "Housing assistance policy covers rent relief, eligibility, and application deadlines.",
                {"jurisdiction": "city", "policy_area": "housing", "language": "en"},
            ),
            template.Document(
                "policy_tax",
                "Tax exemption policy explains business filings and income thresholds.",
                {"jurisdiction": "state", "policy_area": "tax", "language": "en"},
            ),
            template.Document(
                "policy_health",
                "Public health policy describes vaccination clinics and safety reporting.",
                {"jurisdiction": "city", "policy_area": "health", "language": "en"},
            ),
        ])

    def test_exact_keyword_query_ranks_matching_policy_first(self):
        results = self.store.search_bm25("rent relief eligibility", top_k=3)
        self.assertEqual(results[0]["doc_id"], "policy_housing")
        self.assertGreater(results[0]["score"], results[-1]["score"])

    def test_bm25_results_have_expected_keys_and_sorted_scores(self):
        results = self.store.search_bm25("policy", top_k=2)
        self.assertLessEqual(len(results), 2)
        for result in results:
            self.assertIn("content", result)
            self.assertIn("metadata", result)
            self.assertIn("score", result)

        scores = [result["score"] for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_bm25_filter_by_metadata(self):
        results = self.store.search_bm25_with_filter(
            "policy",
            top_k=10,
            metadata_filter={"jurisdiction": "city"},
        )
        for result in results:
            self.assertEqual(result["metadata"]["jurisdiction"], "city")

    def test_bm25_no_filter_matches_unfiltered_count(self):
        results_filtered = self.store.search_bm25_with_filter("policy", top_k=10, metadata_filter=None)
        results_unfiltered = self.store.search_bm25("policy", top_k=10)
        self.assertEqual(len(results_filtered), len(results_unfiltered))

    def test_bm25_empty_query_or_non_positive_top_k_returns_empty(self):
        self.assertEqual(self.store.search_bm25("", top_k=5), [])
        self.assertEqual(self.store.search_bm25("policy", top_k=0), [])

    def test_hybrid_results_have_component_scores_and_sorted_scores(self):
        results = self.store.search_hybrid("rent relief eligibility", top_k=3)
        self.assertGreater(len(results), 0)
        for result in results:
            self.assertIn("score", result)
            self.assertIn("bm25_score", result)
            self.assertIn("vector_score", result)

        scores = [result["score"] for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_hybrid_filter_by_metadata(self):
        results = self.store.search_hybrid_with_filter(
            "policy",
            top_k=10,
            metadata_filter={"jurisdiction": "city"},
        )
        for result in results:
            self.assertEqual(result["metadata"]["jurisdiction"], "city")

    def test_hybrid_empty_query_or_non_positive_top_k_returns_empty(self):
        self.assertEqual(self.store.search_hybrid("", top_k=5), [])
        self.assertEqual(self.store.search_hybrid("policy", top_k=0), [])

    def test_hybrid_keyword_weight_keeps_exact_match_first(self):
        results = self.store.search_hybrid("rent relief eligibility", top_k=3, bm25_weight=0.9)
        self.assertEqual(results[0]["doc_id"], "policy_housing")


class TestAILawRAGData(unittest.TestCase):
    def test_ai_law_loader_splits_articles_with_metadata(self):
        main_module = importlib.import_module("main")
        law_path = DAY_DIR / "data" / "Luật Trí tuệ nhân tạo.md"
        docs = main_module.load_documents_from_files([str(law_path)])

        self.assertGreaterEqual(len(docs), 30)
        article_numbers = {doc.metadata.get("article_number") for doc in docs}
        self.assertIn("7", article_numbers)
        self.assertIn("34", article_numbers)

        for doc in docs:
            self.assertEqual(doc.metadata["document_type"], "law")
            self.assertEqual(doc.metadata["language"], "vi")
            self.assertIn("article", doc.metadata)

    def test_ai_law_bm25_retrieves_effective_date_article(self):
        main_module = importlib.import_module("main")
        law_path = DAY_DIR / "data" / "Luật Trí tuệ nhân tạo.md"
        docs = main_module.load_documents_from_files([str(law_path)])
        store = template.EmbeddingStore("ai_law_bm25")
        store.add_documents(docs)

        results = store.search_bm25_with_filter(
            "Luật Trí tuệ nhân tạo có hiệu lực từ ngày nào?",
            top_k=3,
            metadata_filter={"document_type": "law", "language": "vi"},
        )
        top_articles = {result["metadata"]["article_number"] for result in results}
        self.assertIn("34", top_articles)

    def test_ai_law_hybrid_retrieves_transition_article(self):
        main_module = importlib.import_module("main")
        law_path = DAY_DIR / main_module.SAMPLE_FILES[0]
        docs = main_module.load_documents_from_files([str(law_path)])
        store = template.EmbeddingStore("ai_law_hybrid")
        store.add_documents(docs)

        results = store.search_hybrid_with_filter(
            main_module.LAW_BENCHMARK_QUERIES[4],
            top_k=3,
            metadata_filter={"document_type": "law", "language": "vi"},
        )
        self.assertEqual(results[0]["metadata"]["article_number"], "35")


class TestChatDemoBackend(unittest.TestCase):
    def test_chat_demo_preset_questions_returns_five_questions(self):
        chat_demo = importlib.import_module("chat_demo")
        questions = chat_demo.get_preset_questions()

        self.assertEqual(len(questions), 5)
        self.assertTrue(all(isinstance(question, str) and question for question in questions))

    def test_chat_demo_search_returns_results_for_each_strategy(self):
        chat_demo = importlib.import_module("chat_demo")
        question = chat_demo.get_preset_questions()[0]

        for strategy in ("vector", "bm25", "hybrid"):
            result = chat_demo.search_demo(question, strategy=strategy, top_k=3)
            self.assertEqual(result["strategy"], strategy)
            self.assertGreater(len(result["results"]), 0)
            self.assertIn("answer", result)

    def test_chat_demo_compare_returns_three_strategies(self):
        chat_demo = importlib.import_module("chat_demo")
        result = chat_demo.compare_demo(chat_demo.get_preset_questions()[0], top_k=1)

        self.assertEqual(len(result["comparisons"]), 3)
        self.assertEqual(
            {comparison["strategy"] for comparison in result["comparisons"]},
            {"vector", "bm25", "hybrid"},
        )

    def test_chat_demo_hybrid_retrieves_expected_benchmark_articles(self):
        chat_demo = importlib.import_module("chat_demo")
        expected_articles = ["34", "7", "26", "30", "35"]

        for question, expected_article in zip(chat_demo.get_preset_questions(), expected_articles):
            result = chat_demo.search_demo(question, strategy="hybrid", top_k=1)
            self.assertEqual(result["results"][0]["article_number"], expected_article)


if __name__ == "__main__":
    unittest.main()

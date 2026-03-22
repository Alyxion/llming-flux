"""Tests for the pluggable query expander."""

from llming_flux.query_expander import QueryExpander, build_fts_query, clean_word


class TestCleanWord:
    def test_basic(self):
        assert clean_word("Hello!") == "hello"

    def test_hyphenated(self):
        assert clean_word("IDK-120") == "idk-120"

    def test_leading_trailing_hyphens(self):
        assert clean_word("-test-") == "test"

    def test_punctuation(self):
        assert clean_word("test.") == "test"
        assert clean_word("test,") == "test"


class TestBuildFTSQuery:
    def test_single_term(self):
        assert build_fts_query(["widget"]) == "widget*"

    def test_multiple_terms(self):
        q = build_fts_query(["widget", "drift"])
        assert "widget*" in q
        assert "drift*" in q
        assert " OR " in q

    def test_phrase(self):
        q = build_fts_query(["drift reduction"])
        assert '"drift reduction"' in q

    def test_hyphenated(self):
        q = build_fts_query(["IDK-120"])
        assert '"IDK-120"' in q

    def test_empty(self):
        assert build_fts_query([]) == ""


class TestQueryExpander:
    def test_basic_expansion(self):
        qe = QueryExpander()
        terms = qe.expand("What is the best widget?")
        assert "widget" in terms
        assert "what" not in terms  # stopword
        assert "the" not in terms  # stopword
        assert "is" not in terms  # stopword

    def test_short_words_filtered(self):
        qe = QueryExpander()
        terms = qe.expand("a b cd efg")
        assert "a" not in terms
        assert "b" not in terms
        assert "cd" in terms

    def test_custom_stopwords(self):
        qe = QueryExpander(stopwords={"custom", "stop"})
        terms = qe.expand("custom stop word")
        assert "custom" not in terms
        assert "stop" not in terms
        assert "word" in terms

    def test_build_fts_convenience(self):
        qe = QueryExpander()
        fts = qe.build_fts("widget performance")
        assert "widget*" in fts
        assert "drift*" in fts

    def test_subclass_expansion(self):
        class MyExpander(QueryExpander):
            def expand(self, query):
                terms = super().expand(query)
                # Add a synonym
                if "widget" in terms:
                    terms.append("gadget")
                return terms

        qe = MyExpander()
        terms = qe.expand("widget test")
        assert "gadget" in terms
        assert "widget" in terms

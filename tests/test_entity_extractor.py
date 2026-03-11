"""Tests for EntityExtractor regex-based entity extraction."""

from __future__ import annotations

from engine import EntityExtractor


class TestEntityExtractor:
    def setup_method(self):
        self.extractor = EntityExtractor()

    # ── SQL table names ──

    def test_create_table(self):
        entities = self.extractor.extract_entities("CREATE TABLE users (id INT)")
        assert "users" in entities

    def test_insert_into(self):
        entities = self.extractor.extract_entities("INSERT INTO orders VALUES (1)")
        assert "orders" in entities

    def test_select_from(self):
        entities = self.extractor.extract_entities("SELECT * FROM payments WHERE id=1")
        assert "payments" in entities

    def test_join(self):
        entities = self.extractor.extract_entities("FROM accounts JOIN transactions ON a.id = t.account_id")
        assert "accounts" in entities
        assert "transactions" in entities

    def test_create_table_if_not_exists(self):
        entities = self.extractor.extract_entities("CREATE TABLE IF NOT EXISTS metrics (id INT)")
        assert "metrics" in entities

    def test_sql_stop_words_filtered(self):
        # "index" and similar common SQL words should be filtered
        entities = self.extractor.extract_entities("CREATE TABLE index (id INT)")
        assert "index" not in entities

    # ── CLI flags ──

    def test_cli_flag_eq(self):
        entities = self.extractor.extract_entities("--namespace=production --port=8080")
        assert "production" in entities
        assert "8080" in entities

    # ── Environment variables ──

    def test_env_var_assignment(self):
        entities = self.extractor.extract_entities("DATABASE_URL=postgres://localhost/db")
        assert "database_url" in entities

    def test_export_var(self):
        entities = self.extractor.extract_entities("export API_KEY=sk-12345")
        assert "api_key" in entities

    def test_bare_env_var(self):
        entities = self.extractor.extract_entities("Check your CEREBRAS_API_KEY setting")
        assert "cerebras_api_key" in entities

    # ── IP addresses and ports ──

    def test_ip_address(self):
        entities = self.extractor.extract_entities("Server at 10.0.0.1")
        assert "10.0.0.1" in entities

    def test_ip_with_port(self):
        entities = self.extractor.extract_entities("Connect to 192.168.1.1:5432")
        assert "192.168.1.1:5432" in entities

    # ── URLs ──

    def test_http_url(self):
        entities = self.extractor.extract_entities("Visit http://example.com/api/v1")
        assert "http://example.com/api/v1" in entities

    def test_https_url(self):
        entities = self.extractor.extract_entities("Docs at https://docs.example.com/guide")
        assert "https://docs.example.com/guide" in entities

    # ── API endpoints ──

    def test_api_endpoint(self):
        entities = self.extractor.extract_entities("POST /api/users/create")
        assert "/api/users/create" in entities

    # ── File paths ──

    def test_file_path_py(self):
        entities = self.extractor.extract_entities("Edit engine.py")
        assert "engine.py" in entities

    def test_file_path_json(self):
        entities = self.extractor.extract_entities("Check config.json")
        assert "config.json" in entities

    def test_absolute_path(self):
        entities = self.extractor.extract_entities("File at /usr/local/bin/script.sh")
        assert "/usr/local/bin/script.sh" in entities

    # ── Version numbers ──

    def test_version_number(self):
        entities = self.extractor.extract_entities("Upgrade to v2.3.1")
        assert "v2.3.1" in entities

    def test_version_without_v(self):
        entities = self.extractor.extract_entities("Version 1.0.0")
        assert "1.0.0" in entities

    # ── API keys ──

    def test_stripe_live_key(self):
        entities = self.extractor.extract_entities("Key: sk_live_abc123def456")
        assert "sk_live_abc123def456" in entities

    def test_stripe_test_key(self):
        entities = self.extractor.extract_entities("Key: sk_test_xyz789")
        assert "sk_test_xyz789" in entities

    # ── Error codes ──

    def test_error_code(self):
        entities = self.extractor.extract_entities("Error PAY-4012-RETRY-EXHAUSTED occurred")
        assert "pay-4012-retry-exhausted" in entities

    # ── Tickets ──

    def test_jira_ticket(self):
        entities = self.extractor.extract_entities("Fix INC-7734 urgently")
        assert "inc-7734" in entities

    def test_sre_ticket(self):
        entities = self.extractor.extract_entities("See SRE-2291 for details")
        assert "sre-2291" in entities

    # ── JSON keys ──

    def test_json_key(self):
        entities = self.extractor.extract_entities('{"user_id": 123, "order_total": 99.99}')
        assert "user_id" in entities
        assert "order_total" in entities

    # ── Snake_case identifiers ──

    def test_snake_case_identifier(self):
        entities = self.extractor.extract_entities("Check the billing_events table")
        assert "billing_events" in entities

    def test_short_snake_case_filtered(self):
        # Identifiers <= 4 chars should be excluded
        entities = self.extractor.extract_entities("Use a_b as variable")
        assert "a_b" not in entities

    def test_stop_words_filtered(self):
        entities = self.extractor.extract_entities("The error_code is critical")
        assert "error_code" not in entities  # In stop words

    # ── Edge cases ──

    def test_empty_input(self):
        entities = self.extractor.extract_entities("")
        assert entities == set()

    def test_unicode_input(self):
        entities = self.extractor.extract_entities("サーバー 192.168.0.1 で問題が発生")
        assert "192.168.0.1" in entities

    def test_mixed_content(self):
        text = "Fix INC-7734: error in billing_events table at 10.0.0.1:5432, see /api/payments/retry"
        entities = self.extractor.extract_entities(text)
        assert "inc-7734" in entities
        assert "billing_events" in entities
        assert "10.0.0.1:5432" in entities
        assert "/api/payments/retry" in entities

    # ── _expand_entities (tested via EntityAwareScorer._expand_entities static method) ──

    def test_expand_entities_underscore(self):
        from engine import EntityAwareScorer

        expanded = EntityAwareScorer._expand_entities({"billing_events"})
        assert "billing events" in expanded
        assert "billing-events" in expanded

    def test_expand_entities_hyphen(self):
        from engine import EntityAwareScorer

        expanded = EntityAwareScorer._expand_entities({"retry-exhausted"})
        assert "retry_exhausted" in expanded
        assert "retry exhausted" in expanded

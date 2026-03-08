#!/usr/bin/env python3
"""Boilerplate NIAH — tests TF-IDF uniqueness signal weakness with repetitive critical content.

PROBLEM: The uniqueness signal (60% of TF-IDF score) penalizes content that is
similar to other content in the context. But sometimes repetitive content is
critical — a JSON schema, a CLI command, a SQL migration, an env var config,
or an error template that must be preserved EXACTLY.

DESIGN:
- 5 needles that are DELIBERATELY REPETITIVE (look like boilerplate):
  1. JSON schema for the /api/v2/orders response format
  2. Exact CLI deploy command with specific flags
  3. Database migration SQL for the billing_events table
  4. Env var config block for the Stripe integration
  5. Error message template for the payment retry flow
- Filler is SIMILAR-LOOKING content: other JSON schemas, CLI commands, SQL
  migrations, config blocks, error templates — same structure, different details
- Recall question asks for the SPECIFIC version (exact field, exact flag, exact
  table name, exact env var value, exact error code)
- 30 turns, 8k window, 10 sessions

Expected results:
- Goal-Guided TF-IDF: expected to STRUGGLE (uniqueness penalizes repetitive needles)
- Hardcoded priority: ceiling (always preserves needles)
- Keyword-only: expected to struggle (keywords overlap heavily)
- Naive sliding window: baseline
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine import ChunkLog, _estimate_tokens, GoalGuidedScorer

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_CONTEXT_TOKENS = 8000
SYSTEM_PROMPT = "You are a helpful assistant with perfect memory. Answer questions based on the conversation history provided."
SYSTEM_PROMPT_TOKENS = _estimate_tokens(SYSTEM_PROMPT)

# --- 5 Needles: critical boilerplate that LOOKS like other boilerplate ---
# Each needle contains a SPECIFIC detail that distinguishes it from similar filler.

NEEDLES = [
    {
        "id": "needle_1",
        "fact": (
            'IMPORTANT — The /api/v2/orders response schema MUST follow this EXACT format:\n'
            '```json\n'
            '{\n'
            '  "order_id": "string (UUID v4)",\n'
            '  "status": "pending | confirmed | shipped | delivered | cancelled",\n'
            '  "customer_id": "string (UUID v4)",\n'
            '  "line_items": [{"sku": "string", "quantity": "integer", "unit_price_cents": "integer"}],\n'
            '  "subtotal_cents": "integer",\n'
            '  "tax_cents": "integer",\n'
            '  "shipping_cents": "integer",\n'
            '  "total_cents": "integer",\n'
            '  "currency": "ISO 4217 (e.g. USD)",\n'
            '  "idempotency_key": "string (client-generated UUID v4)",\n'
            '  "created_at": "ISO 8601",\n'
            '  "updated_at": "ISO 8601"\n'
            '}\n'
            '```\n'
            'The idempotency_key field is REQUIRED and must be a client-generated UUID v4. '
            'This was agreed upon in the API contract review on February 12th.'
        ),
        "keyword": "idempotency_key",
    },
    {
        "id": "needle_2",
        "fact": (
            'CRITICAL — The exact production deploy command for the billing service is:\n'
            '```\n'
            'kubectl apply -f deploy/billing-v2.yaml --namespace=payments-prod '
            '--server-side --force-conflicts --field-manager=ci-deployer '
            '--prune --selector=app.kubernetes.io/part-of=billing-v2 '
            '--dry-run=none --validate=strict\n'
            '```\n'
            'You MUST use --server-side --force-conflicts together. Without --force-conflicts, '
            'the server-side apply will fail on the HPA resource due to ownership conflicts '
            'with the cluster autoscaler. The --field-manager=ci-deployer flag is required '
            'to match the CI service account permissions. Using any other field manager name '
            'will result in a 403 Forbidden from the API server.'
        ),
        "keyword": "ci-deployer",
    },
    {
        "id": "needle_3",
        "fact": (
            'REQUIRED MIGRATION — Run this SQL during the March maintenance window:\n'
            '```sql\n'
            'CREATE TABLE billing_events (\n'
            '    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n'
            '    subscription_id UUID NOT NULL REFERENCES subscriptions(id),\n'
            '    event_type VARCHAR(50) NOT NULL CHECK (event_type IN (\n'
            "        'charge_succeeded', 'charge_failed', 'refund_issued',\n"
            "        'dispute_opened', 'dispute_resolved', 'proration_applied'\n"
            '    )),\n'
            '    amount_cents INTEGER NOT NULL,\n'
            '    currency VARCHAR(3) NOT NULL DEFAULT \'USD\',\n'
            '    stripe_event_id VARCHAR(255) UNIQUE,\n'
            '    idempotency_hash VARCHAR(64) NOT NULL,\n'
            '    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),\n'
            '    CONSTRAINT unique_idempotency UNIQUE (subscription_id, idempotency_hash)\n'
            ');\n'
            'CREATE INDEX idx_billing_events_sub ON billing_events(subscription_id, created_at DESC);\n'
            'CREATE INDEX idx_billing_events_stripe ON billing_events(stripe_event_id) WHERE stripe_event_id IS NOT NULL;\n'
            '```\n'
            'The idempotency_hash constraint prevents duplicate event processing from Stripe webhooks.'
        ),
        "keyword": "billing_events",
    },
    {
        "id": "needle_4",
        "fact": (
            'IMPORTANT — The Stripe integration env vars MUST be set exactly as follows in production:\n'
            '```\n'
            'STRIPE_API_KEY=sk_live_51N8xYZABCdef_payments_v2_prod\n'
            'STRIPE_WEBHOOK_SECRET=whsec_8kLmN3pQrStUvWxYz\n'
            'STRIPE_CONNECT_CLIENT_ID=ca_HjKlMnOpQrStUv\n'
            'STRIPE_PAYMENT_METHODS=card,us_bank_account,sepa_debit\n'
            'STRIPE_WEBHOOK_TOLERANCE_SECONDS=300\n'
            'STRIPE_IDEMPOTENCY_TTL_HOURS=48\n'
            'STRIPE_MAX_RETRY_ATTEMPTS=5\n'
            'STRIPE_RETRY_BACKOFF_BASE_MS=1000\n'
            '```\n'
            'The STRIPE_WEBHOOK_TOLERANCE_SECONDS=300 (5 minutes) is critical — the default of 30s '
            'causes webhook signature verification failures during peak load when event delivery '
            'is delayed. This was the root cause of the February 8th payment processing outage.'
        ),
        "keyword": "whsec_8kLmN3pQrStUvWxYz",
    },
    {
        "id": "needle_5",
        "fact": (
            'IMPORTANT — The payment retry error template MUST use this exact format:\n'
            '```\n'
            'ERROR_CODE: PAY-4012-RETRY-EXHAUSTED\n'
            'SEVERITY: critical\n'
            'MESSAGE: Payment retry exhausted for subscription {subscription_id} after '
            '{retry_count} attempts over {elapsed_hours}h. Last failure: {last_error}. '
            'Stripe charge ID: {stripe_charge_id}. Customer notification sent: {notified}.\n'
            'ACTION: Escalate to payments-oncall via PagerDuty incident. '
            'Do NOT manually retry — the idempotency key may be stale.\n'
            'DASHBOARD: https://grafana.internal/d/pay-retries?sub={subscription_id}\n'
            '```\n'
            'The error code PAY-4012-RETRY-EXHAUSTED triggers the automatic PagerDuty escalation '
            'policy. Using any other error code format will NOT trigger the alert.'
        ),
        "keyword": "PAY-4012-RETRY-EXHAUSTED",
    },
]

RECALL_QUESTION = (
    "I need you to recall the EXACT boilerplate specifications from our conversation. "
    "These are critical — wrong details will break production systems:\n\n"
    "1. What is the /api/v2/orders response schema? What special field is REQUIRED and why?\n"
    "2. What is the exact kubectl deploy command for the billing service? What specific --field-manager value is needed?\n"
    "3. What is the SQL CREATE TABLE statement for billing events? What table name and what unique constraint?\n"
    "4. What are the Stripe integration env vars? What is the exact STRIPE_WEBHOOK_SECRET value?\n"
    "5. What is the payment retry error template? What is the exact ERROR_CODE?\n\n"
    "Answer with the EXACT details from our conversation. Do NOT paraphrase or guess."
)

# --- Boilerplate filler generators ---
# Each generates content that LOOKS LIKE the needles (same category: JSON schemas,
# CLI commands, SQL migrations, config blocks, error templates) but with DIFFERENT
# specific details. This maximizes TF-IDF similarity between needles and filler.

_ENDPOINTS = [
    "/api/v2/users", "/api/v2/products", "/api/v2/invoices", "/api/v2/shipments",
    "/api/v2/returns", "/api/v2/inventory", "/api/v2/coupons", "/api/v2/reviews",
    "/api/v2/subscriptions", "/api/v2/webhooks", "/api/v2/notifications",
    "/api/v2/analytics", "/api/v2/payments", "/api/v2/refunds", "/api/v2/disputes",
]

_DEPLOY_SERVICES = [
    "auth-service", "notification-service", "inventory-service", "analytics-service",
    "user-service", "search-service", "email-service", "reporting-service",
    "gateway-service", "scheduler-service", "cache-service", "logging-service",
]

_TABLE_NAMES = [
    "audit_events", "user_sessions", "notification_logs", "api_request_logs",
    "inventory_snapshots", "search_indexes", "email_templates", "report_schedules",
    "cache_entries", "job_schedules", "feature_flags", "rate_limit_counters",
    "webhook_deliveries", "access_tokens", "migration_history",
]

_SERVICES = [
    "Datadog", "PagerDuty", "Twilio", "SendGrid", "Redis", "Elasticsearch",
    "CloudWatch", "Sentry", "LaunchDarkly", "Segment", "Amplitude", "Mixpanel",
]

_ERROR_PREFIXES = [
    "AUTH", "INV", "SHIP", "NOTIF", "SYNC", "CACHE", "QUEUE", "SEARCH",
    "RATE", "CONN", "PARSE", "VALID", "PERM", "TOUT", "CONF",
]

_FILLER_GENERATORS = [
    # JSON schema fillers (5 generators)
    lambda rng: (
        f'API schema update — The {rng.choice(_ENDPOINTS)} response format:\n'
        f'```json\n'
        f'{{\n'
        f'  "id": "string (UUID v4)",\n'
        f'  "type": "string",\n'
        f'  "status": "{rng.choice(["active", "inactive", "pending"])} | {rng.choice(["archived", "deleted", "suspended"])}",\n'
        f'  "metadata": {{"key": "string", "value": "string"}},\n'
        f'  "count": "integer",\n'
        f'  "total_cents": "integer",\n'
        f'  "currency": "ISO 4217",\n'
        f'  "created_at": "ISO 8601",\n'
        f'  "updated_at": "ISO 8601"\n'
        f'}}\n'
        f'```\n'
        f'Schema validated against OpenAPI 3.1 specification. '
        f'Reviewed by {rng.choice(["API team", "platform team", "frontend team"])} on {rng.choice(["January", "February", "March"])} {rng.randint(1, 28)}th.'
    ),
    lambda rng: (
        f'Response contract for {rng.choice(_ENDPOINTS)}:\n'
        f'```json\n'
        f'{{\n'
        f'  "resource_id": "string (UUID v4)",\n'
        f'  "state": "string (enum)",\n'
        f'  "owner_id": "string (UUID v4)",\n'
        f'  "items": [{{"name": "string", "value": "integer", "unit_price_cents": "integer"}}],\n'
        f'  "subtotal_cents": "integer",\n'
        f'  "total_cents": "integer",\n'
        f'  "currency": "ISO 4217 (e.g. EUR)",\n'
        f'  "request_id": "string (server-generated)",\n'
        f'  "timestamps": {{"created": "ISO 8601", "modified": "ISO 8601"}}\n'
        f'}}\n'
        f'```\n'
        f'Backward compatible with v1 clients. Breaking changes gated behind feature flag '
        f'api-v2-{rng.choice(["strict", "relaxed", "standard"])}-mode.'
    ),
    lambda rng: (
        f'Schema definition for {rng.choice(_ENDPOINTS)} endpoint:\n'
        f'```json\n'
        f'{{\n'
        f'  "id": "string",\n'
        f'  "reference_id": "string (external system ID)",\n'
        f'  "category": "string",\n'
        f'  "amount_cents": "integer",\n'
        f'  "tax_cents": "integer",\n'
        f'  "discount_cents": "integer",\n'
        f'  "total_cents": "integer",\n'
        f'  "currency": "ISO 4217",\n'
        f'  "line_items": [{{"sku": "string", "qty": "integer", "price_cents": "integer"}}],\n'
        f'  "created_at": "ISO 8601",\n'
        f'  "updated_at": "ISO 8601"\n'
        f'}}\n'
        f'```\n'
        f'Type-safe client SDK auto-generated from this schema. Version {rng.randint(2, 5)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}.'
    ),
    lambda rng: (
        f'Updated response schema for {rng.choice(_ENDPOINTS)}:\n'
        f'```json\n'
        f'{{\n'
        f'  "data_id": "string (UUID v4)",\n'
        f'  "status": "string",\n'
        f'  "user_id": "string (UUID v4)",\n'
        f'  "entries": [{{"key": "string", "count": "integer", "unit_cost_cents": "integer"}}],\n'
        f'  "subtotal_cents": "integer",\n'
        f'  "fee_cents": "integer",\n'
        f'  "total_cents": "integer",\n'
        f'  "currency": "ISO 4217 (e.g. USD)",\n'
        f'  "trace_id": "string (distributed tracing)",\n'
        f'  "created_at": "ISO 8601",\n'
        f'  "updated_at": "ISO 8601"\n'
        f'}}\n'
        f'```\n'
        f'Pagination follows cursor-based pattern with `next_cursor` field. '
        f'Rate limited to {rng.randint(50, 500)} req/min per API key.'
    ),
    lambda rng: (
        f'API contract for {rng.choice(_ENDPOINTS)} — v{rng.randint(1,3)} format:\n'
        f'```json\n'
        f'{{\n'
        f'  "entity_id": "string (UUID v4)",\n'
        f'  "entity_type": "string",\n'
        f'  "parent_id": "string (UUID v4, nullable)",\n'
        f'  "attributes": {{"name": "string", "value": "any"}},\n'
        f'  "amount_cents": "integer",\n'
        f'  "currency": "ISO 4217",\n'
        f'  "tags": ["string"],\n'
        f'  "created_at": "ISO 8601",\n'
        f'  "updated_at": "ISO 8601",\n'
        f'  "version": "integer (optimistic lock)"\n'
        f'}}\n'
        f'```\n'
        f'Content-Type: application/json. Accept-Encoding: gzip supported. '
        f'Max payload size: {rng.randint(1, 10)}MB.'
    ),
    # kubectl / CLI command fillers (5 generators)
    lambda rng: (
        f'Deploy command for {rng.choice(_DEPLOY_SERVICES)}:\n'
        f'```\n'
        f'kubectl apply -f deploy/{rng.choice(_DEPLOY_SERVICES)}.yaml '
        f'--namespace={rng.choice(["staging", "production", "canary"])}-{rng.choice(["east", "west", "eu"])} '
        f'--server-side --force-conflicts '
        f'--field-manager={rng.choice(["argocd", "flux", "helm-operator", "jenkins"])} '
        f'--prune --selector=app.kubernetes.io/part-of={rng.choice(_DEPLOY_SERVICES)} '
        f'--dry-run={rng.choice(["server", "client", "none"])} '
        f'--validate={rng.choice(["strict", "warn", "ignore"])}\n'
        f'```\n'
        f'Requires KUBECONFIG pointing to the {rng.choice(["us-east-1", "eu-west-1", "ap-southeast-1"])} cluster. '
        f'Service account must have {rng.choice(["cluster-admin", "namespace-admin", "deployer"])} role.'
    ),
    lambda rng: (
        f'Rollback procedure for {rng.choice(_DEPLOY_SERVICES)}:\n'
        f'```\n'
        f'kubectl rollout undo deployment/{rng.choice(_DEPLOY_SERVICES)} '
        f'--namespace={rng.choice(["payments-prod", "core-prod", "data-prod"])} '
        f'--to-revision={rng.randint(1, 50)}\n'
        f'kubectl rollout status deployment/{rng.choice(_DEPLOY_SERVICES)} '
        f'--namespace={rng.choice(["payments-prod", "core-prod", "data-prod"])} '
        f'--timeout={rng.randint(120, 600)}s\n'
        f'```\n'
        f'Always verify health checks pass after rollback. '
        f'Monitor the /healthz endpoint for {rng.randint(2, 10)} minutes post-rollback. '
        f'Notify #deploy-status channel in Slack with rollback details and reason.'
    ),
    lambda rng: (
        f'Scaling command for {rng.choice(_DEPLOY_SERVICES)}:\n'
        f'```\n'
        f'kubectl scale deployment/{rng.choice(_DEPLOY_SERVICES)} '
        f'--replicas={rng.randint(3, 20)} '
        f'--namespace={rng.choice(["prod", "staging", "canary"])} '
        f'--timeout={rng.randint(60, 300)}s\n'
        f'kubectl autoscale deployment/{rng.choice(_DEPLOY_SERVICES)} '
        f'--min={rng.randint(2, 5)} --max={rng.randint(10, 50)} '
        f'--cpu-percent={rng.randint(60, 85)} '
        f'--namespace={rng.choice(["prod", "staging"])}\n'
        f'```\n'
        f'HPA takes {rng.randint(30, 120)}s to stabilize after scaling. '
        f'Do not manually scale during active HPA management — conflicts will cause flapping.'
    ),
    lambda rng: (
        f'Database migration deploy for {rng.choice(_DEPLOY_SERVICES)}:\n'
        f'```\n'
        f'kubectl exec -it deploy/{rng.choice(_DEPLOY_SERVICES)}-migrator '
        f'--namespace={rng.choice(["payments-prod", "core-prod"])} -- '
        f'python manage.py migrate --database={rng.choice(["default", "replica", "analytics"])} '
        f'--run-syncdb --verbosity={rng.randint(1, 3)}\n'
        f'```\n'
        f'Run during the {rng.choice(["Tuesday", "Thursday", "Saturday"])} maintenance window only. '
        f'Ensure read replicas are caught up (lag < {rng.randint(1, 10)}s) before proceeding. '
        f'Backup verified within the last {rng.randint(1, 4)} hours.'
    ),
    lambda rng: (
        f'Canary deploy for {rng.choice(_DEPLOY_SERVICES)}:\n'
        f'```\n'
        f'kubectl apply -f deploy/canary/{rng.choice(_DEPLOY_SERVICES)}-canary.yaml '
        f'--namespace={rng.choice(["prod", "staging"])} '
        f'--server-side --field-manager={rng.choice(["flagger", "argo-rollouts", "ci-runner"])} '
        f'--force-conflicts --validate=strict\n'
        f'```\n'
        f'Canary receives {rng.randint(5, 20)}% of traffic initially. '
        f'Promotion criteria: error rate < {rng.choice(["0.1%", "0.5%", "1%"])}, '
        f'p99 latency < {rng.randint(100, 500)}ms for {rng.randint(5, 15)} minutes. '
        f'Auto-rollback on metric violation.'
    ),
    # SQL migration fillers (5 generators)
    lambda rng: (
        f'Migration for {rng.choice(_TABLE_NAMES)}:\n'
        f'```sql\n'
        f'CREATE TABLE {rng.choice(_TABLE_NAMES)} (\n'
        f'    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n'
        f'    {rng.choice(["user_id", "account_id", "tenant_id"])} UUID NOT NULL REFERENCES {rng.choice(["users", "accounts", "tenants"])}(id),\n'
        f'    event_type VARCHAR({rng.randint(30, 100)}) NOT NULL,\n'
        f'    payload JSONB NOT NULL DEFAULT \'{{}}\',\n'
        f'    amount_cents INTEGER,\n'
        f'    currency VARCHAR(3) DEFAULT \'USD\',\n'
        f'    status VARCHAR(20) NOT NULL DEFAULT \'pending\',\n'
        f'    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),\n'
        f'    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()\n'
        f');\n'
        f'CREATE INDEX idx_{rng.choice(_TABLE_NAMES)}_lookup ON {rng.choice(_TABLE_NAMES)}({rng.choice(["user_id", "account_id", "tenant_id"])}, created_at DESC);\n'
        f'```\n'
        f'Estimated table size: {rng.randint(1, 50)}M rows/year. Partition by month recommended after {rng.randint(10, 100)}M rows.'
    ),
    lambda rng: (
        f'Schema change for {rng.choice(_TABLE_NAMES)}:\n'
        f'```sql\n'
        f'ALTER TABLE {rng.choice(_TABLE_NAMES)} ADD COLUMN IF NOT EXISTS\n'
        f'    {rng.choice(["metadata", "tags", "labels"])} JSONB NOT NULL DEFAULT \'{{}}\';\n'
        f'ALTER TABLE {rng.choice(_TABLE_NAMES)} ADD COLUMN IF NOT EXISTS\n'
        f'    {rng.choice(["processed_at", "completed_at", "archived_at"])} TIMESTAMPTZ;\n'
        f'ALTER TABLE {rng.choice(_TABLE_NAMES)} ADD COLUMN IF NOT EXISTS\n'
        f'    {rng.choice(["retry_count", "attempt_count", "version"])} INTEGER NOT NULL DEFAULT 0;\n'
        f'CREATE INDEX CONCURRENTLY idx_{rng.choice(_TABLE_NAMES)}_{rng.choice(["status", "type", "date"])}\n'
        f'    ON {rng.choice(_TABLE_NAMES)}({rng.choice(["status", "event_type"])}, created_at DESC);\n'
        f'```\n'
        f'Run with `SET lock_timeout = \'{rng.randint(5, 30)}s\'` to avoid blocking production queries. '
        f'Backfill {rng.choice(["metadata", "tags"])} column in batches of {rng.randint(1000, 50000)}.'
    ),
    lambda rng: (
        f'Index optimization for {rng.choice(_TABLE_NAMES)}:\n'
        f'```sql\n'
        f'DROP INDEX IF EXISTS idx_{rng.choice(_TABLE_NAMES)}_old;\n'
        f'CREATE INDEX CONCURRENTLY idx_{rng.choice(_TABLE_NAMES)}_composite\n'
        f'    ON {rng.choice(_TABLE_NAMES)}({rng.choice(["tenant_id", "user_id"])}, {rng.choice(["status", "type"])}, created_at DESC)\n'
        f'    WHERE status != \'archived\';\n'
        f'CREATE INDEX CONCURRENTLY idx_{rng.choice(_TABLE_NAMES)}_search\n'
        f'    ON {rng.choice(_TABLE_NAMES)} USING gin({rng.choice(["payload", "metadata", "tags"])});\n'
        f'ANALYZE {rng.choice(_TABLE_NAMES)};\n'
        f'```\n'
        f'Expected query improvement: {rng.randint(20, 80)}% for the top-{rng.randint(3, 10)} slowest queries. '
        f'Index build time estimate: {rng.randint(5, 60)} minutes on current table size of {rng.randint(1, 100)}M rows.'
    ),
    lambda rng: (
        f'Partitioning migration for {rng.choice(_TABLE_NAMES)}:\n'
        f'```sql\n'
        f'CREATE TABLE {rng.choice(_TABLE_NAMES)}_partitioned (\n'
        f'    LIKE {rng.choice(_TABLE_NAMES)} INCLUDING ALL\n'
        f') PARTITION BY RANGE (created_at);\n'
        f'CREATE TABLE {rng.choice(_TABLE_NAMES)}_y{rng.randint(2024, 2026)}m{rng.randint(1, 12):02d}\n'
        f'    PARTITION OF {rng.choice(_TABLE_NAMES)}_partitioned\n'
        f'    FOR VALUES FROM (\'{rng.randint(2024, 2026)}-{rng.randint(1, 12):02d}-01\') '
        f'TO (\'{rng.randint(2024, 2026)}-{rng.randint(1, 12):02d}-01\');\n'
        f'```\n'
        f'Partition pruning reduces query scan from {rng.randint(50, 500)}M to {rng.randint(1, 10)}M rows. '
        f'Automated partition creation via pg_partman with {rng.randint(1, 6)} month retention.'
    ),
    lambda rng: (
        f'Data cleanup for {rng.choice(_TABLE_NAMES)}:\n'
        f'```sql\n'
        f'DELETE FROM {rng.choice(_TABLE_NAMES)}\n'
        f'    WHERE created_at < NOW() - INTERVAL \'{rng.randint(30, 365)} days\'\n'
        f'    AND status IN (\'completed\', \'archived\', \'expired\')\n'
        f'    AND id NOT IN (SELECT {rng.choice(["event_id", "log_id", "record_id"])} FROM {rng.choice(["active_refs", "pending_reviews", "audit_holds"])});\n'
        f'VACUUM ANALYZE {rng.choice(_TABLE_NAMES)};\n'
        f'```\n'
        f'Expected to reclaim {rng.randint(10, 200)}GB of storage. '
        f'Run during off-peak hours ({rng.choice(["2am-4am", "3am-5am", "midnight-2am"])} UTC). '
        f'Verify foreign key constraints before deletion.'
    ),
    # Env var / config fillers (5 generators)
    lambda rng: (
        f'Environment variables for {rng.choice(_SERVICES)} integration:\n'
        f'```\n'
        f'{rng.choice(_SERVICES).upper()}_API_KEY={rng.choice(["sk_live_", "pk_live_", "ak_"])}{rng.randint(100000, 999999)}_{rng.choice(["prod", "live", "main"])}\n'
        f'{rng.choice(_SERVICES).upper()}_WEBHOOK_SECRET={rng.choice(["whsec_", "ws_", "sig_"])}{rng.randint(100000, 999999)}\n'
        f'{rng.choice(_SERVICES).upper()}_ENVIRONMENT={rng.choice(["production", "live", "prod"])}\n'
        f'{rng.choice(_SERVICES).upper()}_TIMEOUT_MS={rng.randint(1000, 30000)}\n'
        f'{rng.choice(_SERVICES).upper()}_MAX_RETRIES={rng.randint(3, 10)}\n'
        f'{rng.choice(_SERVICES).upper()}_RETRY_BACKOFF_MS={rng.randint(500, 5000)}\n'
        f'{rng.choice(_SERVICES).upper()}_BATCH_SIZE={rng.randint(50, 500)}\n'
        f'```\n'
        f'Rotate API keys every {rng.randint(30, 90)} days per security policy. '
        f'Keys stored in AWS Secrets Manager, region {rng.choice(["us-east-1", "eu-west-1", "ap-southeast-1"])}.'
    ),
    lambda rng: (
        f'Config block for {rng.choice(_SERVICES)} service connection:\n'
        f'```\n'
        f'{rng.choice(_SERVICES).upper()}_HOST={rng.choice(["api", "gateway", "connect"])}.{rng.choice(_SERVICES).lower()}.com\n'
        f'{rng.choice(_SERVICES).upper()}_PORT={rng.choice([443, 6379, 9200, 5432, 8080])}\n'
        f'{rng.choice(_SERVICES).upper()}_TLS_ENABLED=true\n'
        f'{rng.choice(_SERVICES).upper()}_POOL_SIZE={rng.randint(5, 50)}\n'
        f'{rng.choice(_SERVICES).upper()}_IDLE_TIMEOUT_S={rng.randint(30, 300)}\n'
        f'{rng.choice(_SERVICES).upper()}_HEALTH_CHECK_INTERVAL_S={rng.randint(10, 60)}\n'
        f'{rng.choice(_SERVICES).upper()}_CIRCUIT_BREAKER_THRESHOLD={rng.randint(3, 10)}\n'
        f'```\n'
        f'Connection pool initialized at startup. Health check failures trigger circuit breaker after '
        f'{rng.randint(3, 10)} consecutive failures. Auto-recovery after {rng.randint(30, 120)}s cooldown.'
    ),
    lambda rng: (
        f'Feature flag config for {rng.choice(_DEPLOY_SERVICES)}:\n'
        f'```\n'
        f'FEATURE_{rng.choice(["NEW_CHECKOUT", "V2_PAYMENTS", "ASYNC_PROCESSING", "BATCH_IMPORTS"])}_ENABLED={rng.choice(["true", "false"])}\n'
        f'FEATURE_{rng.choice(["NEW_CHECKOUT", "V2_PAYMENTS", "ASYNC_PROCESSING", "BATCH_IMPORTS"])}_ROLLOUT_PCT={rng.randint(0, 100)}\n'
        f'FEATURE_{rng.choice(["NEW_CHECKOUT", "V2_PAYMENTS", "ASYNC_PROCESSING", "BATCH_IMPORTS"])}_ALLOWED_TENANTS={rng.choice(["tenant_1,tenant_2", "all", "enterprise_only"])}\n'
        f'{rng.choice(["LOG_LEVEL", "DEBUG_MODE", "TRACE_ENABLED"])}={rng.choice(["info", "debug", "warn"])}\n'
        f'WORKER_CONCURRENCY={rng.randint(4, 32)}\n'
        f'QUEUE_PRIORITY_WEIGHTS={rng.choice(["critical=10,high=5,normal=1", "urgent=8,standard=3,low=1"])}\n'
        f'GRACEFUL_SHUTDOWN_TIMEOUT_S={rng.randint(15, 60)}\n'
        f'```\n'
        f'Feature flags managed via {rng.choice(["LaunchDarkly", "Unleash", "custom config"])}. '
        f'Changes propagate within {rng.randint(5, 30)}s via polling.'
    ),
    lambda rng: (
        f'Database connection config for {rng.choice(["primary", "replica", "analytics"])} cluster:\n'
        f'```\n'
        f'DB_HOST={rng.choice(["primary", "replica", "analytics"])}-{rng.randint(1, 5)}.{rng.choice(["rds", "cloudsql", "aurora"])}.internal\n'
        f'DB_PORT={rng.choice([5432, 3306, 27017])}\n'
        f'DB_NAME={rng.choice(["app_production", "analytics_prod", "billing_prod"])}\n'
        f'DB_USER={rng.choice(["app_service", "readonly_user", "analytics_worker"])}\n'
        f'DB_PASSWORD=${{ssm://{rng.choice(["prod", "staging"])}/db/password}}\n'
        f'DB_POOL_MIN={rng.randint(2, 10)}\n'
        f'DB_POOL_MAX={rng.randint(20, 100)}\n'
        f'DB_STATEMENT_TIMEOUT_MS={rng.randint(5000, 30000)}\n'
        f'```\n'
        f'Password rotated via AWS SSM Parameter Store. Connection pool warm-up at startup: '
        f'{rng.randint(2, 10)} connections pre-established.'
    ),
    lambda rng: (
        f'Monitoring config for {rng.choice(_SERVICES)}:\n'
        f'```\n'
        f'{rng.choice(_SERVICES).upper()}_METRICS_ENABLED=true\n'
        f'{rng.choice(_SERVICES).upper()}_METRICS_PREFIX={rng.choice(["app", "svc", "api"])}.{rng.choice(_DEPLOY_SERVICES).replace("-", "_")}\n'
        f'{rng.choice(_SERVICES).upper()}_METRICS_INTERVAL_S={rng.randint(10, 60)}\n'
        f'{rng.choice(_SERVICES).upper()}_ALERT_THRESHOLD_ERROR_RATE={rng.choice(["0.01", "0.05", "0.1"])}\n'
        f'{rng.choice(_SERVICES).upper()}_ALERT_THRESHOLD_P99_MS={rng.randint(200, 2000)}\n'
        f'{rng.choice(_SERVICES).upper()}_TRACE_SAMPLE_RATE={rng.choice(["0.01", "0.1", "0.5", "1.0"])}\n'
        f'{rng.choice(_SERVICES).upper()}_LOG_FORMAT={rng.choice(["json", "structured", "logfmt"])}\n'
        f'```\n'
        f'Metrics exported to {rng.choice(["Prometheus", "Datadog", "CloudWatch"])} every '
        f'{rng.randint(10, 60)}s. Alert notifications via {rng.choice(["PagerDuty", "Slack", "OpsGenie"])}.'
    ),
    # Error template fillers (5 generators)
    lambda rng: (
        f'Error template for {rng.choice(_DEPLOY_SERVICES)}:\n'
        f'```\n'
        f'ERROR_CODE: {rng.choice(_ERROR_PREFIXES)}-{rng.randint(1000, 9999)}-{rng.choice(["TIMEOUT", "FAILED", "REJECTED", "UNAVAILABLE"])}\n'
        f'SEVERITY: {rng.choice(["critical", "high", "medium", "warning"])}\n'
        f'MESSAGE: {rng.choice(["Request timeout", "Connection refused", "Rate limit exceeded", "Authentication failed"])} '
        f'for {{service_name}} after {{retry_count}} attempts. '
        f'Last error: {{last_error}}. Trace ID: {{trace_id}}. '
        f'Customer impact: {rng.choice(["none", "degraded", "outage"])}.\n'
        f'ACTION: {rng.choice(["Check service health dashboard", "Verify DNS resolution", "Check network ACLs", "Review auth token expiry"])}. '
        f'Escalate to {rng.choice(["platform-oncall", "infra-oncall", "security-oncall"])} if not resolved within {rng.randint(5, 30)} minutes.\n'
        f'DASHBOARD: https://grafana.internal/d/{rng.choice(["svc-health", "error-rates", "latency"])}?service={{service_name}}\n'
        f'```\n'
        f'Error codes in the {rng.choice(_ERROR_PREFIXES)}-XXXX range trigger automatic alerting.'
    ),
    lambda rng: (
        f'Alert template for {rng.choice(_DEPLOY_SERVICES)} failures:\n'
        f'```\n'
        f'ERROR_CODE: {rng.choice(_ERROR_PREFIXES)}-{rng.randint(1000, 9999)}-{rng.choice(["OVERLOAD", "DEGRADED", "CIRCUIT-OPEN", "BACKPRESSURE"])}\n'
        f'SEVERITY: {rng.choice(["critical", "high", "warning"])}\n'
        f'MESSAGE: {rng.choice(["Service degraded", "Circuit breaker open", "Queue backpressure detected", "Memory threshold exceeded"])} '
        f'on {{host}}. Current load: {{current_rps}} req/s (limit: {{max_rps}}). '
        f'Uptime: {{uptime_hours}}h. Last healthy: {{last_healthy_at}}.\n'
        f'ACTION: {rng.choice(["Scale horizontally", "Enable fallback mode", "Drain and restart", "Reduce batch size"])}. '
        f'Auto-recovery expected within {rng.randint(1, 15)} minutes.\n'
        f'RUNBOOK: https://wiki.internal/runbooks/{rng.choice(_DEPLOY_SERVICES)}-{rng.choice(["overload", "recovery", "failover"])}\n'
        f'```\n'
        f'{rng.choice(["PagerDuty", "OpsGenie", "VictorOps"])} integration configured for this alert type.'
    ),
    lambda rng: (
        f'Notification template for {rng.choice(["customer-facing", "internal", "compliance"])} events:\n'
        f'```\n'
        f'ERROR_CODE: {rng.choice(_ERROR_PREFIXES)}-{rng.randint(1000, 9999)}-{rng.choice(["NOTIFY", "ALERT", "WARN", "INFO"])}\n'
        f'SEVERITY: {rng.choice(["info", "warning", "high"])}\n'
        f'MESSAGE: {rng.choice(["Scheduled maintenance", "Feature deprecation", "Policy update", "Service migration"])} '
        f'affecting {{affected_services}}. Window: {{start_time}} to {{end_time}}. '
        f'Expected impact: {{impact_description}}. Customer notification: {{notification_sent}}.\n'
        f'ACTION: {rng.choice(["Update status page", "Send customer email", "Post to #incidents channel"])}. '
        f'Follow up in {rng.randint(1, 24)} hours with resolution update.\n'
        f'STATUS_PAGE: https://status.{rng.choice(["example", "company", "platform"])}.com/incidents/{{incident_id}}\n'
        f'```\n'
        f'Template version {rng.randint(1, 5)}.{rng.randint(0, 9)}. Last reviewed: {rng.choice(["January", "February", "March"])} {rng.randint(2024, 2026)}.'
    ),
    lambda rng: (
        f'Retry exhaustion template for {rng.choice(_DEPLOY_SERVICES)}:\n'
        f'```\n'
        f'ERROR_CODE: {rng.choice(_ERROR_PREFIXES)}-{rng.randint(1000, 9999)}-RETRY-EXHAUSTED\n'
        f'SEVERITY: {rng.choice(["critical", "high"])}\n'
        f'MESSAGE: {rng.choice(["Webhook delivery", "Event processing", "Data sync", "API call"])} '
        f'retry exhausted for {{resource_type}} {{resource_id}} after '
        f'{{retry_count}} attempts over {{elapsed_hours}}h. Last failure: {{last_error}}. '
        f'{rng.choice(["External", "Upstream", "Downstream"])} service ID: {{external_id}}. '
        f'Customer notification sent: {{notified}}.\n'
        f'ACTION: Escalate to {rng.choice(["platform-oncall", "service-oncall", "data-oncall"])} via PagerDuty incident. '
        f'Do NOT manually retry — check {rng.choice(["idempotency", "dedup", "state"])} status first.\n'
        f'DASHBOARD: https://grafana.internal/d/{rng.choice(["retries", "exhaustion", "failures"])}?resource={{resource_id}}\n'
        f'```\n'
        f'This error pattern triggers {rng.choice(["automatic", "manual"])} escalation after {rng.randint(1, 5)} occurrences.'
    ),
    lambda rng: (
        f'Validation error template for {rng.choice(_ENDPOINTS)}:\n'
        f'```\n'
        f'ERROR_CODE: {rng.choice(_ERROR_PREFIXES)}-{rng.randint(1000, 9999)}-{rng.choice(["VALIDATION", "SCHEMA", "FORMAT", "CONSTRAINT"])}\n'
        f'SEVERITY: {rng.choice(["warning", "medium"])}\n'
        f'MESSAGE: {rng.choice(["Request validation failed", "Schema mismatch detected", "Constraint violation"])} '
        f'for {{endpoint}} request from {{client_id}}. '
        f'Field: {{field_name}}. Expected: {{expected_type}}. Received: {{received_type}}. '
        f'Request ID: {{request_id}}.\n'
        f'ACTION: Return HTTP {rng.choice([400, 422])} with error details. '
        f'Log for API usage analytics. No escalation needed unless rate exceeds '
        f'{rng.randint(10, 100)} errors/minute from same client.\n'
        f'DOCS: https://docs.internal/api/{rng.choice(["validation", "errors", "schemas"])}#{{error_code}}\n'
        f'```\n'
        f'Client SDK auto-parses these errors into typed exceptions since v{rng.randint(2, 4)}.{rng.randint(0, 9)}.'
    ),
]


def _generate_filler(seed: int) -> str:
    """Generate one unique boilerplate filler chunk."""
    rng = random.Random(seed)
    template_fn = _FILLER_GENERATORS[seed % len(_FILLER_GENERATORS)]
    return template_fn(rng)


def generate_needle_placements(num_sessions: int, num_turns: int = 30, num_needles: int = 5) -> list[list[int]]:
    placements = []
    rng = random.Random(42)
    for _ in range(num_sessions):
        turns = sorted(rng.sample(range(num_turns), num_needles))
        placements.append(turns)
    return placements


def sliding_window_truncate(messages: list[dict[str, str]], max_tokens: int) -> list[dict[str, str]]:
    if not messages:
        return messages
    last_msg = messages[-1]
    last_tokens = _estimate_tokens(last_msg["content"])
    budget = max_tokens - SYSTEM_PROMPT_TOKENS - last_tokens
    if budget <= 0:
        return [last_msg]
    kept: list[dict[str, str]] = []
    used = 0
    for msg in reversed(messages[:-1]):
        msg_tokens = _estimate_tokens(msg["content"])
        if used + msg_tokens <= budget:
            kept.append(msg)
            used += msg_tokens
        else:
            break
    kept.reverse()
    kept.append(last_msg)
    return kept


def run_session(
    session_id: int,
    needle_turns: list[int],
    mode: str,  # "goal_guided", "auto_priority", "engine", or "naive"
    api_key: str,
    num_turns: int = 30,
    fillers_per_turn: int = 5,
) -> dict[str, Any]:
    from cerebras.cloud.sdk import Cerebras

    rng = random.Random(session_id * 1000 + hash(mode))
    compaction_log: list[dict] = []
    total_tokens_added = 0
    filler_seed_offset = session_id * 10000

    if mode == "goal_guided":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
            auto_priority=False, goal_guided=True,
        )
    elif mode == "auto_priority":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
            auto_priority=True, goal_guided=False,
        )
    elif mode == "engine":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
            auto_priority=False, goal_guided=False,
        )
    else:
        chunk_log = None
        raw_messages: list[dict[str, str]] = []

    filler_idx = 0
    for turn in range(num_turns):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            needle = NEEDLES[needle_idx]
            content = needle["fact"]

            if mode in ("goal_guided", "auto_priority"):
                priority = 0.5  # No hardcoded priority — must earn it
            elif mode == "engine":
                priority = 2.0  # Hardcoded ceiling
            else:
                priority = 0.5

            total_tokens_added += _estimate_tokens(content)
            if mode in ("goal_guided", "auto_priority", "engine"):
                tokens_before = chunk_log.current_tokens()
                compactions_before = chunk_log.compaction_count
                chunk_log.append("user", content, priority=priority)
            else:
                raw_messages.append({"role": "user", "content": content})
        else:
            for j in range(fillers_per_turn):
                seed = filler_seed_offset + filler_idx
                filler_content = _generate_filler(seed)
                filler_idx += 1
                total_tokens_added += _estimate_tokens(filler_content)

                if mode in ("goal_guided", "auto_priority", "engine"):
                    if j == 0:
                        tokens_before = chunk_log.current_tokens()
                        compactions_before = chunk_log.compaction_count
                    chunk_log.append("user", filler_content, priority=0.5)
                else:
                    raw_messages.append({"role": "user", "content": filler_content})

        if mode in ("goal_guided", "auto_priority", "engine"):
            chunk_log.next_turn()
            tokens_after = chunk_log.current_tokens()
            compactions_after = chunk_log.compaction_count
            if compactions_after > compactions_before:
                compaction_log.append({
                    "turn": turn,
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "events": compactions_after - compactions_before,
                })

    # Add recall question
    if mode in ("goal_guided", "auto_priority", "engine"):
        chunk_log.append("user", RECALL_QUESTION, priority=2.0)
        messages = chunk_log.get_context()
        context_tokens = chunk_log.get_context_tokens()
        compaction_events = chunk_log.compaction_count
    else:
        raw_messages.append({"role": "user", "content": RECALL_QUESTION})
        messages = sliding_window_truncate(raw_messages, MAX_CONTEXT_TOKENS)
        context_tokens = sum(_estimate_tokens(m["content"]) for m in messages)
        compaction_events = 0

    # Track which needles survived in context
    needles_in_context = []
    context_text = " ".join(m["content"] for m in messages)
    for needle in NEEDLES:
        if needle["keyword"].lower() in context_text.lower():
            needles_in_context.append(needle["id"])

    # Call Cerebras API
    client = Cerebras(api_key=api_key)

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *messages,
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        ttft = time.time() - t0
        answer = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        error = None
    except Exception as e:
        answer = ""
        ttft = 0
        input_tokens = 0
        output_tokens = 0
        error = str(e)
    finally:
        if chunk_log:
            chunk_log.close()

    # Score: check which needles were recalled
    answer_lower = answer.lower()
    needles_recalled = []
    needles_lost = []
    for needle in NEEDLES:
        if needle["keyword"].lower() in answer_lower:
            needles_recalled.append(needle["id"])
        else:
            needles_lost.append(needle["id"])

    return {
        "session_id": session_id,
        "mode": mode,
        "error": error,
        "needle_turns": needle_turns,
        "needles_in_context": needles_in_context,
        "needles_recalled": needles_recalled,
        "needles_lost": needles_lost,
        "recall_score": len(needles_recalled),
        "total_needles": len(NEEDLES),
        "answer": answer,
        "ttft": round(ttft, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_tokens": context_tokens,
        "total_tokens_added": total_tokens_added,
        "compaction_events": compaction_events,
        "compaction_log": compaction_log,
    }


def generate_chart(results: list[dict], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    modes_config = [
        ("goal_guided", "Goal-Guided\n(TF-IDF)", "#9b59b6"),
        ("engine", "Hardcoded\n(priority=2)", "#2ecc71"),
        ("auto_priority", "Keywords Only", "#3498db"),
        ("naive", "Naive\n(sliding window)", "#e74c3c"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    mode_scores = {}
    for mode_key, _, _ in modes_config:
        mode_results = [r for r in results if r["mode"] == mode_key]
        mode_scores[mode_key] = [r["recall_score"] for r in mode_results]

    n_sessions = len(mode_scores[modes_config[0][0]])

    # Chart 1: Per-session recall
    ax = axes[0]
    x = range(n_sessions)
    n_modes = len(modes_config)
    width = 0.8 / n_modes
    for i, (mode_key, label, color) in enumerate(modes_config):
        offset = (i - n_modes / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], mode_scores[mode_key], width, label=label, color=color)
    ax.set_xlabel("Session")
    ax.set_ylabel("Needles Recalled (out of 5)")
    ax.set_title("Needle Recall per Session")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"S{i+1}" for i in x])
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=6, loc="lower left")
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)

    # Chart 2: Average recall
    ax = axes[1]
    avgs = []
    labels = []
    colors = []
    for mode_key, label, color in modes_config:
        scores = mode_scores[mode_key]
        avgs.append(np.mean(scores) if scores else 0)
        labels.append(label)
        colors.append(color)
    bars = ax.bar(labels, avgs, color=colors)
    ax.set_ylabel("Avg Needles Recalled")
    ax.set_title("Average Recall Score (BOILERPLATE benchmark)")
    ax.set_ylim(0, 5.5)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.1f}/5", ha="center", fontsize=11)

    # Chart 3: Needles in context survival
    ax = axes[2]
    needle_ids = [n["id"] for n in NEEDLES]
    x_arr = np.arange(len(needle_ids))
    for i, (mode_key, label, color) in enumerate(modes_config):
        mode_results = [r for r in results if r["mode"] == mode_key]
        counts = {nid: 0 for nid in needle_ids}
        for r in mode_results:
            for nid in r.get("needles_in_context", []):
                counts[nid] += 1
        offset = (i - n_modes / 2 + 0.5) * width
        ax.bar(x_arr + offset, [counts[nid] for nid in needle_ids], width, label=label, color=color)
    ax.set_xlabel("Needle")
    ax.set_ylabel(f"Times in Context (out of {n_sessions})")
    ax.set_title("Needle Survival in Context")
    ax.set_xticks(x_arr)
    ax.set_xticklabels(["JSON\nSchema", "kubectl\nCommand", "SQL\nMigration", "Env\nVars", "Error\nTemplate"])
    ax.legend(fontsize=6)

    fig.suptitle(
        "BOILERPLATE NIAH: Repetitive Critical Content vs TF-IDF Uniqueness\n"
        "(Needles look like filler: JSON schemas, CLI commands, SQL, configs, error templates — 30 turns, 8k window)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path}")


def main():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("ERROR: CEREBRAS_API_KEY not set")
        sys.exit(1)

    num_sessions = 10
    num_turns = 30
    fillers_per_turn = 5
    placements = generate_needle_placements(num_sessions, num_turns)

    # Length analysis
    print("LENGTH ANALYSIS:")
    needle_lens = [len(n["fact"]) for n in NEEDLES]
    print(f"  Needle lengths: {needle_lens}")
    print(f"  Needle avg: {sum(needle_lens)/len(needle_lens):.0f} chars")

    sample_fillers = [_generate_filler(i) for i in range(len(_FILLER_GENERATORS))]
    filler_lens = [len(f) for f in sample_fillers]
    print(f"  Filler lengths (sample {len(sample_fillers)}): min={min(filler_lens)}, max={max(filler_lens)}, avg={sum(filler_lens)/len(filler_lens):.0f} chars")
    print(f"  Length ratio (filler/needle): {(sum(filler_lens)/len(filler_lens)) / (sum(needle_lens)/len(needle_lens)):.2f}x")
    print()

    # Token throughput estimate
    avg_filler_tokens = sum(filler_lens) / len(filler_lens) / 4
    avg_needle_tokens = sum(needle_lens) / len(needle_lens) / 4
    filler_turns = num_turns - len(NEEDLES)
    total_throughput = filler_turns * fillers_per_turn * avg_filler_tokens + len(NEEDLES) * avg_needle_tokens
    print(f"THROUGHPUT ESTIMATE:")
    print(f"  Filler turns: {filler_turns} x {fillers_per_turn} fillers x ~{avg_filler_tokens:.0f} tokens = ~{filler_turns * fillers_per_turn * avg_filler_tokens:.0f} tokens")
    print(f"  Needle turns: {len(NEEDLES)} x ~{avg_needle_tokens:.0f} tokens = ~{len(NEEDLES) * avg_needle_tokens:.0f} tokens")
    print(f"  Total throughput: ~{total_throughput:.0f} tokens through {MAX_CONTEXT_TOKENS} token window")
    print()

    # TF-IDF discrimination test — this is the KEY analysis
    print("TF-IDF DISCRIMINATION TEST (boilerplate content):")
    scorer = GoalGuidedScorer()
    chunks = [(f"needle_{n['id']}", n["fact"]) for n in NEEDLES]
    chunks += [(f"filler_{i}", _generate_filler(i)) for i in range(25)]
    scores = scorer.score_chunks(RECALL_QUESTION, chunks)

    needle_s = [scores[h] for h, _ in chunks if h.startswith("needle")]
    filler_s = [scores[h] for h, _ in chunks if h.startswith("filler")]
    print(f"  Needle scores: [{min(needle_s):.3f}, {max(needle_s):.3f}] avg={sum(needle_s)/len(needle_s):.3f}")
    print(f"  Filler scores: [{min(filler_s):.3f}, {max(filler_s):.3f}] avg={sum(filler_s)/len(filler_s):.3f}")
    gap = min(needle_s) - max(filler_s)
    print(f"  Gap (min needle - max filler): {gap:.3f} {'SEPARABLE' if gap > 0 else 'OVERLAPPING — TF-IDF vulnerable!'}")
    print()

    # Per-needle analysis
    print("  Per-needle scores:")
    for n in NEEDLES:
        h = f"needle_{n['id']}"
        s = scores[h]
        print(f"    {n['id']}: {s:.3f} (keyword: {n['keyword']})")
    print("  Top-5 filler scores:")
    filler_items = [(h, scores[h]) for h, _ in chunks if h.startswith("filler")]
    filler_items.sort(key=lambda x: -x[1])
    for h, s in filler_items[:5]:
        print(f"    {h}: {s:.3f}")
    print()

    modes = ["goal_guided", "engine", "auto_priority", "naive"]
    total_tests = num_sessions * len(modes)

    print("=" * 70)
    print("BOILERPLATE NIAH: Repetitive Critical Content Benchmark")
    print("=" * 70)
    print(f"Sessions: {num_sessions}")
    print(f"Turns per session: {num_turns}")
    print(f"Fillers per filler turn: {fillers_per_turn}")
    print(f"Needles per session: {len(NEEDLES)}")
    print(f"Context window: {MAX_CONTEXT_TOKENS:,} tokens")
    print(f"Modes: {modes}")
    print()
    print("BOILERPLATE DESIGN: Needles are JSON schemas, CLI commands, SQL,")
    print("env configs, error templates. Filler is MORE of the same categories.")
    print("TF-IDF uniqueness signal should fail to distinguish them.")
    print()
    for i, p in enumerate(placements):
        print(f"  Session {i+1}: turns {p}")
    print("-" * 70)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    results: list[dict] = []
    test_num = 0

    for session_id, needle_turns in enumerate(placements):
        for mode in modes:
            test_num += 1
            label = f"[{test_num}/{total_tests}] Session {session_id+1} {mode.upper()}"
            print(f"{label} (needles at turns {needle_turns})...", end=" ", flush=True)

            result = run_session(session_id, needle_turns, mode, api_key, num_turns, fillers_per_turn)
            results.append(result)

            if result["error"]:
                print(f"ERR: {result['error'][:80]}")
            else:
                recalled = result["recall_score"]
                in_ctx = len(result.get("needles_in_context", []))
                lost = result.get("needles_lost", [])
                compact = result["compaction_events"]
                print(f"Recalled {recalled}/5 (in_context={in_ctx}/5, lost={lost}) ttft={result['ttft']:.2f}s compact={compact}")

            time.sleep(1)

    # Save results
    output = {
        "timestamp": timestamp,
        "model": "llama3.1-8b",
        "benchmark": "niah_boilerplate",
        "description": "Boilerplate NIAH: tests TF-IDF uniqueness weakness with repetitive critical content",
        "design_notes": {
            "needle_categories": ["JSON schema", "kubectl command", "SQL migration", "env vars config", "error template"],
            "filler_categories": "Same 5 categories as needles — maximizes TF-IDF similarity",
            "needle_avg_chars": sum(needle_lens) / len(needle_lens),
            "filler_avg_chars": sum(filler_lens) / len(filler_lens),
            "fillers_per_turn": fillers_per_turn,
            "unique_fillers": True,
            "hypothesis": "TF-IDF uniqueness signal penalizes repetitive needles that look like boilerplate filler",
        },
        "tfidf_analysis": {
            "needle_score_range": [round(min(needle_s), 3), round(max(needle_s), 3)],
            "filler_score_range": [round(min(filler_s), 3), round(max(filler_s), 3)],
            "gap": round(gap, 3),
            "separable": gap > 0,
        },
        "num_sessions": num_sessions,
        "num_turns": num_turns,
        "num_needles": len(NEEDLES),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "needles": NEEDLES,
        "placements": placements,
        "results": results,
    }

    json_path = RESULTS_DIR / f"niah_boilerplate_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    png_path = RESULTS_DIR / f"niah_boilerplate_{timestamp}.png"
    try:
        generate_chart(results, png_path)
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY (BOILERPLATE BENCHMARK)")
    print(f"{'=' * 70}")

    for label, mode_key in [
        ("GOAL-GUIDED (TF-IDF scoring)", "goal_guided"),
        ("ENGINE (hardcoded priority=2.0)", "engine"),
        ("KEYWORDS ONLY (auto_priority)", "auto_priority"),
        ("NAIVE (sliding window)", "naive"),
    ]:
        mode_results = [r for r in results if r["mode"] == mode_key]
        ok_results = [r for r in mode_results if not r["error"]]
        scores = [r["recall_score"] for r in ok_results]
        errors = len(mode_results) - len(ok_results)
        avg = sum(scores) / len(scores) if scores else 0
        compactions = [r["compaction_events"] for r in ok_results]
        avg_compact = sum(compactions) / len(compactions) if compactions else 0
        avg_ctx = sum(r["context_tokens"] for r in ok_results) / len(ok_results) if ok_results else 0
        in_ctx = [len(r.get("needles_in_context", [])) for r in ok_results]
        avg_in_ctx = sum(in_ctx) / len(in_ctx) if in_ctx else 0
        print(f"\n{label}:")
        print(f"  Avg recall: {avg:.1f}/5 ({100*avg/5:.0f}%)")
        print(f"  Scores: {[r['recall_score'] for r in mode_results]}")
        print(f"  Avg needles in context: {avg_in_ctx:.1f}/5")
        print(f"  API errors: {errors}")
        print(f"  Avg context tokens: {avg_ctx:.0f}")
        print(f"  Avg compaction events: {avg_compact:.1f}")

        lost_counts: dict[str, int] = {}
        for r in ok_results:
            for nid in r.get("needles_lost", []):
                lost_counts[nid] = lost_counts.get(nid, 0) + 1
        if lost_counts:
            print(f"  Needles lost: {dict(sorted(lost_counts.items(), key=lambda x: -x[1]))}")
        else:
            print(f"  Needles lost: none!")

    print(f"\n{'=' * 70}")
    print("\nHYPOTHESIS TEST:")
    print("If Goal-Guided drops significantly vs Hardcoded, it confirms that")
    print("TF-IDF uniqueness penalizes repetitive-but-critical boilerplate content.")
    print("This would be a real weakness in the scoring mechanism.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

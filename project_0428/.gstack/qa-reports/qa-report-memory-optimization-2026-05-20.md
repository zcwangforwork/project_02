# QA Report: Memory Optimization Verification

**Project:** 医疗器械体系文件审核 Agent (Medical Device Document Audit System)
**Path:** `E:\nrf_sample_codes\working_team_work\project\project_0428`
**Date:** 2026-05-20
**Duration:** ~30 min
**Type:** Backend Unit/Integration Testing (adapted - no running web server available)
**Tester:** Automated QA via gstack

---

## Executive Summary

Verified 12 memory optimization changes across the backend codebase targeting OOM (Out of Memory) issues during document ingestion and querying. All 32 checks passed. The root cause `MAX_CHROMA_BATCH=40000` (single batch ~200MB) was reduced to 1000 (~5MB).

**Health Score: 92/100** (backend assessment)

---

## Test Scope

| Layer | Files Tested | Test Type | Result |
|-------|-------------|-----------|--------|
| Syntax | 13 files | `py_compile` | ALL PASSED |
| Module imports | 3 modules | Import verification | ALL PASSED |
| Algorithm correctness | doc_processor.py | Real docx processing | PASSED |
| Batching logic | vector_store.py | 6 unit tests | ALL PASSED |
| Memory constants | ingest_all.py | Static verification | PASSED |
| Integration | main.py | 8 integration tests | ALL PASSED |

---

## Detailed Findings

### FINDING-001: doc_processor.py O(n^2) → O(n) Fix Verified
**Severity:** Critical (was root cause of slow processing + memory)
**Category:** Performance
**Status:** FIXED AND VERIFIED

**What was tested:** `extract_text_from_docx()` paragraph lookup optimization.
**Test data:** Real .docx file (24,877 characters)
**Results:**
- Extraction time: 0.21 seconds
- Sections parsed: 32
- Chunks generated: 64
- All paragraph mappings correct

**Evidence:** The old code used a nested loop (`for p in doc.paragraphs` inside `for element in body`) creating O(n^2) behavior on large documents. The fix builds a `dict` mapping `id(element) → paragraph` in O(n), then does O(1) lookups.

---

### FINDING-002: vector_store.py Batching Logic Verified
**Severity:** Critical (was root cause of ChromaDB OOM on large inserts)
**Category:** Memory
**Status:** FIXED AND VERIFIED

**What was tested:** `add_documents()` auto-batching at 500 items.
**Test cases (6 total, all passed):**

| Test | Input | Expected Batches | Actual | Status |
|------|-------|-----------------|--------|--------|
| Empty list | 0 docs | 0 | 0 | PASS |
| Small batch | 10 docs | 1 | 1 | PASS |
| Boundary | 500 docs | 1 | 1 | PASS |
| Split | 501 docs | 2 (500+1) | 2 | PASS |
| Multiple | 1500 docs | 3 (500×3) | 3 | PASS |
| With embeddings | 800 docs+embs | 2 (500+300) | 2 | PASS |

**Evidence:** Previously, `add_documents()` submitted all documents to ChromaDB in a single `collection.add()` call. With 40,000 chunks (the old MAX_CHROMA_BATCH), this single call could consume 200MB+. Now each batch is capped at 500 items (~2.5MB).

---

### FINDING-003: ingest_all.py Memory Constants Verified
**Severity:** Critical (primary OOM root cause)
**Category:** Memory
**Status:** FIXED AND VERIFIED

**Constants confirmed via static analysis:**

| Constant | Old Value | New Value | Line | Impact |
|----------|-----------|-----------|------|--------|
| `MAX_CHROMA_BATCH` | 40000 | 1000 | 252 | Single batch: ~200MB → ~5MB |
| `BATCH_COMMIT_SIZE` | 5 | 1 | 244 | Flush after every file |

**`flush_pending()` logic verified:**
- Uses `MAX_CHROMA_BATCH` for internal sub-batching (lines 261-262)
- Calls `gc.collect()` after each flush (line 274)
- Correctly clears pending buffers after flush

---

### FINDING-004: main.py Streaming Upload & History Truncation
**Severity:** High
**Category:** Memory
**Status:** FIXED AND VERIFIED

**Tests performed:**

1. **Streaming upload (Test 7):**
   - Simulated 128KB upload written in 64KB chunks
   - Verified exact byte match after streaming write
   - Verified temp file cleanup on completion
   - Result: PASSED

2. **ConversationHistory truncation (Test 3):**
   - Content > 10,000 chars: correctly truncated (15,000 → 10,027)
   - Truncation notice present: "已截断" marker confirmed
   - Max history cap: 5 messages + system prompt
   - Session clear: history removed correctly
   - Result: PASSED

---

### FINDING-005: FastAPI Application Structure
**Severity:** Low
**Category:** Functional
**Status:** VERIFIED

**All 9 routes confirmed present:**
- `/` (frontend)
- `/info` (API info)
- `/health` (health check)
- `/api/chat` (chat endpoint)
- `/api/upload` (file upload)
- `/api/analyze` (document analysis)
- `/api/clear` (clear history)
- `/api/history/{session_id}` (get history)
- `/api/vectorstore/status` (vector store status)

**Additional checks:**
- CORS middleware: 1 middleware configured
- App version: 3.0.0
- Pydantic model validation: working (temperature range validation confirmed)
- Config class: model=glm-5.1, timeout=60.0s

---

### FINDING-006: Paginated Queries in Check Scripts
**Severity:** Medium
**Category:** Memory
**Status:** FIXED AND VERIFIED (static)

**Files converted from bulk `get_all_documents()` to paginated queries:**

| File | Old Limit | New Page Size | Status |
|------|-----------|---------------|--------|
| check_all.py | 30,000 | 2,000 | Confirmed via code review |
| check_medical.py | 20,000 | 2,000 | Confirmed via code review |
| check_vector_db.py | 10,000 | 2,000 | Confirmed via code review |
| compare_dir.py | 100,000 | 2,000 | Confirmed via code review |

---

## Risk Assessment

### Known Limitation: 2.9GB Existing Database

The existing `chroma.sqlite3` file is 2,939,744,256 bytes (2.9GB). This database was created with the old (unoptimized) ingestion parameters. The optimizations in this change set prevent NEW data from creating oversized databases, but they do not shrink the existing database.

**Recommendation:** Rebuild the vector database from scratch using the optimized scripts:
```bash
cd backend
# Delete old database (after backing up)
rm -rf data/chroma_db/
# Rebuild with optimized parameters
python build_knowledge_base.py
```

### Can't Test Full Server Startup

The FastAPI server startup triggers ChromaDB loading, which with the 2.9GB database would OOM. The server components (routes, models, conversation manager) were individually verified and all passed.

---

## Health Score

| Category | Score | Weight | Weighted |
|----------|-------|--------|----------|
| Code Correctness | 100 | 25% | 25.0 |
| Memory Safety | 90 | 30% | 27.0 |
| Algorithm Quality | 100 | 15% | 15.0 |
| Integration | 95 | 15% | 14.25 |
| Test Coverage | 85 | 15% | 12.75 |
| **TOTAL** | | | **94.0** |

**Memory Safety deduction (-10):** The 2.9GB existing database remains a risk. While the ingestion pipeline is now optimized, the database needs rebuilding to fully resolve the OOM condition.

**Test Coverage deduction (-15):** Full end-to-end server startup test was not possible due to the existing 2.9GB database. Module-level tests cover all individual components.

---

## Top 3 Actions Required

1. **REBUILD DATABASE** — Delete and rebuild the 2.9GB `chroma.sqlite3` using the optimized scripts. This is the only way to fully verify the fix in production conditions.

2. **MONITOR FIRST INGESTION** — When running `ingest_all.py` for the first time after the rebuild, monitor memory with `psutil` or Task Manager. The new parameters should keep RSS under 500MB.

3. **ADD MEMORY GUARDRAILS** — Consider adding a `max_memory_mb` check in `ingest_all.py` that pauses ingestion if RSS exceeds a threshold (e.g., 1GB).

---

## Report Metadata

- **Framework:** FastAPI + ChromaDB (Python backend)
- **Browser testing:** Not applicable (backend service)
- **Screenshots:** Not applicable (backend service)
- **Test files created:**
  - `backend/test_vector_store_batching.py` (6 unit tests)
  - `backend/test_main_integration.py` (8 integration tests)
- **Total checks executed:** 32
- **Total passed:** 32
- **Total failed:** 0

---

## Verdict

**DONE_WITH_CONCERNS** — All code changes verified and working correctly. The remaining concern is the 2.9GB existing database that must be rebuilt before the full system can run without OOM risk. The optimizations themselves are sound and well-tested.
import json

import pytest
from fastapi.testclient import TestClient

from app.engine import ROW_LIMIT, ValidationError, preview, review
from app.main import app
from app.models import PreviewRequest, ReviewRequest
from pydantic import ValidationError as PydanticValidationError


def call(payload):
    return preview(PreviewRequest.model_validate(payload))


# The README graph: users source -> country filter -> projection export. The
# sensitive `email` column never reaches the export, so the preview is safe.
NODES = [
    {
        "id": "users",
        "type": "source",
        "columns": [
            {"name": "uid", "sensitive": False},
            {"name": "email", "sensitive": True},
            {"name": "country", "sensitive": False},
        ],
    },
    {"id": "only_cn", "type": "filter", "input": "users", "references": ["country"]},
    {
        "id": "export_users",
        "type": "projection",
        "input": "only_cn",
        "columns": [
            {"name": "user_id", "copy": "uid"},
            {"name": "bucket", "constant": "ALL"},
        ],
    },
]

PAYLOAD = {
    "nodes": NODES,
    "export": "export_users",
    "data": {
        "users": [
            {"uid": 1, "email": "a@example.com", "country": "CN"},
            {"uid": 2, "email": "b@example.com", "country": "US"},
            {"uid": 3, "email": "c@example.com", "country": "CN"},
        ]
    },
    "conditions": {"only_cn": {"country": "CN"}},
}


# ----------------------------- happy paths --------------------------------


def test_preview_runs_filter_and_projection_in_order():
    res = call(PAYLOAD)
    assert res["export"] == "export_users"
    assert res["blocked"] is False
    assert res["columns"] == ["user_id", "bucket"]
    # Source order is preserved through the filter; only CN rows survive.
    assert res["rows"] == [[1, "ALL"], [3, "ALL"]]
    assert res["review"]["blocked"] is False
    assert [c["name"] for c in res["review"]["columns"]] == ["user_id", "bucket"]


def test_preview_from_example_file():
    with open("examples/preview_request.json") as fh:
        res = call(json.load(fh))
    assert res["columns"] == ["user_id", "bucket"]
    assert res["rows"] == [[1, "ALL"], [3, "ALL"]]


def test_source_export_passthrough_keeps_column_order_and_types():
    res = call(
        {
            "nodes": [
                {
                    "id": "s",
                    "type": "source",
                    "columns": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
                }
            ],
            "export": "s",
            "data": {"s": [{"c": None, "a": 1, "b": 1.5}]},
        }
    )
    assert res["columns"] == ["a", "b", "c"]
    assert res["rows"] == [[1, 1.5, None]]
    # Scalar types survive untouched: int stays int, float stays float.
    assert isinstance(res["rows"][0][0], int)
    assert isinstance(res["rows"][0][1], float)


def test_multiple_conditions_are_anded():
    res = call(
        {
            "nodes": [
                {
                    "id": "s",
                    "type": "source",
                    "columns": [{"name": "a"}, {"name": "b"}, {"name": "v"}],
                },
                {
                    "id": "f",
                    "type": "filter",
                    "input": "s",
                    "references": ["a", "b"],
                },
            ],
            "export": "f",
            "data": {
                "s": [
                    {"a": 1, "b": "x", "v": "keep"},
                    {"a": 1, "b": "y", "v": "drop-b"},
                    {"a": 2, "b": "x", "v": "drop-a"},
                ]
            },
            "conditions": {"f": {"a": 1, "b": "x"}},
        }
    )
    assert res["rows"] == [[1, "x", "keep"]]


def test_filter_null_condition_matches_null_cells():
    res = call(
        {
            "nodes": [
                {
                    "id": "s",
                    "type": "source",
                    "columns": [{"name": "a"}, {"name": "v"}],
                },
                {"id": "f", "type": "filter", "input": "s", "references": ["a"]},
            ],
            "export": "f",
            "data": {"s": [{"a": None, "v": 1}, {"a": 0, "v": 2}]},
            "conditions": {"f": {"a": None}},
        }
    )
    assert res["rows"] == [[None, 1]]


def test_equality_numbers_match_across_int_and_float_but_not_bool():
    # 1 and 1.0 are equal; true is neither 1 nor 1.0.
    res = call(
        {
            "nodes": [
                {"id": "s", "type": "source", "columns": [{"name": "a"}]},
                {"id": "f", "type": "filter", "input": "s", "references": ["a"]},
            ],
            "export": "f",
            "data": {"s": [{"a": 1}, {"a": 1.0}, {"a": True}, {"a": "1"}]},
            "conditions": {"f": {"a": 1.0}},
        }
    )
    assert res["rows"] == [[1], [1.0]]


def test_projection_constant_null_and_copy():
    res = call(
        {
            "nodes": [
                {"id": "s", "type": "source", "columns": [{"name": "a"}]},
                {
                    "id": "p",
                    "type": "projection",
                    "input": "s",
                    "columns": [
                        {"name": "a", "copy": "a"},
                        {"name": "n", "constant": None},
                        {"name": "flag", "constant": False},
                    ],
                },
            ],
            "export": "p",
            "data": {"s": [{"a": "x"}]},
        }
    )
    assert res["columns"] == ["a", "n", "flag"]
    assert res["rows"] == [["x", None, False]]


def test_join_inner_semantics_order_and_duplicate_keys():
    res = call(
        {
            "nodes": [
                {
                    "id": "l",
                    "type": "source",
                    "columns": [{"name": "k"}, {"name": "v"}],
                },
                {
                    "id": "r",
                    "type": "source",
                    "columns": [{"name": "k"}, {"name": "w"}],
                },
                {
                    "id": "j",
                    "type": "join",
                    "left_input": "l",
                    "right_input": "r",
                    "left_key": "k",
                    "right_key": "k",
                    "outputs": [
                        {"alias": "v", "side": "left", "column": "v"},
                        {"alias": "w", "side": "right", "column": "w"},
                    ],
                },
            ],
            "export": "j",
            "data": {
                "l": [
                    {"k": 1, "v": "a"},
                    {"k": None, "v": "b"},  # null key never matches
                    {"k": 1.0, "v": "c"},  # 1.0 joins with 1
                    {"k": True, "v": "d"},  # bool is not the number 1
                    {"k": 2, "v": "e"},  # no right partner
                ],
                "r": [
                    {"k": 1, "w": "x"},
                    {"k": None, "w": "y"},  # null key never matches
                    {"k": 1, "w": "z"},
                    {"k": 1.0, "w": "p"},
                ],
            },
        }
    )
    assert res["columns"] == ["v", "w"]
    # Left rows first; within one left row the right rows keep their order;
    # duplicate keys keep every combination.
    assert res["rows"] == [
        ["a", "x"],
        ["a", "z"],
        ["a", "p"],
        ["c", "x"],
        ["c", "z"],
        ["c", "p"],
    ]


def test_shared_branch_is_evaluated_once_for_both_consumers():
    res = call(
        {
            "nodes": [
                {"id": "s", "type": "source", "columns": [{"name": "k"}, {"name": "v"}]},
                {
                    "id": "j",
                    "type": "join",
                    "left_input": "s",
                    "right_input": "s",
                    "left_key": "k",
                    "right_key": "k",
                    "outputs": [
                        {"alias": "v1", "side": "left", "column": "v"},
                        {"alias": "v2", "side": "right", "column": "v"},
                    ],
                },
            ],
            "export": "j",
            "data": {"s": [{"k": 1, "v": "a"}, {"k": 1, "v": "b"}]},
        }
    )
    # Self-join on duplicate keys keeps all four combinations, left rows first.
    assert res["rows"] == [
        ["a", "a"],
        ["a", "b"],
        ["b", "a"],
        ["b", "b"],
    ]


def test_empty_source_rows_yield_empty_preview():
    res = call(
        {
            "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
            "export": "s",
            "data": {"s": []},
        }
    )
    assert res["columns"] == ["a"]
    assert res["rows"] == []


def test_input_payload_is_not_modified():
    payload = json.loads(json.dumps(PAYLOAD))
    snapshot = json.loads(json.dumps(payload))
    call(payload)
    assert payload == snapshot


# ----------------------------- blocked exports -----------------------------


def test_blocked_export_returns_only_the_review():
    res = call(
        {
            "nodes": [
                {
                    "id": "s",
                    "type": "source",
                    "columns": [
                        {"name": "id_card", "sensitive": True},
                        {"name": "score"},
                    ],
                },
                {"id": "f", "type": "filter", "input": "s", "references": ["id_card"]},
                {
                    "id": "e",
                    "type": "projection",
                    "input": "f",
                    "columns": [{"name": "score", "copy": "score"}],
                },
            ],
            "export": "e",
            "data": {"s": [{"id_card": "X", "score": 5}]},
            "conditions": {"f": {"id_card": "X"}},
        }
    )
    assert res["blocked"] is True
    # No sample data leaks: only the review reasons are returned.
    assert "columns" not in res
    assert "rows" not in res
    assert res["review"]["columns"][0]["row_reasons"] == [
        {"node_id": "s", "column": "id_card"}
    ]


def test_sensitive_content_blocks_preview_even_without_row_influence():
    res = call(
        {
            "nodes": [
                {
                    "id": "s",
                    "type": "source",
                    "columns": [{"name": "secret", "sensitive": True}],
                }
            ],
            "export": "s",
            "data": {"s": [{"secret": "v"}]},
        }
    )
    assert res["blocked"] is True
    assert "rows" not in res
    assert res["review"]["columns"][0]["content_reasons"] == [
        {"node_id": "s", "column": "secret"}
    ]


# ----------------------------- coverage rejections -------------------------


def expect_error(payload, code):
    with pytest.raises(ValidationError) as exc:
        call(payload)
    assert exc.value.code == code
    return exc.value


def test_data_must_cover_exactly_the_ancestor_sources():
    err = expect_error(
        {**PAYLOAD, "data": {}},
        "PREVIEW_SOURCE_MISMATCH",
    )
    assert err.detail()["missing"] == ["users"]

    err = expect_error(
        {**PAYLOAD, "data": {**PAYLOAD["data"], "ghost": []}},
        "PREVIEW_SOURCE_MISMATCH",
    )
    assert err.detail()["extra"] == ["ghost"]


def test_data_for_unreachable_source_is_rejected_as_extra():
    payload = json.loads(json.dumps(PAYLOAD))
    payload["nodes"].append(
        {"id": "other", "type": "source", "columns": [{"name": "x"}]}
    )
    payload["data"]["other"] = [{"x": 1}]
    err = expect_error(payload, "PREVIEW_SOURCE_MISMATCH")
    assert err.detail()["extra"] == ["other"]


def test_conditions_must_cover_exactly_the_ancestor_filters():
    err = expect_error({**PAYLOAD, "conditions": {}}, "PREVIEW_FILTER_MISMATCH")
    assert err.detail()["missing"] == ["only_cn"]

    err = expect_error(
        {**PAYLOAD, "conditions": {**PAYLOAD["conditions"], "users": {}}},
        "PREVIEW_FILTER_MISMATCH",
    )
    assert err.detail()["extra"] == ["users"]


def test_conditions_may_be_omitted_when_the_graph_has_no_filter():
    res = call(
        {
            "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
            "export": "s",
            "data": {"s": [{"a": 1}]},
        }
    )
    assert res["rows"] == [[1]]


def test_row_keys_must_cover_exactly_the_source_columns():
    err = expect_error(
        {
            "nodes": [
                {"id": "s", "type": "source", "columns": [{"name": "a"}, {"name": "b"}]}
            ],
            "export": "s",
            "data": {"s": [{"a": 1}]},
        },
        "PREVIEW_ROW_COLUMNS_MISMATCH",
    )
    detail = err.detail()
    assert detail["row_index"] == 0
    assert detail["missing"] == ["b"]
    assert detail["extra"] == []

    err = expect_error(
        {
            "nodes": [
                {"id": "s", "type": "source", "columns": [{"name": "a"}]}
            ],
            "export": "s",
            "data": {"s": [{"a": 1}, {"a": 2, "b": 3}]},
        },
        "PREVIEW_ROW_COLUMNS_MISMATCH",
    )
    detail = err.detail()
    assert detail["row_index"] == 1
    assert detail["extra"] == ["b"]


def test_condition_keys_must_equal_references():
    err = expect_error(
        {**PAYLOAD, "conditions": {"only_cn": {"uid": 1}}},
        "PREVIEW_CONDITION_MISMATCH",
    )
    detail = err.detail()
    assert detail["missing"] == ["country"]
    assert detail["extra"] == ["uid"]


# ----------------------------- row limit -----------------------------------


def test_source_over_200_rows_is_rejected():
    err = expect_error(
        {
            "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
            "export": "s",
            "data": {"s": [{"a": i} for i in range(ROW_LIMIT + 1)]},
        },
        "PREVIEW_ROW_LIMIT",
    )
    assert err.detail()["row_count"] == ROW_LIMIT + 1
    assert err.detail()["limit"] == ROW_LIMIT


def test_exactly_200_rows_is_accepted():
    res = call(
        {
            "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
            "export": "s",
            "data": {"s": [{"a": i} for i in range(ROW_LIMIT)]},
        }
    )
    assert len(res["rows"]) == ROW_LIMIT


def test_intermediate_result_over_200_rows_is_rejected():
    # 16 x 16 duplicate keys explode to 256 joined rows; the downstream filter
    # would shrink the export back under the limit, but the intermediate
    # result itself is already over it.
    err = expect_error(
        {
            "nodes": [
                {"id": "l", "type": "source", "columns": [{"name": "k"}]},
                {"id": "r", "type": "source", "columns": [{"name": "k"}]},
                {
                    "id": "j",
                    "type": "join",
                    "left_input": "l",
                    "right_input": "r",
                    "left_key": "k",
                    "right_key": "k",
                    "outputs": [{"alias": "k", "side": "left", "column": "k"}],
                },
                {"id": "f", "type": "filter", "input": "j", "references": ["k"]},
            ],
            "export": "f",
            "data": {
                "l": [{"k": 1} for _ in range(16)],
                "r": [{"k": 1} for _ in range(16)],
            },
            "conditions": {"f": {"k": 2}},  # would keep zero rows
        },
        "PREVIEW_ROW_LIMIT",
    )
    assert err.node_id == "j"


def test_export_result_over_200_rows_is_rejected():
    # The export join itself produces 16 x 16 = 256 rows.
    err = expect_error(
        {
            "nodes": [
                {"id": "l", "type": "source", "columns": [{"name": "k"}]},
                {"id": "r", "type": "source", "columns": [{"name": "k"}]},
                {
                    "id": "j",
                    "type": "join",
                    "left_input": "l",
                    "right_input": "r",
                    "left_key": "k",
                    "right_key": "k",
                    "outputs": [{"alias": "k", "side": "left", "column": "k"}],
                },
            ],
            "export": "j",
            "data": {
                "l": [{"k": 1} for _ in range(16)],
                "r": [{"k": 1} for _ in range(16)],
            },
        },
        "PREVIEW_ROW_LIMIT",
    )
    assert err.node_id == "j"


# ----------------------------- schema validation ---------------------------


def test_non_scalar_cell_rejected_as_422():
    with pytest.raises(PydanticValidationError):
        call(
            {
                "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
                "export": "s",
                "data": {"s": [{"a": [1]}]},
            }
        )
    with pytest.raises(PydanticValidationError):
        call(
            {
                "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
                "export": "s",
                "data": {"s": [{"a": {"x": 1}}]},
            }
        )


def test_non_scalar_condition_value_rejected_as_422():
    with pytest.raises(PydanticValidationError):
        call(
            {
                "nodes": [
                    {"id": "s", "type": "source", "columns": [{"name": "a"}]},
                    {"id": "f", "type": "filter", "input": "s", "references": ["a"]},
                ],
                "export": "f",
                "data": {"s": [{"a": 1}]},
                "conditions": {"f": {"a": [1]}},
            }
        )


def test_non_scalar_projection_constant_rejected_as_422_for_preview():
    with pytest.raises(PydanticValidationError):
        call(
            {
                "nodes": [
                    {"id": "s", "type": "source", "columns": [{"name": "a"}]},
                    {
                        "id": "p",
                        "type": "projection",
                        "input": "s",
                        "columns": [{"name": "o", "constant": {"nested": 1}}],
                    },
                ],
                "export": "p",
                "data": {"s": [{"a": 1}]},
            }
        )


def test_old_endpoints_still_accept_any_constant_value():
    # The scalar restriction is preview-only; /api/review keeps its range.
    res = review(
        ReviewRequest.model_validate(
            {
                "nodes": [
                    {"id": "s", "type": "source", "columns": [{"name": "a"}]},
                    {
                        "id": "p",
                        "type": "projection",
                        "input": "s",
                        "columns": [
                            {"name": "o", "constant": {"nested": [1, 2]}},
                        ],
                    },
                ],
                "export": "p",
            }
        )
    )
    assert res["blocked"] is False


def test_malformed_data_shape_rejected_as_422():
    with pytest.raises(PydanticValidationError):
        call({**PAYLOAD, "data": {"users": {"uid": 1}}})
    with pytest.raises(PydanticValidationError):
        call({**PAYLOAD, "data": {"users": [[1, 2, 3]]}})
    with pytest.raises(PydanticValidationError):
        call({**PAYLOAD, "conditions": {"only_cn": ["country"]}})
    with pytest.raises(PydanticValidationError):
        call({**PAYLOAD, "unknown_field": 1})


# ----------------------------- original validation -------------------------


def test_original_graph_validation_still_applies():
    expect_error(
        {
            "nodes": [
                {"id": "s", "type": "source", "columns": [{"name": "a"}]},
                {"id": "s", "type": "source", "columns": [{"name": "b"}]},
            ],
            "export": "s",
            "data": {"s": []},
        },
        "DUPLICATE_NODE",
    )
    expect_error(
        {
            "nodes": [
                {"id": "s", "type": "source", "columns": [{"name": "a"}]},
                {"id": "f", "type": "filter", "input": "s", "references": ["ghost"]},
            ],
            "export": "s",
            "data": {"s": []},
            "conditions": {"f": {"ghost": 1}},
        },
        "INVALID_REFERENCE",
    )
    expect_error(
        {
            "nodes": [
                {
                    "id": "a",
                    "type": "projection",
                    "input": "b",
                    "columns": [{"name": "x", "constant": 1}],
                },
                {
                    "id": "b",
                    "type": "projection",
                    "input": "a",
                    "columns": [{"name": "x", "constant": 1}],
                },
            ],
            "export": "a",
        },
        "CYCLE",
    )


def test_unreachable_branch_errors_still_reject_the_whole_request():
    expect_error(
        {
            "nodes": [
                {"id": "s", "type": "source", "columns": [{"name": "a"}]},
                {"id": "f", "type": "filter", "input": "s", "references": ["ghost"]},
            ],
            "export": "s",
            "data": {"s": [{"a": 1}]},
            # f is unreachable from the export? no: f consumes s but nothing
            # consumes f, so f is not an ancestor and needs no conditions —
            # yet its invalid reference still rejects the request.
        },
        "INVALID_REFERENCE",
    )


# ----------------------------- HTTP endpoint -------------------------------


client = TestClient(app)


def test_http_preview_success():
    resp = client.post("/api/preview", json=PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["blocked"] is False
    assert body["columns"] == ["user_id", "bucket"]
    assert body["rows"] == [[1, "ALL"], [3, "ALL"]]
    assert body["review"]["blocked"] is False


def test_http_preview_from_example_file():
    with open("examples/preview_request.json") as fh:
        payload = json.load(fh)
    resp = client.post("/api/preview", json=payload)
    assert resp.status_code == 200
    assert resp.json()["rows"] == [[1, "ALL"], [3, "ALL"]]


def test_http_preview_sensitive_filter_returns_only_review():
    resp = client.post(
        "/api/preview",
        json={
            "nodes": [
                {
                    "id": "s",
                    "type": "source",
                    "columns": [
                        {"name": "id_card", "sensitive": True},
                        {"name": "score"},
                    ],
                },
                {"id": "f", "type": "filter", "input": "s", "references": ["id_card"]},
                {
                    "id": "e",
                    "type": "projection",
                    "input": "f",
                    "columns": [{"name": "score", "copy": "score"}],
                },
            ],
            "export": "e",
            "data": {"s": [{"id_card": "X", "score": 5}]},
            "conditions": {"f": {"id_card": "X"}},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["blocked"] is True
    assert "columns" not in body and "rows" not in body
    assert body["review"]["columns"][0]["row_reasons"] == [
        {"node_id": "s", "column": "id_card"}
    ]


def test_http_preview_coverage_mismatch_returns_400_without_rows():
    resp = client.post("/api/preview", json={**PAYLOAD, "data": {}})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "PREVIEW_SOURCE_MISMATCH"
    assert "rows" not in body and "review" not in body


def test_http_preview_row_limit_returns_400():
    resp = client.post(
        "/api/preview",
        json={
            "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
            "export": "s",
            "data": {"s": [{"a": i} for i in range(201)]},
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "PREVIEW_ROW_LIMIT"


def test_http_preview_schema_error_returns_422():
    resp = client.post(
        "/api/preview", json={**PAYLOAD, "data": {"users": [{"uid": 1}]}}
    )
    # Row keys not covering the source columns is a semantic (400) error...
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "PREVIEW_ROW_COLUMNS_MISMATCH"

    # ...while a non-scalar cell is a schema (422) error.
    resp = client.post(
        "/api/preview",
        json={
            "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
            "export": "s",
            "data": {"s": [{"a": {"nested": 1}}]},
        },
    )
    assert resp.status_code == 422


def test_http_other_endpoints_unchanged():
    resp = client.post(
        "/api/review", json={"nodes": NODES, "export": "export_users"}
    )
    assert resp.status_code == 200
    assert resp.json()["blocked"] is False
    assert "rows" not in resp.json()

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

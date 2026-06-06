import llama_bench.hf_gguf as hf_gguf


def test_split_gguf_path_extracts_quant_and_shard_info() -> None:
    assert hf_gguf.split_gguf_path("nested/Model-Q4_K_M-00002-of-00003.gguf") == (
        "nested/Model-Q4_K_M",
        "Q4_K_M",
        2,
        3,
    )
    assert hf_gguf.split_gguf_path("Model.UD-Q6_K.gguf") == ("Model.UD-Q6_K", "UD-Q6_K", 1, 1)
    assert hf_gguf.split_gguf_path("Model") == ("Model", "", 1, 1)


def test_split_gguf_path_handles_google_qat_underscore_quant() -> None:
    assert hf_gguf.split_gguf_path("gemma-4-E2B_q4_0-it.gguf") == (
        "gemma-4-E2B_q4_0-it",
        "Q4_0",
        1,
        1,
    )
    assert hf_gguf.split_gguf_path("gemma-4-E4B_q4_0-it.gguf") == (
        "gemma-4-E4B_q4_0-it",
        "Q4_0",
        1,
        1,
    )
    assert hf_gguf.split_gguf_path("gemma-4-26B_q4_0-it.gguf") == (
        "gemma-4-26B_q4_0-it",
        "Q4_0",
        1,
        1,
    )
    assert hf_gguf.split_gguf_path("gemma-4-12b-it-qat-q4_0.gguf") == (
        "gemma-4-12b-it-qat-q4_0",
        "Q4_0",
        1,
        1,
    )


def test_find_matching_model_files_excludes_auxiliary_files_and_sorts_shards() -> None:
    repo_files = [
        "model-Q4_K_M-00002-of-00002.gguf",
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q5_K_M.gguf",
        "mmproj-model-Q4_K_M.gguf",
        "model-Q4_K_M.imatrix.gguf",
        "README.md",
    ]

    assert hf_gguf.find_matching_model_files(repo_files, "q4_k_m") == [
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
    ]


def test_is_model_file_accepts_only_non_auxiliary_ggufs() -> None:
    assert hf_gguf.is_model_file("model-Q4_K_M.gguf") is True
    assert hf_gguf.is_model_file("nested/model-Q4_K_M.gguf") is True
    assert hf_gguf.is_model_file("mmproj-model-Q4_K_M.gguf") is False
    assert hf_gguf.is_model_file("model-Q4_K_M.imatrix.gguf") is False
    assert hf_gguf.is_model_file("README.md") is False


def test_find_matching_model_files_returns_empty_for_no_or_auxiliary_only_matches() -> None:
    assert hf_gguf.find_matching_model_files(["model-Q5_K_M.gguf"], "Q4_K_M") == []
    assert hf_gguf.find_matching_model_files(
        ["mmproj-model-Q4_K_M.gguf", "model-Q4_K_M.imatrix.gguf"], "Q4_K_M"
    ) == []


def test_find_matching_model_files_orders_mixed_sharded_and_unsharded_files() -> None:
    repo_files = [
        "other-Q4_K_M-00002-of-00002.gguf",
        "model-Q4_K_M.gguf",
        "other-Q4_K_M-00001-of-00002.gguf",
        "zmodel-Q4_K_M.gguf",
    ]

    assert hf_gguf.find_matching_model_files(repo_files, "Q4_K_M") == [
        "model-Q4_K_M.gguf",
        "other-Q4_K_M-00001-of-00002.gguf",
        "zmodel-Q4_K_M.gguf",
        "other-Q4_K_M-00002-of-00002.gguf",
    ]


def test_find_best_mmproj_file_prefers_deepest_matching_directory_then_closest_quant() -> None:
    repo_files = [
        "mmproj-Q8_0.gguf",
        "sub/mmproj-Q2_K.gguf",
        "sub/mmproj-Q5_K.gguf",
        "sub/deeper/mmproj-Q4_K.gguf",
        "other/mmproj-Q4_K.gguf",
        "sub/model-Q4_K_M.gguf",
    ]

    assert hf_gguf.find_best_mmproj_file(repo_files, "sub/model-Q4_K_M.gguf") == (
        "sub/mmproj-Q5_K.gguf"
    )
    assert hf_gguf.find_best_mmproj_file(repo_files, "missing/model-Q4_K_M.gguf") == "mmproj-Q8_0.gguf"


def test_find_best_mmproj_file_handles_absent_sibling_child_and_tied_projectors() -> None:
    assert hf_gguf.find_best_mmproj_file(["model-Q4_K_M.gguf"], "model-Q4_K_M.gguf") is None

    repo_files = [
        "mmproj-Q2_K.gguf",
        "mmproj-Q6_K.gguf",
        "sub/child/mmproj-Q4_K.gguf",
        "sub/mmproj-Q8_0.gguf",
        "sub/model-Q4_K_M.gguf",
    ]

    assert hf_gguf.find_best_mmproj_file(repo_files, "sub/model-Q4_K_M.gguf") == (
        "sub/mmproj-Q8_0.gguf"
    )


def test_split_gguf_path_handles_bpw_suffixed_quant_tags() -> None:
    assert hf_gguf.split_gguf_path("Qwen3.6-35B-A3B-IQ2_S-2.17bpw.gguf") == (
        "Qwen3.6-35B-A3B-IQ2_S-2.17bpw",
        "IQ2_S-2.17BPW",
        1,
        1,
    )
    assert hf_gguf.split_gguf_path("Qwen3.5-9B-IQ4_XS-4.43bpw.gguf") == (
        "Qwen3.5-9B-IQ4_XS-4.43bpw",
        "IQ4_XS-4.43BPW",
        1,
        1,
    )
    assert hf_gguf.split_gguf_path("Qwen3.5-9B-Q5_K_S-5.10bpw.gguf") == (
        "Qwen3.5-9B-Q5_K_S-5.10bpw",
        "Q5_K_S-5.10BPW",
        1,
        1,
    )
    assert hf_gguf.split_gguf_path("Qwen3.6-35B-A3B-IQ3_S-3.00bpw-00001-of-00002.gguf") == (
        "Qwen3.6-35B-A3B-IQ3_S-3.00bpw",
        "IQ3_S-3.00BPW",
        1,
        2,
    )


def test_find_matching_model_files_matches_bpw_suffixed_files() -> None:
    repo_files = [
        "Qwen3.6-35B-A3B-IQ2_S-2.17bpw.gguf",
        "Qwen3.6-35B-A3B-IQ3_S-3.00bpw.gguf",
        "Qwen3.6-35B-A3B-IQ4_XS-4.15bpw.gguf",
        "mmproj-Qwen3.6-35B-A3B.gguf",
    ]

    assert hf_gguf.find_matching_model_files(repo_files, "IQ3_S-3.00bpw") == [
        "Qwen3.6-35B-A3B-IQ3_S-3.00bpw.gguf",
    ]
    assert hf_gguf.find_matching_model_files(repo_files, "IQ2_S-2.17bpw") == [
        "Qwen3.6-35B-A3B-IQ2_S-2.17bpw.gguf",
    ]


def test_split_gguf_path_handles_apex_quant_tags() -> None:
    assert hf_gguf.split_gguf_path("Qwen3.6-35B-A3B-APEX-I-Compact.gguf") == (
        "Qwen3.6-35B-A3B-APEX-I-Compact",
        "APEX-I-COMPACT",
        1,
        1,
    )
    assert hf_gguf.split_gguf_path("Qwen3.6-35B-A3B-APEX-I-Balanced.gguf") == (
        "Qwen3.6-35B-A3B-APEX-I-Balanced",
        "APEX-I-BALANCED",
        1,
        1,
    )
    assert hf_gguf.split_gguf_path("Qwen3.6-35B-A3B-APEX-Compact.gguf") == (
        "Qwen3.6-35B-A3B-APEX-Compact",
        "APEX-COMPACT",
        1,
        1,
    )


def test_find_matching_model_files_matches_apex_files() -> None:
    repo_files = [
        "Qwen3.6-35B-A3B-APEX-I-Compact.gguf",
        "Qwen3.6-35B-A3B-APEX-I-Quality.gguf",
        "Qwen3.6-35B-A3B-APEX-I-Balanced.gguf",
        "mmproj.gguf",
    ]

    assert hf_gguf.find_matching_model_files(repo_files, "APEX-I-Compact") == [
        "Qwen3.6-35B-A3B-APEX-I-Compact.gguf",
    ]


def test_find_matching_model_files_matches_google_qat_underscore_quant() -> None:
    repo_files = [
        "gemma-4-E2B_q4_0-it.gguf",
        "gemma-4-E2B-it-mmproj.gguf",
        "README.md",
    ]

    assert hf_gguf.find_matching_model_files(repo_files, "Q4_0") == [
        "gemma-4-E2B_q4_0-it.gguf",
    ]


def test_extract_quant_bits_returns_first_number_from_quant_tag() -> None:
    assert hf_gguf.extract_quant_bits("model-Q4_K_M.gguf") == 4
    assert hf_gguf.extract_quant_bits("model-UD-IQ2_XXS.gguf") == 2
    assert hf_gguf.extract_quant_bits("model.gguf") == 0
